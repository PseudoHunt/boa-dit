"""Stage 1 / 4.2-4.5 in one FP sampling pass over the Stage 1 seeds:

4.2 gate energies per (block, gate, branch, step, channel) -> gates.pt, gates_mean.png, gates_channels.png
4.3 Gram trace sum ||x||^2 per step for qkv/proj/fc1/fc2 inputs of blocks 0,13,27 -> trace.png
4.4 per-bucket Grams (branch weights folded in) -> dissimilarity metrics -> gram_dissim.json/.png
4.5 per-token ||x||^2 distribution per (layer, bucket) -> token_norms.json
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json, atomic_savefig
from boa_dit.model import LINEARS, block_linear, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.probe_gain import STAGE1_SEEDS
from boa_dit.sample import CTX, ddim_sample

D = 1152
PROBE_BLOCKS = (0, 13, 27)
GATE_NAMES = ("msa", "mlp")
GATE_CHUNK = {"msa": 2, "mlp": 5}  # index into adaLN_modulation(c).chunk(6)
OUT = os.path.join(RESULTS, "stage1")


def bucket_of(step, steps=50, B=10):
    return step // (steps // B)


def kmeans_inertia_ratio(curves, K=4, seed=0):
    """curves: [n_ch, T] normalized to unit mean. within-cluster SS / total SS."""
    from sklearn.cluster import KMeans
    if len(curves) <= K:
        return float("nan")
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(curves)
    total = float(((curves - curves.mean(0, keepdims=True)) ** 2).sum())
    return float(km.inertia_ / total) if total > 0 else float("nan")


@torch.no_grad()
def collect(model, diffusion, seeds, cfg, dev, B=10):
    S = diffusion.num_timesteps
    gate_sum = torch.zeros(28, 2, 2, S, D, device=dev)       # [block, gate, branch(cond,uncond), step, ch]
    gate_cnt = torch.zeros(2, S, device=dev)
    trace = {(b, l): torch.zeros(2, S, device=dev) for b in PROBE_BLOCKS for l in LINEARS}
    grams = {}
    tok = {(b, l): [[None] * S for _ in range(2)] for b in PROBE_BLOCKS for l in LINEARS}
    s2, u2 = cfg ** 2, (cfg - 1) ** 2
    hooks = []

    def gate_hook(blk_idx):
        @torch.autocast("cuda", enabled=False)  # hooks fire inside the fp16 autocast region
        def h(mod, inp, out):
            n, i = CTX.n, CTX.step
            o = out.float()
            for gi, g in enumerate(GATE_NAMES):
                c = GATE_CHUNK[g]
                e = o[:, c * D:(c + 1) * D].pow(2)
                gate_sum[blk_idx, gi, 0, i] += e[:n].sum(0)
                gate_sum[blk_idx, gi, 1, i] += e[n:].sum(0)
            if blk_idx == 0:
                gate_cnt[0, i] += n
                gate_cnt[1, i] += n
        return h

    def lin_hook(blk_idx, lname):
        @torch.autocast("cuda", enabled=False)  # Grams must accumulate in fp32
        def h(mod, inp):
            n, i = CTX.n, CTX.step
            x = inp[0].float()                                 # [2n, 256, d]
            norms = x.pow(2).sum(-1)                           # [2n, 256]
            trace[(blk_idx, lname)][0, i] += norms[:n].sum()
            trace[(blk_idx, lname)][1, i] += norms[n:].sum()
            tok[(blk_idx, lname)][0][i] = norms[:n].cpu()
            tok[(blk_idx, lname)][1][i] = norms[n:].cpu()
            xc, xu = x[:n].reshape(-1, x.shape[-1]), x[n:].reshape(-1, x.shape[-1])
            key = (blk_idx, lname, bucket_of(i, S, B))
            G = s2 * (xc.T @ xc) + u2 * (xu.T @ xu)
            assert G.dtype == torch.float32
            grams[key] = G if key not in grams else grams[key] + G
        return h

    for bi, blk in enumerate(model.blocks):
        hooks.append(blk.adaLN_modulation.register_forward_hook(gate_hook(bi)))
    for bi in PROBE_BLOCKS:
        for l in LINEARS:
            hooks.append(block_linear(model.blocks[bi], l).register_forward_pre_hook(lin_hook(bi, l)))
    try:
        ddim_sample(model, diffusion, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), cfg)
    finally:
        for h in hooks:
            h.remove()
    gates = (gate_sum / gate_cnt.view(1, 1, 2, S, 1)).cpu()   # mean over samples of gate^2
    return gates, {k: v.cpu() for k, v in trace.items()}, grams, tok


def gate_analysis(gates):
    """gates [28,2,2,S,D]; returns per-block stats and the branch-averaged energies [28,2,S,D]."""
    g = gates.mean(2)                                          # branch-averaged [28, 2, S, D]
    out = {}
    for gi, gname in enumerate(GATE_NAMES):
        rows = []
        for b in range(28):
            curve = g[b, gi]                                   # [S, D]
            mean_t = curve.mean(1).numpy()
            dead = (curve < 1e-4).all(0)
            alive = curve[:, ~dead].T.numpy()                  # [n_alive, S]
            norm = alive / np.maximum(alive.mean(1, keepdims=True), 1e-30)
            rows.append({"block": b, "max_over_median_t": float(mean_t.max() / np.median(mean_t)),
                         "argmax_step": int(mean_t.argmax()), "argmin_step": int(mean_t.argmin()),
                         "dead_frac": float(dead.float().mean()),
                         "kmeans4_inertia_ratio": kmeans_inertia_ratio(norm, 4)})
        out[gname] = rows
    return out, g


def gram_dissim(grams, B, dev, k=32, n_delta=256, seed=0):
    from scipy.stats import spearmanr
    res = {}
    gen = torch.Generator(device="cpu").manual_seed(seed)
    for bi in PROBE_BLOCKS:
        for l in LINEARS:
            Hs = [grams[(bi, l, b)].to(dev) for b in range(B)]
            d = Hs[0].shape[0]
            Hn = [H / torch.trace(H) for H in Hs]
            fro = np.zeros((B, B))
            ovl = np.zeros((B, B))
            sp = np.ones((B, B))
            # CPU LAPACK in float64: cusolver's eigh fails to converge on some of these Grams
            eig = [torch.linalg.eigh(H.double().cpu()) for H in Hn]
            U = [e[1][:, -k:].float().to(dev) for e in eig]
            delta = (torch.randint(0, 2, (n_delta, d), generator=gen).float() * 2 - 1).to(dev)
            quad = [((delta @ H) * delta).sum(1).cpu().numpy() for H in Hn]
            for a in range(B):
                for b in range(B):
                    fro[a, b] = float(torch.linalg.norm(Hn[a] - Hn[b]) / torch.linalg.norm(Hn[b]))
                    ovl[a, b] = float((U[a].T @ U[b]).pow(2).sum() / k)
                    if a != b:
                        sp[a, b] = float(spearmanr(quad[a], quad[b]).statistic)
            off = ~np.eye(B, dtype=bool)
            # share of the trace in the rank-one mean direction (diagnostic for "dominated by the mean")
            top1 = [float(e[0][-1]) for e in eig]
            res[f"b{bi}.{l}"] = {"block": bi, "layer": l, "d": d, "rel_fro": fro.tolist(), "eig_overlap_top32": ovl.tolist(),
                                 "spearman_quad": sp.tolist(),
                                 "median_offdiag_rel_fro": float(np.median(fro[off])),
                                 "min_spearman": float(sp[off].min()), "median_spearman": float(np.median(sp[off])),
                                 "median_offdiag_overlap": float(np.median(ovl[off])),
                                 "top_eig_share_per_bucket": top1, "raw_trace_per_bucket": [float(torch.trace(H)) for H in Hs]}
            del Hs, Hn, U, eig
            torch.cuda.empty_cache()
    return res


def token_norm_analysis(tok, S, B):
    res = {}
    for (bi, l), per_branch in tok.items():
        rows = []
        for b in range(B):
            steps = range(b * (S // B), (b + 1) * (S // B))
            # [n_img, 256] per step and branch -> stack: [2, steps, n, 256]
            arr = torch.stack([torch.stack([per_branch[br][s] for s in steps]) for br in range(2)])
            flat = arr.flatten()
            med = float(flat.median())
            mx = float(flat.max())
            r = {"bucket": b, "max": mx, "p99": float(torch.quantile(flat.float(), 0.99)), "median": med,
                 "max_over_median": mx / med}
            if mx / med > 100:
                big = arr > 100 * med                       # [2, steps, n, 256]
                pos_count = big.sum((0, 1, 2))              # per grid position
                top = torch.argsort(pos_count, descending=True)[:8]
                r["sink_positions_top8"] = [[int(p), int(pos_count[p])] for p in top if pos_count[p] > 0]
                r["sink_tokens_total"] = int(big.sum())
                r["sink_images_with_any"] = int(big.any(-1).any(0).any(0).sum())
                r["sink_steps_with_any"] = int(big.any(-1).any(-1).any(0).sum())
            rows.append(r)
        res[f"b{bi}.{l}"] = {"block": bi, "layer": l, "buckets": rows,
                             "max_over_median_worst": max(r["max_over_median"] for r in rows)}
    return res


def plots(g, gates_stats, trace, dissim, S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    steps = np.arange(S)
    # gates_mean
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    cmap = plt.get_cmap("viridis")
    for gi, gname in enumerate(GATE_NAMES):
        for b in range(28):
            axs[gi].plot(steps, g[b, gi].mean(1).numpy(), color=cmap(b / 27), lw=1)
        axs[gi].set_yscale("log"); axs[gi].set_title(f"gate_{gname}: mean_j gate² vs step (color = block 0→27)")
        axs[gi].set_xlabel("sampling step i (0 = noisiest)")
    axs[0].set_ylabel("mean gate²")
    atomic_savefig(fig, os.path.join(OUT, "gates_mean.png")); plt.close(fig)
    # gates_channels
    fig, axs = plt.subplots(3, 3, figsize=(13, 10))
    for r, b in enumerate(PROBE_BLOCKS):
        for c, s in enumerate((5, 25, 45)):
            ax = axs[r, c]
            for gi, gname in enumerate(GATE_NAMES):
                v = g[b, gi, s].clamp_min(1e-10).log10().numpy()
                ax.hist(v, bins=60, alpha=0.6, label=f"gate_{gname}")
            ax.set_title(f"block {b}, step {s}"); ax.set_xlabel("log10 gate² per channel")
            if r == 0 and c == 0:
                ax.legend()
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "gates_channels.png")); plt.close(fig)
    # trace
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    for ci, b in enumerate(PROBE_BLOCKS):
        for l in LINEARS:
            t = trace[(b, l)].sum(0).numpy()
            axs[ci].plot(steps, t / t.mean(), label=l)
        axs[ci].set_yscale("log"); axs[ci].set_title(f"block {b}: Σ‖x‖² per step / its mean")
        axs[ci].set_xlabel("sampling step i")
    axs[0].legend()
    atomic_savefig(fig, os.path.join(OUT, "trace.png")); plt.close(fig)
    # gram dissim heatmaps
    fig, axs = plt.subplots(3, 4, figsize=(16, 11.5))
    for r, b in enumerate(PROBE_BLOCKS):
        for c, l in enumerate(LINEARS):
            m = np.array(dissim[f"b{b}.{l}"]["rel_fro"])
            im = axs[r, c].imshow(m, cmap="magma", vmin=0)
            axs[r, c].set_title(f"b{b} {l}\nmed {np.median(m[~np.eye(10, dtype=bool)]):.3f}  "
                                f"minρ {dissim[f'b{b}.{l}']['min_spearman']:.3f}", fontsize=9)
            axs[r, c].set_xlabel("bucket b'"); axs[r, c].set_ylabel("bucket b")
            fig.colorbar(im, ax=axs[r, c], fraction=0.046)
    fig.suptitle("‖Ĥ_b − Ĥ_b'‖_F / ‖Ĥ_b'‖_F (unit-trace per-bucket Grams; bucket 0 = noisiest)")
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "gram_dissim.png")); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--reuse", action="store_true", help="reuse ckpt/stage1_collect.pt")
    args = ap.parse_args()
    dev, S, B = args.device, args.steps, args.buckets
    model, diffusion = load_dit(dev), make_diffusion(S)
    with Run("stage1", "probe_gates", dev, vars(args)) as run:
        cache = os.path.join(CKPT_DIR, "stage1_collect.pt")
        if args.reuse and os.path.exists(cache):
            gates, trace, grams, tok = torch.load(cache, map_location="cpu")
        else:
            gates, trace, grams, tok = collect(model, diffusion, STAGE1_SEEDS, args.cfg, dev, B)
            atomic_torch_save((gates, trace, {k: v.cpu() for k, v in grams.items()}, tok), cache)
        del model
        torch.cuda.empty_cache()
        atomic_torch_save({"gates": gates, "dims": "[block 28, gate (msa,mlp), branch (cond,uncond), step 50, channel 1152]",
                           "seeds": STAGE1_SEEDS, "value": "mean over samples of gate^2",
                           "per_step_mean_over_channels": gates.mean(-1)}, os.path.join(OUT, "gates.pt"))
        gstats, g = gate_analysis(gates)
        dissim = gram_dissim(grams, B, dev)
        tnorm = token_norm_analysis(tok, S, B)
        plots(g, gstats, trace, dissim, S)
        rec = run.record()
    atomic_write_json(os.path.join(OUT, "gates_stats.json"), {"per_block": gstats, **rec})
    atomic_write_json(os.path.join(OUT, "trace.json"), {f"b{b}.{l}": v.tolist() for (b, l), v in trace.items()} | {"dims": "[branch(cond,uncond), step]"})
    atomic_write_json(os.path.join(OUT, "gram_dissim.json"), dissim | {"_run": rec})
    atomic_write_json(os.path.join(OUT, "token_norms.json"), tnorm | {"_run": rec})
    for k, v in dissim.items():
        print(k, f"medFro {v['median_offdiag_rel_fro']:.3f} minSp {v['min_spearman']:.3f} medOvl {v['median_offdiag_overlap']:.3f}")
    for k, v in tnorm.items():
        print(k, f"max/med worst {v['max_over_median_worst']:.1f}")


if __name__ == "__main__":
    main()
