"""Gate 1 (attention sharing): verdict + plots from results/attn_share/gate1_stats.json and gate1_examples.pt."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import atomic_savefig, atomic_write_json, atomic_write_text  # noqa: E402

OUT = os.path.join(ROOT, "results", "attn_share")
MARK = "\n---\n\n## Results (appended by `scripts/make_attn_gate1.py`)\n"
EX_HEAD = 0


def verdict(st):
    w = np.array(st["within_median"])                 # [S, L, H]
    kr = np.array(st["kl_ratio_median"])
    frac = (w >= 0.8).reshape(w.shape[0], -1).mean(1)  # per step
    early = slice(0, 10)
    agree_steps = bool((frac[early] >= 0.7).all())
    pooled = float((w[early] >= 0.8).mean())
    clear = w[early] >= 0.8
    kl_share = float((kr[early][clear] >= 3).mean()) if clear.any() else 0.0
    ok = np.where(frac >= 0.7)[0]
    s_max = int(ok.max()) if len(ok) else -1
    contig = -1
    for s in range(len(frac)):
        if frac[s] < 0.7:
            break
        contig = s
    passed = agree_steps and kl_share >= 0.5
    if passed:
        v = "PASS"
    elif agree_steps:
        v = "STOP (agreement high but structure absent: early attention is near-uniform)"
    else:
        v = "STOP (cross-image agreement below the bar)"
    return {"frac_pairs_within_ge_0.8_per_step": frac.tolist(), "agreement_every_step_0_9": agree_steps,
            "pooled_frac_s0_9": pooled, "pooled_reading_passes": pooled >= 0.7,
            "kl_ratio_ge3_share_among_clearing_s0_9": kl_share,
            "kl_ratio_loo_ge3_share_among_clearing_s0_9":
                float((np.array(st["kl_ratio_loo_median"])[early][clear] >= 3).mean()) if clear.any() else 0.0,
            "s_max": s_max, "s_max_contiguous": contig, "verdict": v}


def plots(d, st):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    S = len(d["steps"])
    fig, axs = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (k, lab, kw) in zip(axs, [("within_median", "median within (head mean)", dict(vmin=0, vmax=1)),
                                      ("ent_median", "median row entropy / log256 (head mean)", dict(vmin=0, vmax=1)),
                                      ("kl_ratio_median", "log10 KL ratio (U / M), head mean of log", {})]):
        a = np.array(st[k])
        a = np.log10(a).mean(-1) if k == "kl_ratio_median" else a.mean(-1)
        im = ax.imshow(a.T, aspect="auto", origin="lower", cmap="viridis", **kw)
        ax.set_xlabel("step s"); ax.set_ylabel("block"); ax.set_title(lab); fig.colorbar(im, ax=ax)
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "gate1_heatmaps.png")); plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 4.5), sharey=True)
    w = np.array(st["within_median"])
    for ax, b in zip(axs, (0, 13, 27)):
        for h in range(w.shape[2]):
            ax.plot(range(S), w[:, b, h], lw=1)
        ax.axhline(0.8, color="k", ls=":"); ax.set_title(f"block {b}: 16 heads"); ax.set_xlabel("step s")
    axs[0].set_ylabel("median within"); fig.tight_layout()
    atomic_savefig(fig, os.path.join(OUT, "gate1_heads.png")); plt.close(fig)

    sc, dc = np.array(st["within_sameclass"])[:10].mean((0, 2)), np.array(st["within_diffclass"])[:10].mean((0, 2))
    fig, ax = plt.subplots(figsize=(13, 4))
    x = np.arange(len(sc))
    ax.bar(x - 0.2, sc, 0.4, label="same class, different noise"); ax.bar(x + 0.2, dc, 0.4, label="different class")
    ax.set_xlabel("block"); ax.set_ylabel("median within (mean over heads, s 0–9)"); ax.set_ylim(0, 1); ax.legend()
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "gate1_class_vs_noise.png")); plt.close(fig)

    ex = torch.load(os.path.join(OUT, "gate1_examples.pt"), map_location="cpu")
    Mc = torch.load(os.path.join(OUT, "maps", "M_cond.pt"), map_location="cpu", mmap=True)
    b, pos = 13, 136
    fig, axs = plt.subplots(4, 3, figsize=(10, 13))
    for c, s in enumerate((2, 10, 20)):
        M = Mc[s, b, EX_HEAD].float()
        axs[0, c].imshow(M.numpy(), cmap="magma"); axs[0, c].set_title(f"M, block {b} head {EX_HEAD}, s={s}")
        axs[1, c].imshow(M[pos].reshape(16, 16).numpy(), cmap="magma"); axs[1, c].set_title(f"M row {pos}")
        for r, m in ((2, 0), (3, 1)):
            p = ex[f"b{b}.s{s}"][m, EX_HEAD, 1].float()
            axs[r, c].imshow(p.reshape(16, 16).numpy(), cmap="magma")
            axs[r, c].set_title(f"image {(0, 2)[m]} (class {d['labels'][(0, 2)[m]]}) row {pos}")
    for a in axs.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "gate1_examples.png")); plt.close(fig)


def main():
    d = json.load(open(os.path.join(OUT, "gate1_stats.json")))
    st = d["stats"]
    v = verdict(st)
    atomic_write_json(os.path.join(OUT, "gate1_verdict.json"), v)
    plots(d, st)
    early = slice(0, 10)
    A = {k: np.array(x) for k, x in st.items()}
    L = [MARK]
    L.append(f"Parity: explicit-softmax vs fused block outputs, max relative difference "
             f"{d['parity']['explicit_vs_fused_max_rel']:.2e} (tolerance 1e-3). The share path fed one image's own "
             f"maps reproduces the explicit path to {d['parity']['share_own_maps_vs_explicit_max_rel']:.2e}. "
             f"Classes: {d['classes']}.\n")
    L.append("### Per step (cond branch; medians over the 448 (block, head) pairs unless noted)\n")
    L.append("| s | frac pairs within ≥ 0.8 | within | within p10 | within uncond | ent (images) | ent (mean map) | "
             "KL ratio | KL ratio LOO | same-class | diff-class | branch cos | kernel cos | M[s]·M[s+1] | M[s]·M[s+5] |")
    L.append("| ---: |" + " ---: |" * 14)
    med = lambda k, s: np.nanmedian(A[k][s])
    for s in range(len(d["steps"])):
        L.append(f"| {s} | {v['frac_pairs_within_ge_0.8_per_step'][s]:.2f} | " + " | ".join(
            f"{med(k, s):.3f}" if not np.isnan(med(k, s)) else "—" for k in
            ["within_median", "within_p10", "within_uncond_median", "ent_median", "ent_meanmap_median",
             "kl_ratio_median", "kl_ratio_loo_median", "within_sameclass", "within_diffclass", "branch_cos_median",
             "kernel_cos_median", "step_cos_s1_median", "step_cos_s5_median"]) + " |")
    L.append("\nPlots: `gate1_heatmaps.png` (within / entropy / KL ratio over block × step), `gate1_heads.png` "
             "(16 heads, blocks 0, 13, 27), `gate1_class_vs_noise.png`, `gate1_examples.png` (block 13, head 0).\n")
    L.append("### Verdict\n")
    L.append(f"- Agreement at every step 0–9 (≥ 70% of pairs with within ≥ 0.8): **{v['agreement_every_step_0_9']}** "
             f"(pooled fraction over s ≤ 9: {v['pooled_frac_s0_9']:.3f}).")
    L.append(f"- Share of clearing triples (s ≤ 9) with KL ratio ≥ 3: **{v['kl_ratio_ge3_share_among_clearing_s0_9']:.3f}** "
             f"(needs ≥ 0.5; leave-one-out: {v['kl_ratio_loo_ge3_share_among_clearing_s0_9']:.3f}).")
    L.append(f"- s_max = **{v['s_max']}** (contiguous from s = 0: {v['s_max_contiguous']}).")
    L.append(f"- Class vs noise (s ≤ 9, median over pairs): same-class {np.median(A['within_sameclass'][early]):.3f}, "
             f"different-class {np.median(A['within_diffclass'][early]):.3f}.")
    L.append(f"- Branch agreement (s ≤ 9): {np.median(A['branch_cos_median'][early]):.3f}; offset-kernel "
             f"reconstruction (s ≤ 9): {np.median(A['kernel_cos_median'][early]):.3f}.")
    e = A["ent_median"][early]
    L.append(f"- Caveat, near-uniformity: at s ≤ 9 the median row entropy is {np.median(e):.3f} of uniform, i.e. a "
             f"median KL(P‖U) of only {np.median((1 - e) * np.log(256)):.3f} nats; the KL ratio says the mean map explains "
             f"most of that small deviation, not that the deviation is large. Blocks with head-median KL ratio < 1.5 at "
             f"every s ≤ 9 (attention ≈ uniform there): "
             f"{[b for b in range(A['kl_ratio_median'].shape[1]) if (np.median(A['kl_ratio_median'][early, b], -1) < 1.5).all()]}. "
             f"Whether the stored map is worth more than a uniform map is tested directly by the extra Gate 2 control "
             f"arm `k10-uniform`.")
    L.append(f"\n**Gate 1: {v['verdict']}**\n")
    path = os.path.join(OUT, "gate1.md")
    pre = open(path).read().split(MARK)[0]
    atomic_write_text(path, pre + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
