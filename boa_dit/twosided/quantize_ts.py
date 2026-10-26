"""Gate 1 engine arms: DC-8 (or uniform) + extensions. Pipeline identical to `deflate/quantize_dc.py --mode dc --r 8`
(adaLN W8 RTN, sequential quantized-prefix Grams, uniform buckets, cfg branches, BoA GPTQ path, replace 1/256, P H P to
the solver, carry delta = (W - W_q) V attached right after each layer), plus, from one factor pass per block (same pass
as the Grams, before any layer of the block is quantized; see factors.py):
  --act_order      : act_order_col on every layer; act_order_row on two-sided layers
  --v v1|v2        : V rows of qkv two-sided (per head: H_col = P G_h P attention-weighted, H_row = V factor), q,k
                     rows one-sided (qkv split helper)
  --fc1 v1         : fc1 two-sided with H_row(fc1)
  --wo v2          : attn.proj two-sided with H_row(W_o) (block Jacobian factor)
  --split          : route qkv through the split helper with all three slices one-sided (the bit-exactness check)
  --base uniform   : no deflation, no carry (the read-1 fallback)
  --order downstream : provisional one-sided GPTQ of W_o and W2 first (their effective weights incl. carry feed the V1
                     and fc1 factors), then qkv/fc1 (two-sided as flagged), then a second Gram pass with qkv/fc1 quantized
                     and W_o, W2 re-quantized from FP with the new Grams, then their carries
  --base heig      : V = top-r eigenvectors of the full H (quantize_dc --mode heig), deflate + carry
  --gates sample   : Variant 2 factors with each token's own sample gates (no bucket means)
  --fc1 rg         : fc1 one-sided per row group: K = 8 groups k-means-clustered on each row's exact diagonal weight
                     w_jb = (W2^T diag(E_b[g2^2]) W2)_jj E_b[d_j^2] over buckets (normalized), group H_col =
                     sum_b p_b mean_{j in g} w_jb H_b (then P . P); the diagonal of H_row(fc1) split by bucket
Outputs: ckpt/ckpt_<arm>/model.pt; results/gate1/quant/<arm>.json; results/gate1/quant/<arm>/carry.pt;
factor eigen summaries into results/gate1/factors.json under the key <arm>.
"""
import argparse
import os
import time

import torch

from boa_dit import weights as W
from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json, assert_no_grad
from boa_dit.deflate.carry import attach, extra_bits, make_carry
from boa_dit.deflate.g1 import eigh64
from boa_dit.deflate.quantize_dc import N_PARAMS, MeanGramAccumulator
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.qutils import rtn_
from boa_dit.quantize import CALIB_SEEDS, block_state
from boa_dit.twosided.factors import FactorAccumulator, eig_summary
from boa_dit.twosided.solver import QKV_SLICES, solve_qkv_split, solve_rows

QUANT = os.path.join(RESULTS, "gate1", "quant")
FACTORS = os.path.join(RESULTS, "gate1", "factors.json")


def arm_name(a):
    n = f"ts_{a.base}{a.r if a.base != 'uniform' else ''}"
    for flag, val in (("v", a.v), ("fc1", a.fc1), ("wo", a.wo)):
        if val != "none":
            n += f"_{flag}{val}"
    if a.gates == "sample":
        n += "_gs"
    if a.order == "downstream":
        n += "_dsf"
    if a.act_order:
        n += "_ao"
    if a.split:
        n += "_split"
    return n + f"_w{a.bits}"


def update_factors_json(key, entry):
    import fcntl
    import json
    os.makedirs(os.path.dirname(FACTORS), exist_ok=True)
    with open(FACTORS + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        d = json.load(open(FACTORS)) if os.path.exists(FACTORS) else {}
        d[key] = entry
        atomic_write_json(FACTORS, d)


DOWNSTREAM = ("attn.proj", "mlp.fc2")


def deflate_basis(a, H, Hm):
    """carry directions V [d, r] for the arm's base (dc: top-r of the token-mean Gram; heig: top-r of H), or None."""
    if a.base == "dc":
        return eigh64(Hm)[1].flip(1)[:, :a.r]
    if a.base == "heig":
        return eigh64(H)[1].flip(1)[:, :a.r]
    return None


@torch.no_grad()
def fc1_rowgroups(lin, means, F, fb, P, K, B, cfg, kw):
    """--fc1 rg: one-sided solves per row group (the boa_gptq_groups machinery); see module docstring."""
    from sklearn.cluster import KMeans
    from boa_dit.twosided.solver import solve_rows
    dev = lin.weight.device
    W2 = fb.mlp.fc2.weight.data.to(dev).double()
    Mdiag = F["g2sq"] @ W2.pow(2)                                                # [B, DH]: (W2^T diag(E_b g2^2) W2)_jj
    w = (Mdiag * F["Cdiag_b"]).T                                                 # [DH, B]
    curves = (w / w.mean(1, keepdim=True).clamp_min(1e-30)).cpu().numpy()
    labels = KMeans(n_clusters=K, n_init=10, random_state=0).fit(curves).labels_
    wc, wu = cfg ** 2, (cfg - 1) ** 2
    wc, wu = wc / (wc + wu), wu / (wc + wu)
    Hb = [(wc * means[("mlp.fc1", b, 0)] + wu * means[("mlp.fc1", b, 1)]).double() for b in range(B)]
    Wt = lin.weight.data
    t, info = 0.0, []
    for g in range(K):
        rows = torch.as_tensor((labels == g).nonzero()[0], device=dev)
        if not len(rows):
            continue
        wg = w[rows].mean(0)                                                     # [B]
        H = sum(wg[b] * Hb[b] for b in range(B)) / B
        H = H / wg.mean()                                                        # scale-free (per-row constants do not matter)
        if P is not None:
            H = P @ H @ P
        Q, s = solve_rows(Wt[rows], H.float(), None, **kw)
        Wt[rows] = Q.to(Wt.dtype)
        t += s
        info.append({"n_rows": int(len(rows)), "bucket_weight_rel": (wg / wg.mean()).tolist()})
    return t, info


@torch.no_grad()
def run_block_cur(block, X, C, S, N, B, accs=(), write_back=False):
    """calib.run_block with cur = (step, bucket, branch) for the factor hooks; same forward, same slab order."""
    per = S // B
    for s in range(S):
        for br in range(2):
            r0 = s * 2 * N + br * N
            for acc in accs:
                acc.cur = (s // per, br) if isinstance(acc, MeanGramAccumulator) else (s, s // per, br)
            with torch.autocast("cuda", dtype=torch.float16):
                y = block(X[r0:r0 + N].float(), C[r0:r0 + N])
            if write_back:
                assert torch.isfinite(y).all(), "non-finite block output"
                X[r0:r0 + N] = y.to(torch.float16)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dc", choices=["dc", "heig", "uniform"])
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--v", default="none", choices=["none", "v1", "v2"])
    ap.add_argument("--fc1", default="none", choices=["none", "v1", "rg"])
    ap.add_argument("--gates", default="bucket", choices=["bucket", "sample"])
    ap.add_argument("--rg_K", type=int, default=8)
    ap.add_argument("--wo", default="none", choices=["none", "v2"])
    ap.add_argument("--act_order", action="store_true")
    ap.add_argument("--order", default="usual", choices=["usual", "downstream"],
                    help="downstream: GPTQ W_o, W2 first; V/fc1 two-sided against them; re-quantize W_o, W2 (new Grams)")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--nrows_fc1", type=int, default=64)
    ap.add_argument("--nrows_wo", type=int, default=64)
    ap.add_argument("--nrows_v", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--blocks", type=int, default=28, help="dev only")
    ap.add_argument("--print_name", action="store_true")
    a = ap.parse_args()
    if a.print_name:
        print(arm_name(a))
        return
    dev = a.device
    torch.cuda.set_device(dev)
    name = arm_name(a)
    ck = os.path.join(CKPT_DIR, f"ckpt_{name}")
    os.makedirs(ck, exist_ok=True)
    os.makedirs(os.path.join(QUANT, name), exist_ok=True)
    needs = set()
    if a.fc1 == "v1":
        needs.add("fc1")
    if a.fc1 == "rg":
        needs.add("fc1diag")
    if a.v != "none":
        needs.add("attn")
    if a.v == "v2" or a.wo == "v2":
        needs.add("tok")
    B, CFG = 10, 1.5
    p_uni = [1.0 / B] * B
    kw = dict(bits=a.bits, replace=1 / 256, act_order_col=a.act_order, act_order_row=a.act_order)
    W_fp = load_dit("cpu")
    model, diffusion = load_dit(dev), make_diffusion(50)
    S, N = diffusion.num_timesteps, len(CALIB_SEEDS)
    carry, diag, fsum, tsolve, rg_info = {}, {}, {}, {ln: 0.0 for ln in LINEARS}, {}
    tfactor = 0.0
    with Run("gate1", f"quant_{name}", dev, vars(a)) as run:
        t0 = time.time()
        X, C = capture_block0_inputs(model, diffusion, CALIB_SEEDS, CFG, dev)
        for l in range(a.blocks):
            blk = model.blocks[l]
            bfile, cfile = os.path.join(ck, f"block{l}.pt"), os.path.join(ck, f"carry{l}.pt")
            if a.resume and os.path.exists(bfile) and os.path.exists(cfile):
                blk.load_state_dict(torch.load(bfile, map_location="cpu"))
                cb = torch.load(cfile, map_location="cpu")
                for key, c in cb["carry"].items():
                    attach(block_linear(blk, key.split(".", 1)[1]), c["V"], c["delta"])
                carry.update(cb["carry"]); diag.update(cb["diag"]); fsum.update(cb.get("fsum", {}))
                for k, v in cb.get("tsolve", {}).items():
                    tsolve[k] += v
                run_block_cur(blk, X, C, S, N, B, write_back=True)
                print(f"block {l}: resumed", flush=True)
                continue
            tb = time.time()
            rtn_(blk.adaLN_modulation[1], 8)
            acc = MeanGramAccumulator(blk, dev)
            facc = FactorAccumulator(blk, l, N, dev, needs, gates=a.gates) if needs else None
            try:
                run_block_cur(blk, X, C, S, N, B, accs=[acc] + ([facc] if facc else []))
            finally:
                acc.remove()
                if facc:
                    facc.remove()
            means, means_m = acc.means()
            del acc
            tf = time.time()
            fb = W_fp.blocks[l]
            W2_down, Wo_down = fb.mlp.fc2.weight.data.to(dev).float(), fb.attn.proj.weight.data.to(dev).float()
            prov = {}
            if a.order == "downstream":   # provisional one-sided GPTQ of W_o, W2; factors see their effective weights
                for ln in DOWNSTREAM:
                    Hd = W.combine(means, ln, p_uni, B, "cfg", CFG).double()
                    Wfd = block_linear(fb, ln).weight.data.to(dev).float()
                    Vd = deflate_basis(a, Hd, W.combine(means_m, ln, p_uni, B, "cfg", CFG).double())
                    Pd = None if Vd is None else torch.eye(Hd.shape[0], device=dev, dtype=torch.float64) - Vd @ Vd.T
                    Qd, _ = solve_rows(Wfd, (Hd if Pd is None else Pd @ Hd @ Pd).float(), None, **kw)
                    prov[ln] = Qd if Vd is None else Qd + (Wfd - Qd) @ Vd.float() @ Vd.float().T
                W2_down, Wo_down = prov["mlp.fc2"], prov["attn.proj"]
            F = facc.finalize(fb.mlp.fc1.weight.data.to(dev).float(), W2_down, Wo_down) if facc else {}
            del facc
            fs = {}
            for k in ("H_row_fc1", "C_mean", "G_attn", "R", "H_row_Wo", "H_row_V1", "H_row_V2"):
                if k in F:
                    fs[k] = eig_summary(F[k])
            if F:
                fs["gate_energy_ratio_median"] = F["gate_energy_ratio"]
            tfactor += time.time() - tf
            cb, bt = {}, {ln: 0.0 for ln in LINEARS}

            def do_layer(ln, means, means_m, F):
                key = f"b{l}.{ln}"
                lin = block_linear(blk, ln)
                H = W.combine(means, ln, p_uni, B, "cfg", CFG).double()
                Wf = block_linear(fb, ln).weight.data.to(dev).float()
                V = deflate_basis(a, H, W.combine(means_m, ln, p_uni, B, "cfg", CFG).double() if a.base == "dc" else None)
                if V is not None:
                    P = torch.eye(H.shape[0], device=dev, dtype=torch.float64) - V @ V.T
                    Hs = (P @ H @ P).float()
                else:
                    P, Hs = None, H.float()
                if ln == "attn.qkv" and (a.v != "none" or a.split):
                    specs = {"q": (Hs, None, {}), "k": (Hs, None, {})}
                    if a.v == "none":
                        specs["v"] = (Hs, None, {})
                    else:
                        G = F["G_attn"]
                        Gs = (P.unsqueeze(0) @ G @ P.unsqueeze(0)) if P is not None else G
                        specs["v"] = (Gs.float(), F["H_row_V1" if a.v == "v1" else "H_row_V2"].float(), {"n_rows": a.nrows_v})
                    bt[ln] = sum(solve_qkv_split(lin, specs, **kw).values())
                elif ln == "attn.proj" and a.wo == "v2":
                    Q, bt[ln] = solve_rows(lin.weight.data, Hs, F["H_row_Wo"].float(), n_rows=a.nrows_wo, **kw)
                    lin.weight.data = Q.to(lin.weight.dtype)
                elif ln == "mlp.fc1" and a.fc1 == "rg":
                    bt[ln], rg_info[l] = fc1_rowgroups(lin, means, F, fb, P, a.rg_K, B, CFG, kw)
                elif ln == "mlp.fc1" and a.fc1 == "v1":
                    Q, bt[ln] = solve_rows(lin.weight.data, Hs, F["H_row_fc1"].float(), n_rows=a.nrows_fc1, **kw)
                    lin.weight.data = Q.to(lin.weight.dtype)
                else:
                    Q, bt[ln] = solve_rows(lin.weight.data, Hs, None, **kw)
                    lin.weight.data = Q.to(lin.weight.dtype)
                Wq = lin.weight.data.float()
                d = {"drift": float((Wq - Wf).norm() / Wf.norm()), "solve_s": bt[ln]}
                if V is not None:
                    c = make_carry(Wf, Wq, V.float())
                    attach(lin, c["V"], c["delta"])
                    cb[key] = c
                    d["r"] = int(V.shape[1])
                    d["carry_rel"] = float(c["delta"].float().norm() / ((Wf - Wq).norm() + 1e-12))
                diag[key] = d
                tsolve[ln] += bt[ln]

            if a.order == "downstream":
                for ln in ("attn.qkv", "mlp.fc1"):
                    do_layer(ln, means, means_m, F)
                acc2 = MeanGramAccumulator(blk, dev)                                  # Grams with qkv, fc1 quantized
                try:
                    run_block_cur(blk, X, C, S, N, B, accs=[acc2])
                finally:
                    acc2.remove()
                means2, means_m2 = acc2.means()
                del acc2
                for ln in DOWNSTREAM:                                                 # re-quantize W_o, W2 from FP
                    do_layer(ln, means2, means_m2, F)
                del means2, means_m2
            else:
                for ln in LINEARS:
                    do_layer(ln, means, means_m, F)
            del means, means_m, F
            torch.cuda.empty_cache()
            run_block_cur(blk, X, C, S, N, B, write_back=True)
            atomic_torch_save(block_state(blk), bfile)
            atomic_torch_save({"carry": cb, "diag": {k: diag[k] for k in diag if k.startswith(f"b{l}.")},
                               "fsum": {l: fs}, "tsolve": bt}, cfile)
            carry.update(cb)
            fsum[l] = fs
            print(f"block {l}: {time.time() - tb:.1f}s  solve " + " ".join(f"{k.split('.')[1]}={v:.1f}" for k, v in bt.items()),
                  flush=True)
        assert_no_grad(X, C)
        del X, C
        if a.blocks < 28:
            return
        atomic_torch_save(model.state_dict(), os.path.join(ck, "model.pt"))
        atomic_torch_save(carry, os.path.join(QUANT, name, "carry.pt"))
        rec = run.record()
    uni = load_dit("cpu")
    load_quant_state(uni, os.path.join(CKPT_DIR, f"ckpt_uniform_w{a.bits}_K1_lam1_brcfg", "model.pt"))
    for key in diag:
        l, ln = key.split(".", 1)
        Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.float()
        Wu = block_linear(uni.blocks[int(l[1:])], ln).weight.data.float()
        diag[key]["drift_uniform"] = float((Wu - Wf).norm() / Wf.norm())
        diag[key]["drift_ratio"] = diag[key]["drift"] / diag[key]["drift_uniform"]
    eb, per_type = extra_bits(carry, N_PARAMS) if carry else (0.0, {})
    res = {"name": name, "args": vars(a), "extra_bits_per_param": eb,
           "extra_bits_per_type": {k: v / N_PARAMS for k, v in per_type.items()},
           "solver_wall_s_per_type": tsolve, "fc1_rowgroups": rg_info, "factor_finalize_wall_s": tfactor,
           "max_drift_ratio": max(d["drift_ratio"] for d in diag.values()), "layers": diag,
           "quant_wall_s": time.time() - t0, **rec}
    atomic_write_json(os.path.join(QUANT, f"{name}.json"), res)
    if any(fsum.values()):
        update_factors_json(name, {"source": f"engine arm {name} (its own quantized prefix)",
                                   "blocks": {str(k): v for k, v in sorted(fsum.items(), key=lambda kv: int(kv[0]))}})
    for f in os.listdir(ck):
        if f.startswith(("block", "carry")):
            os.remove(os.path.join(ck, f))
    print(f"done {name}: extra bits/param {eb:.4f}; max drift ratio {res['max_drift_ratio']:.2f}; solver s {tsolve}",
          flush=True)


if __name__ == "__main__":
    main()
