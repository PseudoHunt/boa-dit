"""Oracle for the block objective: BRECQ-style block reconstruction (AdaRound on the true block-2 output loss) vs E4.

Inputs to block 2 come from the E4 arm's own quantized prefix (blocks 0-1 with carries), FP trajectory capture as in
quantize.py. Target: FP block 2 with the arm's W8 adaLN (so only the four linears differ). AdaRound learns the rounding
of all four linears jointly: W_q = s (clamp(floor(W/s) + h(V) + z, 0, maxq) - z), h = clamp(sigmoid(V)(1.2) - 0.1, 0, 1),
per-channel (s, z) from the fork's Hessian grid search on P H P (H = this block's input Gram on the training inputs,
P from the arm's own carry directions V); equal bits: the oracle keeps the same carry delta = (W - W_q) V.
Loss = weighted block-output MSE / (the same for nearest rounding) + lam * mean(1 - |2h - 1|^beta), beta 20 -> 2 after a
20% warm-up, Adam, 1000 steps, batch 16 sequences, fp32. lr and lam are chosen on validation seeds 1028-1031 (training
1000-1027); judged on held-out calibration seeds 1032-1047 under the pipeline's fp16 autocast forward.
Error = cfg-branch-weighted mean of ||y - y_fp||^2 over tokens. Run: python -m boa_dit.twosided.oracle --device cuda:0
"""
import argparse
import copy
import os
import time

import torch
from torch.func import functional_call

from boa_dit import weights as W
from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.carry import attach, carry_path, install_from, make_carry
from boa_dit.deflate.quantize_dc import MeanGramAccumulator
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.qutils import MinMaxQuantizer
from boa_dit.twosided.quantize_ts import run_block_cur

L_BLK = 2
ZETA, GAMMA = 1.1, -0.1
TRAIN, VAL, TEST = list(range(1000, 1028)), list(range(1028, 1032)), list(range(1032, 1048))
REFS = ["ts_heig8_vv1_fc1v1_w4", "deflate_heig_r8_w4", "ts_uniform_vv1_fc1v1_w4", "uniform_w4_K1_lam1_brcfg"]


def load_arm(name, dev):
    m = load_dit(dev)
    load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{name}", "model.pt"))
    if os.path.exists(carry_path(name)):
        install_from(m, name)
    return m


@torch.no_grad()
def block_inputs(model, fp, seeds, dev, diffusion):
    """Block-2 inputs from `model`'s blocks 0..1 (carries active), FP-trajectory capture. Returns X [S*2N, 256, D], C."""
    X, C = capture_block0_inputs(fp, diffusion, seeds, 1.5, dev)
    S, N = diffusion.num_timesteps, len(seeds)
    for l in range(L_BLK):
        run_block_cur(model.blocks[l], X, C, S, N, 10, write_back=True)
    return X, C


def row_weights(S, N, cfg=1.5):
    wc, wu = cfg ** 2, (cfg - 1) ** 2
    r = torch.arange(S * 2 * N)
    br = (r % (2 * N)) // N
    return torch.where(br == 0, wc / (wc + wu), wu / (wc + wu))


@torch.no_grad()
def block_err(blk, fpb, X, C, w, bs=32):
    """cfg-weighted mean over rows of per-token-mean ||y - y_fp||^2, and the same normalized by ||y_fp - x||^2."""
    num, den, tot = 0.0, 0.0, 0.0
    for i in range(0, len(X), bs):
        x, c = X[i:i + bs].float(), C[i:i + bs]
        with torch.autocast("cuda", dtype=torch.float16):
            y, yf = blk(x, c).float(), fpb(x, c).float()
        ww = w[i:i + bs].to(x.device)
        num += float((ww * (y - yf).pow(2).sum(-1).mean(-1)).sum())
        den += float((ww * (yf - x).pow(2).sum(-1).mean(-1)).sum())
        tot += float(ww.sum())
    return {"mse": num / tot, "rel_to_update": num / den}


def adaround(fpb, Wfp, grids, Vs, X, C, w, Xv, Cv, wv, lr, lam, steps, dev, seed=0):
    """Returns hard-rounded W_q per layer and the validation error."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    params, base = {}, {}
    for ln in LINEARS:
        s, z, maxq = grids[ln]
        Wf = Wfp[ln]
        fl = torch.floor(Wf / s)
        frac = (Wf / s - fl).clamp(0.01, 0.99)
        params[ln] = (-torch.log((ZETA - GAMMA) / (frac - GAMMA) - 1)).requires_grad_(True)
        base[ln] = (fl, s, z, maxq)
    opt = torch.optim.Adam(list(params.values()), lr=lr)

    def weights(hard):
        out = {}
        for ln in LINEARS:
            fl, s, z, maxq = base[ln]
            h = (params[ln].sigmoid() * (ZETA - GAMMA) + GAMMA).clamp(0, 1)
            if hard:
                h = (h >= 0.5).float()
            Wq = s * (torch.clamp(fl + h + z, 0, maxq) - z)
            V = Vs[ln]
            out[f"{ln}.weight"] = Wq + (Wfp[ln] - Wq) @ V @ V.T                 # carry folded in (training only)
        return out, h

    def rec(Wd, x, c, wr):
        y = functional_call(fpb, Wd, (x, c))
        with torch.no_grad():
            yf = fpb(x, c)
        return ((wr * (y - yf).pow(2).sum(-1).mean(-1)).sum() / wr.sum())

    idx_all = torch.arange(len(X))
    with torch.no_grad():
        Wh, _ = weights(True)
        ii = idx_all[torch.randperm(len(X), generator=g)[:256]]
        r_rtn = sum(float(rec(Wh, X[ii[k:k + 16]].float(), C[ii[k:k + 16]], w[ii[k:k + 16]].to(dev))) for k in range(0, 256, 16)) / 16
    warm = int(0.2 * steps)
    for it in range(steps):
        ii = idx_all[torch.randint(len(X), (16,), generator=g)]
        Wd, _ = weights(False)
        loss = rec(Wd, X[ii].float(), C[ii], w[ii].to(dev)) / r_rtn
        if it >= warm:
            beta = 20 + (2 - 20) * (it - warm) / max(1, steps - warm)
            reg = 0.0
            n = 0
            for ln in LINEARS:
                h = (params[ln].sigmoid() * (ZETA - GAMMA) + GAMMA).clamp(0, 1)
                reg = reg + (1 - (2 * h - 1).abs().pow(beta)).sum()
                n += h.numel()
            loss = loss + lam * reg / n
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        Wh, _ = weights(True)
        Wq = {}
        for ln in LINEARS:
            fl, s, z, maxq = base[ln]
            h = ((params[ln].sigmoid() * (ZETA - GAMMA) + GAMMA).clamp(0, 1) >= 0.5).float()
            Wq[ln] = s * (torch.clamp(fl + h + z, 0, maxq) - z)
        vb = copy.deepcopy(fpb)
        for ln in LINEARS:
            lin = block_linear(vb, ln)
            lin.weight.data = Wq[ln].clone()
            c = make_carry(Wfp[ln], Wq[ln], Vs[ln])
            attach(lin, c["V"], c["delta"])
        ve = block_err(vb, fpb, Xv, Cv, wv)
    return Wq, ve, r_rtn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="ts_heig8_vv1_fc1v1_w4")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    diffusion = make_diffusion(50)
    S = diffusion.num_timesteps
    out = {"arm": a.arm, "block": L_BLK, "steps": a.steps}
    with Run("gate1", f"oracle_b{L_BLK}_{a.arm}", dev, vars(a)) as run:
        fp = load_dit(dev)
        arm = load_arm(a.arm, dev)
        with torch.no_grad():
            Xtr, Ctr = block_inputs(arm, fp, TRAIN, dev, diffusion)
            Xv, Cv = block_inputs(arm, fp, VAL, dev, diffusion)
            Xte, Cte = block_inputs(arm, fp, TEST, dev, diffusion)
        wtr, wv, wte = row_weights(S, len(TRAIN)), row_weights(S, len(VAL)), row_weights(S, len(TEST))
        fpb = copy.deepcopy(fp.blocks[L_BLK])
        fpb.adaLN_modulation[1].weight.data.copy_(arm.blocks[L_BLK].adaLN_modulation[1].weight.data)
        fpb.requires_grad_(False)
        Wfp = {ln: block_linear(fpb, ln).weight.data.float().clone() for ln in LINEARS}
        # grids: fork's Hessian grid search on P H P (H from the training inputs, FP block), V = the arm's carry directions
        acc = MeanGramAccumulator(fpb, dev)
        try:
            with torch.no_grad():
                run_block_cur(fpb, Xtr, Ctr, S, len(TRAIN), 10, accs=[acc])
        finally:
            acc.remove()
        means, _ = acc.means()
        carry = torch.load(carry_path(a.arm), map_location="cpu")
        grids, Vs = {}, {}
        for ln in LINEARS:
            H = W.combine(means, ln, [0.1] * 10, 10, "cfg", 1.5).double()
            key = f"b{L_BLK}.{ln}"
            V = carry[key]["V"].to(dev).float() if key in carry else torch.zeros(H.shape[0], 0, device=dev)
            P = torch.eye(H.shape[0], device=dev, dtype=torch.float64) - V.double() @ V.double().T
            Hs = (P @ H @ P).float()
            Hs = Hs + 0.01 * Hs.diagonal().mean() * torch.eye(Hs.shape[0], device=dev)
            q = MinMaxQuantizer()
            q.configure(4, per_channel=True, sym=False, mse=False)
            q.find_params(Wfp[ln])
            s, z = q.find_params_H(Wfp[ln], Hs, search=True)
            grids[ln], Vs[ln] = (s, z, float(q.maxq)), V
        del means, acc
        torch.cuda.empty_cache()
        cfgs = []
        for lr in (1e-3, 1e-2):
            for lam in (0.01, 0.1):
                t0 = time.time()
                Wq, ve, r_rtn = adaround(fpb, Wfp, grids, Vs, Xtr, Ctr, wtr, Xv, Cv, wv, lr, lam, a.steps, dev)
                cfgs.append({"lr": lr, "lam": lam, "val": ve, "wall_s": time.time() - t0, "Wq": Wq})
                print(f"lr {lr} lam {lam}: val mse {ve['mse']:.4e} ({time.time() - t0:.0f} s)", flush=True)
        best = min(cfgs, key=lambda c: c["val"]["mse"])
        ob = copy.deepcopy(fpb)
        for ln in LINEARS:
            lin = block_linear(ob, ln)
            lin.weight.data = best["Wq"][ln].clone()
            c = make_carry(Wfp[ln], best["Wq"][ln], Vs[ln])
            attach(lin, c["V"], c["delta"])
        out["oracle_configs"] = [{k: v for k, v in c.items() if k != "Wq"} for c in cfgs]
        out["oracle_chosen"] = {"lr": best["lr"], "lam": best["lam"]}
        out["test"] = {"oracle": block_err(ob, fpb, Xte, Cte, wte)}
        # nearest rounding on the same grid (+ same carry): the AdaRound starting point
        nb = copy.deepcopy(fpb)
        for ln in LINEARS:
            s, z, maxq = grids[ln]
            Wn = s * (torch.clamp(torch.round(Wfp[ln] / s) + z, 0, maxq) - z)
            lin = block_linear(nb, ln)
            lin.weight.data = Wn
            c = make_carry(Wfp[ln], Wn, Vs[ln])
            attach(lin, c["V"], c["delta"])
        out["test"]["nearest_same_grid"] = block_err(nb, fpb, Xte, Cte, wte)
        out["test"][a.arm] = block_err(arm.blocks[L_BLK], fpb, Xte, Cte, wte)
        for ref in REFS:
            if ref == a.arm or not os.path.exists(os.path.join(CKPT_DIR, f"ckpt_{ref}", "model.pt")):
                continue
            m = load_arm(ref, dev)
            out["test"][ref] = block_err(m.blocks[L_BLK], fpb, Xte, Cte, wte)
            del m
            torch.cuda.empty_cache()
        out["oracle_vs_arm_error_ratio"] = out["test"]["oracle"]["mse"] / out["test"][a.arm]["mse"]
        out["oracle_wins_by_20pct"] = out["oracle_vs_arm_error_ratio"] <= 0.8
        out.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "gate1", f"oracle_b{L_BLK}_{a.arm}{'' if a.steps == 1000 else f'_steps{a.steps}'}.json"), out)
    print({k: v for k, v in out.items() if k not in ("config",)}, flush=True)


if __name__ == "__main__":
    main()
