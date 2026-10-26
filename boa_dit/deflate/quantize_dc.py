"""Deflate-and-carry quantization (G2 arms). Same pipeline as the uniform arm (`boa_dit/quantize.py`: adaLN W8 RTN,
sequential quantized-prefix Grams, uniform bucket weights, branch cfg, BoA GPTQ path, replace 1/256), plus:
  * a token-mean Gram per (layer, bucket, branch): Gm = sum_s 256 xbar_s xbar_s^T (same normalization as G)
  * V per layer: top-r eigenvectors of H_mean (dc, c), or top-r right singular vectors of the FP weight (wsvd);
    for `c` V is loaded from the matching dc arm's carry (same V)
  * the solver gets P H P (dc, wsvd) or H (c); optional extra damping (pre-added, see G2.md)
  * carry delta = (W - W_q) V attached right after the layer is quantized (next block's inputs see it)
Outputs: ckpt/ckpt_<arm>/model.pt (W_q), results/deflate/quant/<arm>/carry.pt, results/deflate/quant/<arm>.json.
"""
import argparse
import json
import os
import time

import torch

from boa_dit import weights as W
from boa_dit.calib import capture_block0_inputs, run_block
from boa_dit.common import CKPT_DIR, ROOT, RESULTS, Run, atomic_torch_save, atomic_write_json, assert_no_grad
from boa_dit.deflate.carry import QUANT, attach, carry_path, extra_bits, make_carry
from boa_dit.deflate.g1 import eigh64
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.qutils import boa_gptq_groups, rtn_
from boa_dit.quantize import CALIB_SEEDS, block_state

N_PARAMS = 445_906_944


class MeanGramAccumulator:
    """Like calib.GramAccumulator (fp32 sum x x^T per (layer, bucket, branch)) plus the token-mean Gram."""

    def __init__(self, block, dev):
        self.G, self.Gm, self.n, self.cur = {}, {}, {}, None
        self.hooks = [block_linear(block, l).register_forward_pre_hook(self._hook(l)) for l in LINEARS]

    def _hook(self, ln):
        @torch.autocast("cuda", enabled=False)
        def h(mod, inp):
            x = inp[0].float()                                                    # [N, 256, d]
            xf = x.reshape(-1, x.shape[-1])
            xb = x.mean(1)
            key = (ln, *self.cur)
            g, gm = xf.T @ xf, 256.0 * (xb.T @ xb)
            if key in self.G:
                self.G[key] += g; self.Gm[key] += gm; self.n[key] += xf.shape[0]
            else:
                self.G[key], self.Gm[key], self.n[key] = g, gm, xf.shape[0]
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()

    def means(self):
        return ({k: v / self.n[k] for k, v in self.G.items()}, {k: v / self.n[k] for k, v in self.Gm.items()})


def arm_name(a):
    lab = json.load(open(a.r_config)).get("label", "alloc") if getattr(a, "r_config", None) else a.r
    n = f"deflate_{a.mode}_r{lab}_w{a.bits}"
    return n + (f"_damp{a.damp:g}" if a.damp != 0.01 else "")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["dc", "c", "wsvd", "heig"])
    ap.add_argument("--r", default="star", help="'star' = r_l per layer from configs/deflate_r.json, or an integer")
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--r_config", default=None, help="per-layer rank file (with --r star); arm named r'alloc'")
    ap.add_argument("--damp", type=float, default=0.01, help="total damping as a fraction of mean diag (solver adds 0.01)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--blocks", type=int, default=28, help="dev only: quantize the first n blocks, then exit")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    name = arm_name(a)
    ck = os.path.join(CKPT_DIR, f"ckpt_{name}")
    os.makedirs(ck, exist_ok=True)
    os.makedirs(os.path.join(QUANT, name), exist_ok=True)
    rcfg = json.load(open(a.r_config or os.path.join(ROOT, "configs", "deflate_r.json")))["r_per_layer"]
    r_of = (lambda key: rcfg[key]) if a.r == "star" else (lambda key: int(a.r))
    v_from = None
    if a.mode == "c":                                                             # same V as the dc arm
        v_from = torch.load(carry_path(f"deflate_dc_r{a.r}_w{a.bits}"), map_location="cpu")
    B, CFG = 10, 1.5
    p_uni = [1.0 / B] * B
    W_fp = load_dit("cpu")
    model, diffusion = load_dit(dev), make_diffusion(50)
    S, N = diffusion.num_timesteps, len(CALIB_SEEDS)
    carry, diag = {}, {}
    with Run("deflate", f"quant_{name}", dev, vars(a)) as run:
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
                carry.update(cb["carry"]); diag.update(cb["diag"])
                run_block(blk, X, C, S, N, B, write_back=True)
                print(f"block {l}: resumed", flush=True)
                continue
            tb = time.time()
            rtn_(blk.adaLN_modulation[1], 8)
            acc = MeanGramAccumulator(blk, dev)
            try:
                run_block(blk, X, C, S, N, B, acc=acc)
            finally:
                acc.remove()
            means, means_m = acc.means()
            del acc
            cb = {}
            for ln in LINEARS:
                key = f"b{l}.{ln}"
                lin = block_linear(blk, ln)
                H = W.combine(means, ln, p_uni, B, "cfg", CFG).double()
                r = r_of(key)
                Wf = block_linear(W_fp.blocks[l], ln).weight.data.to(dev).float()
                if a.mode == "wsvd":
                    V = torch.linalg.svd(Wf.double(), full_matrices=False)[2][:r].T     # [d_in, r]
                elif a.mode == "heig":                                            # top-r eigenvectors of the full H
                    V = eigh64(H)[1].flip(1)[:, :r]
                elif a.mode == "c":
                    V = v_from[key]["V"].to(dev).double()
                else:
                    Hm = W.combine(means_m, ln, p_uni, B, "cfg", CFG).double()
                    V = eigh64(Hm)[1].flip(1)[:, :r]
                if a.mode in ("dc", "wsvd", "heig"):
                    P = torch.eye(H.shape[0], device=dev, dtype=torch.float64) - V @ V.T
                    Hs = P @ H @ P
                else:
                    Hs = H
                if a.damp != 0.01:
                    Hs = Hs + ((a.damp - 0.01) / 1.01) * Hs.diagonal().mean() * torch.eye(Hs.shape[0], device=dev, dtype=Hs.dtype)
                Hs = Hs.float()
                boa_gptq_groups(lin, [(list(range(lin.weight.shape[0])), Hs)], a.bits, replace=1 / 256)
                Wq = lin.weight.data.float()
                c = make_carry(Wf, Wq, V.float())
                attach(lin, c["V"], c["delta"])
                cb[key] = c
                lo, hi = Wq.min(1, keepdim=True).values, Wq.max(1, keepdim=True).values
                diag[key] = {"r": int(V.shape[1]), "drift": float((Wq - Wf).norm() / Wf.norm()),
                             "clip_frac": float(((Wf < lo - 1e-8) | (Wf > hi + 1e-8)).float().mean()),
                             "carry_rel": float(c["delta"].float().norm() / ((Wf - Wq).norm() + 1e-12))}
            del means, means_m
            torch.cuda.empty_cache()
            run_block(blk, X, C, S, N, B, write_back=True)                         # next inputs, carry active
            atomic_torch_save(block_state(blk), bfile)
            atomic_torch_save({"carry": cb, "diag": {k: diag[k] for k in cb}}, cfile)
            carry.update(cb)
            print(f"block {l}: {time.time() - tb:.1f}s", flush=True)
        assert_no_grad(X, C)
        del X, C
        if a.blocks < 28:
            return
        atomic_torch_save(model.state_dict(), os.path.join(ck, "model.pt"))
        atomic_torch_save(carry, carry_path(name))
        rec = run.record()
    # drift against the uniform arm's quantized weights
    uni = load_dit("cpu")
    load_quant_state(uni, os.path.join(CKPT_DIR, f"ckpt_uniform_w{a.bits}_K1_lam1_brcfg", "model.pt"))
    for key in diag:
        l, ln = key.split(".", 1)
        Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.float()
        Wu = block_linear(uni.blocks[int(l[1:])], ln).weight.data.float()
        diag[key]["drift_uniform"] = float((Wu - Wf).norm() / Wf.norm())
        lo, hi = Wu.min(1, keepdim=True).values, Wu.max(1, keepdim=True).values
        diag[key]["clip_frac_uniform"] = float(((Wf < lo - 1e-8) | (Wf > hi + 1e-8)).float().mean())
        diag[key]["drift_ratio"] = diag[key]["drift"] / diag[key]["drift_uniform"]
    eb, per_type = extra_bits(carry, N_PARAMS)
    res = {"name": name, "mode": a.mode, "r": a.r, "bits": a.bits, "damp": a.damp, "extra_bits_per_param": eb,
           "extra_bits_per_type": {k: v / N_PARAMS for k, v in per_type.items()},
           "max_drift_ratio": max(d["drift_ratio"] for d in diag.values()), "layers": diag, "quant_wall_s": time.time() - t0, **rec}
    atomic_write_json(os.path.join(QUANT, f"{name}.json"), res)
    for f in os.listdir(ck):
        if f.startswith(("block", "carry")):
            os.remove(os.path.join(ck, f))
    print(f"done {name}: extra bits/param {eb:.4f}; max drift ratio vs uniform {res['max_drift_ratio']:.2f}", flush=True)


if __name__ == "__main__":
    main()
