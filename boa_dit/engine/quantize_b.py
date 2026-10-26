"""§4 idea B: true-sequential, order-aware, optionally asymmetric calibration on the heig-8 recipe (results/engine/rule.md).

Per block (adaLN W8 RTN on the quantized block; the paired FP block is fully FP):
  for each step of the order: if the layer's input Gram for the current state is not yet collected, one pass over the
  calibration slabs accumulates H = E[x~ x~^T] (quantized block, carries active, layers earlier in the order quantized)
  and, for --target asym, C = E[x x~^T] with x the FP model's input to the same layer (FP-propagated block input, FP
  block); bucket-uniform, cfg branch weights. V = top-8 eigenvectors of H; target W~ = W (sym) or
  W + W (C - H)(H + 0.01 mean diag(H) I)^-1 (asym; correction form); W~[rows] quantized against P H P (fork GPTQ path); carry delta = (W~ - W_q) V,
  zero rows for rows not yet quantized; re-attached after each step.
Orders: fwd = Q,K,V,O,F1,F2 ; err = V,O,F1,F2,Q,K. q, k, v are row slices of the fused qkv (one shared input Gram).
Outputs: ckpt/ckpt_<arm>/model.pt, results/engine/quant/<arm>/carry.pt, results/engine/quant/<arm>.json.
Run: python -m boa_dit.engine.quantize_b --order fwd|err --target sym|asym --device cuda:1
"""
import argparse
import os
import time

import torch

from boa_dit import weights as W
from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json, assert_no_grad
from boa_dit.deflate.carry import attach, extra_bits
from boa_dit.deflate.g1 import eigh64
from boa_dit.deflate.quantize_dc import N_PARAMS
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.qutils import rtn_
from boa_dit.quantize import CALIB_SEEDS, block_state
from boa_dit.twosided.solver import QKV_SLICES, solve_rows

QUANT = os.path.join(RESULTS, "engine", "quant")
ORDERS = {"fwd": [("attn.qkv", "q"), ("attn.qkv", "k"), ("attn.qkv", "v"), ("attn.proj", None), ("mlp.fc1", None), ("mlp.fc2", None)],
          "err": [("attn.qkv", "v"), ("attn.proj", None), ("mlp.fc1", None), ("mlp.fc2", None), ("attn.qkv", "q"), ("attn.qkv", "k")]}
R = 8


class PairGram:
    """Forward pre-hooks on one linear of the quantized block (x~) and of the FP block (x); accumulates per (bucket,
    branch) sums of x~^T x~ and (asym) x^T x~. The FP hook stores x; the quantized hook (run second) consumes it."""

    def __init__(self, qlin, flin, asym):
        self.asym, self.cur, self.x = asym, None, None
        self.H, self.C, self.n = {}, {}, {}
        self.h = [qlin.register_forward_pre_hook(self._q)]
        if asym:
            self.h.append(flin.register_forward_pre_hook(self._f))

    def _f(self, mod, inp):
        with torch.autocast("cuda", enabled=False):
            self.x = inp[0].reshape(-1, inp[0].shape[-1]).float()

    def _q(self, mod, inp):
        with torch.autocast("cuda", enabled=False):
            xq = inp[0].reshape(-1, inp[0].shape[-1]).float()
            k = self.cur
            g = xq.T @ xq
            self.H[k] = self.H[k] + g if k in self.H else g
            self.n[k] = self.n.get(k, 0) + xq.shape[0]
            if self.asym:
                c = self.x.T @ xq
                self.C[k] = self.C[k] + c if k in self.C else c

    def remove(self):
        for h in self.h:
            h.remove()

    def combine(self, B=10, cfg=1.5):
        wc, wu = cfg ** 2, (cfg - 1) ** 2
        wc, wu = wc / (wc + wu), wu / (wc + wu)
        H = sum((wc * self.H[(b, 0)] / self.n[(b, 0)] + wu * self.H[(b, 1)] / self.n[(b, 1)]) for b in range(B)) / B
        C = sum((wc * self.C[(b, 0)] / self.n[(b, 0)] + wu * self.C[(b, 1)] / self.n[(b, 1)]) for b in range(B)) / B if self.asym else None
        return H.double(), (C.double() if C is not None else None)


@torch.no_grad()
def gram_pass(qb, fb, X, Xf, C, S, N, B, ln, asym):
    acc = PairGram(block_linear(qb, ln), block_linear(fb, ln), asym)
    per = S // B
    try:
        for s in range(S):
            for br in range(2):
                r0 = s * 2 * N + br * N
                acc.cur = (s // per, br)
                with torch.autocast("cuda", dtype=torch.float16):
                    if asym:
                        fb(Xf[r0:r0 + N].float(), C[r0:r0 + N])      # FP first: stores x for the paired hook
                    qb(X[r0:r0 + N].float(), C[r0:r0 + N])
    finally:
        acc.remove()
    return acc.combine(B)


@torch.no_grad()
def write_back(blk, X, C, S, N):
    for s in range(S):
        for br in range(2):
            r0 = s * 2 * N + br * N
            with torch.autocast("cuda", dtype=torch.float16):
                y = blk(X[r0:r0 + N].float(), C[r0:r0 + N])
            assert torch.isfinite(y).all()
            X[r0:r0 + N] = y.to(torch.float16)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--order", choices=list(ORDERS), required=True)
    ap.add_argument("--target", choices=["sym", "asym"], required=True)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--blocks", type=int, default=28)
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    name = f"engB_{a.order}_{a.target}_heig8_w{a.bits}"
    ck = os.path.join(CKPT_DIR, f"ckpt_{name}")
    os.makedirs(ck, exist_ok=True)
    os.makedirs(os.path.join(QUANT, name), exist_ok=True)
    asym = a.target == "asym"
    B = 10
    W_fp = load_dit("cpu")
    model, diffusion = load_dit(dev), make_diffusion(50)
    fpm = load_dit(dev) if asym else None
    S, N = diffusion.num_timesteps, len(CALIB_SEEDS)
    carry, diag, tsolve = {}, {}, {ln: 0.0 for ln in LINEARS}
    kw = dict(bits=a.bits, replace=1 / 256)
    with Run("engine", f"quant_{name}", dev, vars(a)) as run:
        t0 = time.time()
        X, C = capture_block0_inputs(model, diffusion, CALIB_SEEDS, 1.5, dev)
        Xf = X.clone() if asym else None
        for l in range(a.blocks):
            blk = model.blocks[l]
            fb = fpm.blocks[l] if asym else None
            bfile, cfile = os.path.join(ck, f"block{l}.pt"), os.path.join(ck, f"carry{l}.pt")
            if a.resume and os.path.exists(bfile) and os.path.exists(cfile):
                blk.load_state_dict(torch.load(bfile, map_location="cpu"))
                cb = torch.load(cfile, map_location="cpu")
                for key, c in cb["carry"].items():
                    attach(block_linear(blk, key.split(".", 1)[1]), c["V"], c["delta"])
                carry.update(cb["carry"]); diag.update(cb["diag"])
                write_back(blk, X, C, S, N)
                if asym:
                    write_back(fb, Xf, C, S, N)
                print(f"block {l}: resumed", flush=True)
                continue
            tb = time.time()
            rtn_(blk.adaLN_modulation[1], 8)
            grams, handles, done_rows, cb, bt = {}, {}, {}, {}, {ln: 0.0 for ln in LINEARS}
            for ln, sl in ORDERS[a.order]:
                lin = block_linear(blk, ln)
                Wf = block_linear(W_fp.blocks[l], ln).weight.data.to(dev).float()
                if ln not in grams:                                            # Gram for the current state
                    H, Cx = gram_pass(blk, fb, X, Xf, C, S, N, B, ln, asym)
                    V = eigh64(H)[1].flip(1)[:, :R]
                    P = torch.eye(H.shape[0], device=dev, dtype=torch.float64) - V @ V.T
                    if asym:
                        Hd = H + 0.01 * H.diagonal().mean() * torch.eye(H.shape[0], device=dev, dtype=H.dtype)
                        # correction form (DEVIATION, see REPORT): W~ = W + W (C - H)(H + damp)^-1; equals W when C = H.
                        # The literal W C (H + damp)^-1 shrank W by 40% at block 0, where x~ = x exactly.
                        Wt = (Wf.double() + Wf.double() @ (Cx - H) @ torch.linalg.inv(Hd)).float()
                    else:
                        Wt = Wf
                    grams[ln] = ((P @ H @ P).float(), V.float(), Wt)
                    done_rows[ln] = torch.zeros(Wf.shape[0], dtype=torch.bool, device=dev)
                Hs, V, Wt = grams[ln]
                rows = QKV_SLICES[sl] if sl else slice(None)
                Q, s = solve_rows(Wt[rows], Hs, None, **kw)
                lin.weight.data[rows] = Q.to(lin.weight.dtype)
                bt[ln] += s
                done_rows[ln][rows] = True
                delta = ((Wt - lin.weight.data.float()) * done_rows[ln][:, None]) @ V
                if ln in handles:
                    handles[ln].remove()
                c = {"V": V.half().cpu(), "delta": delta.half().cpu()}
                handles[ln] = attach(lin, c["V"], c["delta"])
                cb[f"b{l}.{ln}"] = c
            for ln in LINEARS:
                Wq = block_linear(blk, ln).weight.data.float()
                Wf = block_linear(W_fp.blocks[l], ln).weight.data.to(dev).float()
                Wt = grams[ln][2]
                diag[f"b{l}.{ln}"] = {"drift": float((Wq - Wf).norm() / Wf.norm()),
                                      "target_shift": float((Wt - Wf).norm() / Wf.norm()), "solve_s": bt[ln]}
                tsolve[ln] += bt[ln]
            del grams
            torch.cuda.empty_cache()
            write_back(blk, X, C, S, N)
            if asym:
                write_back(fb, Xf, C, S, N)
            atomic_torch_save(block_state(blk), bfile)
            atomic_torch_save({"carry": cb, "diag": {k: diag[k] for k in cb}}, cfile)
            carry.update(cb)
            print(f"block {l}: {time.time() - tb:.1f}s", flush=True)
        assert_no_grad(X, C)
        del X, C, Xf
        if a.blocks < 28:
            return
        atomic_torch_save(model.state_dict(), os.path.join(ck, "model.pt"))
        atomic_torch_save(carry, os.path.join(QUANT, name, "carry.pt"))
        rec = run.record()
    uni = load_dit("cpu")
    load_quant_state(uni, os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt"))
    for key in diag:
        l, ln = key.split(".", 1)
        Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.float()
        Wu = block_linear(uni.blocks[int(l[1:])], ln).weight.data.float()
        diag[key]["drift_ratio"] = diag[key]["drift"] / float((Wu - Wf).norm() / Wf.norm())
    eb, per_type = extra_bits(carry, N_PARAMS)
    atomic_write_json(os.path.join(QUANT, f"{name}.json"),
                      {"name": name, "args": vars(a), "extra_bits_per_param": eb, "solver_wall_s_per_type": tsolve,
                       "max_drift_ratio": max(d["drift_ratio"] for d in diag.values()), "layers": diag,
                       "quant_wall_s": time.time() - t0, **rec})
    for f in os.listdir(ck):
        if f.startswith(("block", "carry")):
            os.remove(os.path.join(ck, f))
    print(f"done {name}: extra bits {eb:.4f}", flush=True)


if __name__ == "__main__":
    main()
