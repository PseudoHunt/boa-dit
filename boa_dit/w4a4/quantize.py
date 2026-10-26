"""W4A4 PixArt-Σ quantization driver (rule.md, Calibration). Per block l:
  inputs X_l = FP-reference block inputs (calib cache, propagated by the FP blocks);
  sequential replay inside the block, passes (targets): [attn1 q,k,v + attn2 k,v] -> [attn1 out] -> [attn2 q]
    -> [attn2 out] -> [fc1] -> [fc2]; each pass runs the block (earlier layers already W4A4) and stops right after its
    targets' inputs are captured; attn2 k,v see the caption (valid tokens only; one row per prompt and branch);
  per layer: H0 = E[x x^T] (all tokens, fp32), amax, 65,536-token subsample (64 tokens x 1,024 rows, seed = layer id);
  solve (method), replace the nn.Linear with a QLinear;
  then X_{l+1} = FP block l (X_l).
Run: python -m boa_dit.w4a4.quantize --method hsvd --r 4 --device cuda:1
Outputs ckpt/w4a4/q_<arm>.pt (QLinear buffers), results/w4a4/quant/<arm>.json (per-layer info, seconds).
"""
import argparse
import copy
import os
import time

import torch

from boa_dit.common import RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.w4a4 import solvers as SV
from boa_dit.w4a4.common import CK, LAYERS, block_linear, load_pipe
from boa_dit.w4a4.fake import QLinear, set_module

PASSES = [("attn1.to_q", "attn1.to_k", "attn1.to_v"), ("attn1.to_out.0",), ("attn2.to_q",), ("attn2.to_out.0",),
          ("ff.net.0.proj",), ("ff.net.2",)]
# X3 error order (V, out, fc1, fc2, cross-attn, Q, K); the caption-input attn2 k, v ride with the attn2.to_q pass
PASSES_ERR = [("attn1.to_v",), ("attn1.to_out.0",), ("ff.net.0.proj",), ("ff.net.2",), ("attn2.to_q",), ("attn2.to_out.0",),
              ("attn1.to_q", "attn1.to_k")]
CROSS_KV = ("attn2.to_k", "attn2.to_v")
N_SUB_ROWS, N_SUB_TOK = 1024, 64


class StopForward(Exception):
    pass


class LayerStats:
    def __init__(self, d, rows_sel, tok_sel, dev):
        self.H = torch.zeros(d, d, device=dev, dtype=torch.float32)
        self.amax = torch.zeros(d, device=dev)
        self.n = 0
        self.rows_sel, self.tok_sel = rows_sel, tok_sel                       # row id -> slot, token positions
        self.Xs = torch.zeros(len(rows_sel) * tok_sel.shape[1], d, device=dev, dtype=torch.bfloat16)
        self.C = None                                                         # E[x_fp x~^T] (X3 asymmetric target)
        self.Hcfg, self.ncfg = None, 0                                        # E[v v^T], v = w x_c + (1 - w) x_u (CFG arm)

    def add_cfg(self, x, w):
        """x [b, T, d]: calibration rows come in (uncond, cond) pairs (cache order prompt, step, branch)."""
        xf = x.float()
        v = (w * xf[1::2] + (1 - w) * xf[0::2]).reshape(-1, x.shape[-1])
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            g = v.T @ v
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        self.Hcfg = g if self.Hcfg is None else self.Hcfg + g
        self.ncfg += v.shape[0]

    def add_cross(self, x_fp, x):
        a, b = x_fp.float().reshape(-1, x_fp.shape[-1]), x.float().reshape(-1, x.shape[-1])
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            c = a.T @ b
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        self.C = c if self.C is None else self.C + c

    def add(self, x, row_ids, valid=None):
        """x [b, T, d] layer input for calibration rows row_ids; valid: optional bool mask [b, T]."""
        xf = x.float()
        flat = xf.reshape(-1, xf.shape[-1]) if valid is None else xf[valid]
        torch.backends.cuda.matmul.allow_tf32 = True                       # Gram GEMM only (DEVIATION, see REPORT)
        try:
            self.H += flat.T @ flat
        finally:
            torch.backends.cuda.matmul.allow_tf32 = False
        self.n += flat.shape[0]
        self.amax = torch.maximum(self.amax, flat.abs().amax(0))
        for j, r in enumerate(row_ids):
            slot = self.rows_sel.get(int(r))
            if slot is not None:
                t = self.tok_sel[slot]
                self.Xs[slot * N_SUB_TOK:(slot + 1) * N_SUB_TOK] = x[j, t].to(torch.bfloat16)

    def finish(self):
        out = {"H0": self.H / self.n, "amax": self.amax, "Xs": self.Xs, "n_tokens": self.n}
        if self.C is not None:
            out["C"] = self.C / self.n
        if self.Hcfg is not None:                                             # rescaled to the per-branch Gram's trace
            Hc = self.Hcfg / self.ncfg
            out["Hcfg"] = Hc * (out["H0"].diagonal().sum() / Hc.diagonal().sum())
        return out


def subsample(n_rows, T, seed, dev):
    g = torch.Generator().manual_seed(seed)
    rows = torch.randperm(n_rows, generator=g)[:N_SUB_ROWS].tolist()
    tok = torch.stack([torch.randperm(T, generator=g)[:N_SUB_TOK] for _ in rows]).to(dev)
    return {r: i for i, r in enumerate(rows)}, tok


def solve(method, a, W, st, li):
    if method == "rtn":
        return SV.rtn_layer(W, st)
    if method == "gptq":
        return SV.gptq_r0(W, st)
    if method == "svdq":
        return SV.svdquant(W, st, r=a.r)
    if method == "hsvd":
        r = a.r if a.r_config is None else R_CFG[li]
        if "C" in st:                                                         # X3: W_t = W + (H + damp)^-1 (C^T - H) W
            H = st["H0"].double()
            Hd = H + 0.01 * H.diagonal().mean() * torch.eye(H.shape[0], device=H.device, dtype=H.dtype)
            Wt = W.T.double()
            Wt = Wt + torch.linalg.solve(Hd, (st["C"].double().T - H) @ Wt)
            q, inf = SV.hsvdquant(Wt.T.float(), st, r=r, lam_A=a.lam_A, kind=a.subspace, beta=a.beta)
            inf["target_shift"] = float((Wt.T.float() - W).norm() / W.norm())
            return q, inf
        return SV.hsvdquant(W, st, r=r, lam_A=a.lam_A, kind=a.subspace, beta=a.beta, Hsolve=st.get("Hcfg"))
    raise ValueError(method)


R_CFG = {}


def arm_name(a):
    if a.method in ("rtn", "gptq"):
        return f"{a.method}_r0"
    n = f"{a.method}_r{a.r if a.r_config is None else os.path.splitext(os.path.basename(a.r_config))[0]}"
    if a.method == "hsvd":
        n += "_heig" if a.subspace == "heig" else f"_b{a.beta:g}"
        if a.lam_A != 0.05:
            n += f"_lam{a.lam_A:g}"
        if a.order == "err" and a.target == "asym":
            n = n.replace("hsvd_", "hsvdB4_", 1)
        elif a.order != "fwd" or a.target != "sym":
            n += f"_{a.order}_{a.target}"
    for f, tag in (("kv16", "kv16"), ("fc2_unsigned", "u2"), ("range_search", "rs"), ("cfg", "cfg")):
        if getattr(a, f, False):
            n += f"_{tag}"
    if getattr(a, "cfg", False) and getattr(a, "cfg_w", 4.5) != 4.5:
        n += f"amp{a.cfg_w:g}"
    return n + (a.suffix or "")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["rtn", "gptq", "svdq", "hsvd"], required=True)
    ap.add_argument("--r", type=int, default=4)
    ap.add_argument("--subspace", choices=["beta", "heig"], default="beta")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--lam_A", type=float, default=0.05)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--r_config", default=None, help="per-layer ranks (json {'r': {layer_index: r}}); X2")
    ap.add_argument("--cfg", action="store_true", help="H_cfg (v = w x_c + (1 - w) x_u) for the GPTQ metric and L1")
    ap.add_argument("--cfg_w", type=float, default=4.5, help="w in v (CFG-amp: 7.5)")
    ap.add_argument("--kv16", action="store_true", help="cross-attn K/V stay 16-bit (deepcompressor skips attn_add)")
    ap.add_argument("--fc2_unsigned", action="store_true", help="post-GELU fc2 input: shifted unsigned A4 (deepcompressor)")
    ap.add_argument("--range_search", action="store_true", help="weight group scale shrink search in [0.8, 1] inside GPTQ")
    ap.add_argument("--order", choices=["fwd", "err"], default="fwd")
    ap.add_argument("--target", choices=["sym", "asym"], default="sym")
    ap.add_argument("--blocks", type=int, default=28)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    SV.OPTS["range_search"] = a.range_search
    if a.r_config:
        import json
        R_CFG.update({int(k): int(v) for k, v in json.load(open(a.r_config))["r"].items()})
    name = arm_name(a)
    info = {"name": name, "args": vars(a), "layers": {}, "solve_s": {ln: 0.0 for ln in LAYERS}}
    with Run("w4a4", f"quant_{name}", dev, vars(a)) as run:
        t0 = time.time()
        cache = torch.load(os.path.join(CK, "calib_block0.pt"), map_location="cpu")
        X = cache["hidden"].to(dev)                                           # [R, 4096, 1152] bf16
        ENC, TEMB, MASK = cache["enc"].to(dev), cache["temb"].to(dev), cache["enc_mask"].to(dev)
        meta = cache["meta"]                                                  # (prompt, step, branch)
        R = X.shape[0]
        # caption rows: one per (prompt, branch); valid tokens from the mask (bias 0 = attend)
        uniq = {}
        for r in range(R):
            uniq.setdefault((int(meta[r, 0]), int(meta[r, 2])), r)
        cap_rows = sorted(uniq.values())
        pipe = load_pipe(dev)
        tr = pipe.transformer
        del pipe.vae
        qstate = {}
        for l in range(a.blocks if a.method != "rtn" else 0):
            tb = time.time()
            blk = tr.transformer_blocks[l]
            fpb = copy.deepcopy(blk)                                          # FP copy for the next block's inputs

            def run_pass(targets, stats):
                hooks, fp_in = [], {}
                last = targets[-1]
                asym = a.target == "asym"
                for ln in targets:
                    def pre(mod, inp, ln=ln):
                        stats[ln].add(inp[0], cur_rows)
                        if a.cfg:
                            stats[ln].add_cfg(inp[0], a.cfg_w)
                        if asym:
                            stats[ln].add_cross(fp_in[ln], inp[0])
                        if ln == last:
                            raise StopForward
                    hooks.append(block_linear(blk, ln).register_forward_pre_hook(pre))
                    if asym:                                                  # paired FP pass: same block input, FP block
                        def fpre(mod, inp, ln=ln):
                            fp_in[ln] = inp[0]
                            if ln == last:
                                raise StopForward
                        hooks.append(block_linear(fpb, ln).register_forward_pre_hook(fpre))
                try:
                    for i in range(0, R, a.bs):
                        cur_rows = list(range(i, min(R, i + a.bs)))
                        kw = dict(encoder_hidden_states=ENC[i:i + a.bs], encoder_attention_mask=MASK[i:i + a.bs], timestep=TEMB[i:i + a.bs])
                        if asym:
                            try:
                                fpb(X[i:i + a.bs], **kw)
                            except StopForward:
                                pass
                        try:
                            blk(X[i:i + a.bs], **kw)
                        except StopForward:
                            pass
                finally:
                    for h in hooks:
                        h.remove()

            def quantize_layers(names, stats):
                for ln in names:
                    li = l * len(LAYERS) + LAYERS.index(ln)
                    lin = block_linear(blk, ln)
                    if a.kv16 and ln in CROSS_KV:                              # stays 16-bit
                        info["layers"][f"{l}.{ln}"] = {"kept_16bit": True}
                        continue
                    SV.OPTS["unsigned"] = a.fc2_unsigned and ln == "ff.net.2"
                    st = stats[ln].finish()
                    torch.cuda.synchronize(); ts = time.time()
                    SV.GPTQ_DAMP.clear()
                    q, inf = solve(a.method, a, lin.weight.float(), st, li)
                    inf["gptq_damp_used"] = list(SV.GPTQ_DAMP)
                    if a.method == "hsvd" and a.r == 0 and a.r_config is None:     # X2's spectra (side output)
                        inf["alloc"] = SV.alloc_spectrum(lin.weight.float(), st)
                    torch.cuda.synchronize(); s = time.time() - ts
                    shift = SV.act_shift(q["d"], lin.in_features, lin.weight.device) if SV.OPTS["unsigned"] else None
                    set_module(blk, ln, QLinear(lin, q["R"], q["L1"], q["L2"], q["d"], act_shift=shift))
                    q["act_shift"] = shift
                    qstate[f"{l}.{ln}"] = {k: (None if v is None else v.detach().cpu()) for k, v in q.items()}
                    info["layers"][f"{l}.{ln}"] = {**inf, "solve_s": s, "n_tokens": st["n_tokens"]}
                    info["solve_s"][ln] += s

            passes = PASSES if a.order == "fwd" else PASSES_ERR
            kv_pass = passes[0] if a.order == "fwd" else ("attn2.to_q",)
            for targets in passes:
                stats = {}
                for ln in targets + (CROSS_KV if targets == kv_pass else ()):
                    li = l * len(LAYERS) + LAYERS.index(ln)
                    d_in = block_linear(blk, ln).in_features
                    T = ENC.shape[1] if ln in CROSS_KV else X.shape[1]
                    nr = len(cap_rows) if ln in CROSS_KV else R
                    rs, tk = subsample(nr, T, li, dev)
                    stats[ln] = LayerStats(d_in, rs, tk, dev)
                run_pass(targets, stats)
                if targets == kv_pass:
                    for j, r in enumerate(cap_rows):                          # caption inputs, valid tokens only
                        valid = (MASK[r, 0] == 0)[None]
                        for ln in CROSS_KV:
                            x = ENC[r:r + 1]
                            stats[ln].add(x, [j], valid=valid)
                quantize_layers(targets + (CROSS_KV if targets == kv_pass else ()), stats)
                del stats
            for i in range(0, R, a.bs):                                       # FP-reference inputs for block l+1
                X[i:i + a.bs] = fpb(X[i:i + a.bs], encoder_hidden_states=ENC[i:i + a.bs], encoder_attention_mask=MASK[i:i + a.bs],
                                    timestep=TEMB[i:i + a.bs]).to(X.dtype)
            del fpb
            torch.cuda.empty_cache()
            print(f"block {l}: {time.time() - tb:.1f}s", flush=True)
        if a.method == "rtn":                                                 # no statistics needed
            for l in range(a.blocks):
                blk = tr.transformer_blocks[l]
                for ln in LAYERS:
                    lin = block_linear(blk, ln)
                    R = SV.rtn(lin.weight.float().T).T.contiguous()
                    set_module(blk, ln, QLinear(lin, R))
                    qstate[f"{l}.{ln}"] = {"R": R.cpu(), "L1": None, "L2": None, "d": None}
        info["quant_wall_s"] = time.time() - t0
        qlayers = [ln for ln in LAYERS if not (a.kv16 and ln in CROSS_KV)]
        n_par = sum(block_linear(tr.transformer_blocks[0], ln).in_features * block_linear(tr.transformer_blocks[0], ln).out_features
                    for ln in qlayers) * 28
        br = sum(0 if v["L1"] is None else v["L1"].numel() + v["L2"].numel() for v in qstate.values())
        info["branch_params"] = br
        info["bits_per_weight"] = 4 + 16 / 64 + 16 * br / n_par          # over the quantized layers, incl. group scales
        info["bits_per_weight_no_scales"] = 4 + 16 * br / n_par
        info["quantized_layers_per_block"] = qlayers
        if a.blocks == 28:
            atomic_torch_save(qstate, os.path.join(CK, f"q_{name}.pt"))
        info.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "w4a4", "quant", f"{name}.json"), info)
    print(f"done {name}: {info['quant_wall_s'] / 60:.1f} min, bits/weight {info['bits_per_weight']:.3f}", flush=True)


if __name__ == "__main__":
    main()
