"""Part B (rule.md): temporal dither of static A8 activations, PixArt-Σ (1024², DPM-Solver++ 20, guidance 4.5).

Quantizer: static per-tensor asymmetric uint8, one (Δ_t, z_t) per layer per step (both CFG branches), 254 intervals
calibrated so that the dithered grid (code 255 spare) always covers the calibrated range:
    x̂ = (clamp(round((clip(x) + o_t)/Δ_t) + z_t, 0, 255) - z_t)·Δ_t - o_t,   clip to [-z_tΔ_t, (254 - z_t)Δ_t].
Deployment: o_t folds into the preceding affine's per-step β table (adaLN shift / previous bias) and the next layer's
per-step bias table (b - W·o_t), so it costs no runtime work; here it is simulated in fp32.
Layers: w4a4.common.LAYERS without cross-attn K/V (16-bit). W16: bf16 weights; W4: QLinear R (GPTQ on X̃ = X/d) of a
rank-0 corrected-base arm, activations X̃.

  cmd setup     : T5 embeds, design20 cache (32 design prompts x 20 steps x 2 branches, inputs + bf16 output),
                  bf16 MJHQ-64 references (images + final latents, batch 8, seed = index), fp16 control
  cmd calib     : static (Δ_t, z_t) per layer per step on the design set (--weights w16 | <w4 arm>)
  cmd b1        : coherence of static-A8 vs dynamic per-token A8 rounding error (design prompts 0-7) + network C/I
  cmd dsolve    : calibrated dither offsets (design prompts 0-15)
  cmd arm       : MJHQ-64 images + latents for an arm, PSNR / LPIPS / latent MSE vs bf16, diagnostics on design20
Run: python -m boa_dit.temporal.pixart <cmd> [--weights w16] [--dither none|golden|random|calibrated]
"""
import argparse
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.temporal.common import CK, OUT, STAGE, boot_diff, boot_reduction, ci_split
from boa_dit.w4a4 import common as WC
from boa_dit.w4a4.fake import set_module

PCK = os.path.join(CK, "pixart")
QLAYERS = tuple(l for l in WC.LAYERS if l not in ("attn2.to_k", "attn2.to_v"))
NSTEP, GW = 20, WC.GUIDANCE
ALPHAS = tuple(round(1.0 - 0.02 * i, 2) for i in range(21))
N_ALPHA_TOK, N_DITHER_TOK, N_B1_TOK, N_OFF = 256, 1024, 256, 16
SPEC_STEPS = (8, 9, 10)
CHUNK = 8


class _State:
    step = 0
    mode = "fp"          # fp | static | dynamic


STATE = _State()


class StepCounter:
    """Sampling-step index from transformer calls (one call per step: CFG rows batched)."""

    def __init__(self, tr):
        STATE.step = -1
        self.h = tr.register_forward_pre_hook(self._pre)

    def _pre(self, mod, args):
        STATE.step += 1

    def remove(self):
        self.h.remove()


def q_static(x, D, Z, O):
    """x fp32 [..., c]; D, Z, O scalars (0-dim tensors)."""
    xc = torch.clamp(x, -Z * D, (254 - Z) * D)
    q = torch.clamp(torch.round((xc + O) / D) + Z, 0, 255)
    return (q - Z) * D - O


def q_dynamic(x):
    """dynamic per-token asymmetric uint8 (min/max per token)."""
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    D = ((hi - lo) / 255).clamp_min(1e-8)
    Z = torch.round(-lo / D)
    return (torch.clamp(torch.round(x / D) + Z, 0, 255) - Z) * D


class A8Linear(nn.Module):
    """Linear (bf16 weight W, or W4 R on X̃ = X/d) with step-indexed static A8 input quantization."""

    def __init__(self, W, bias, d, tab, offs):
        super().__init__()
        self.register_buffer("W", W.to(torch.bfloat16))
        self.register_buffer("bias", None if bias is None else bias.detach().to(torch.bfloat16))
        self.register_buffer("d", None if d is None else d.float())
        self.register_buffer("D", tab["D"].float())
        self.register_buffer("Z", tab["Z"].float())
        self.register_buffer("O", torch.zeros(NSTEP) if offs is None else offs.float())

    def forward(self, x):
        dt = x.dtype
        xs = x.float() if self.d is None else x.float() / self.d
        if STATE.mode == "static":
            t = STATE.step
            xs = q_static(xs, self.D[t], self.Z[t], self.O[t])
        elif STATE.mode == "dynamic":
            xs = q_dynamic(xs)
        return F.linear(xs.to(torch.bfloat16), self.W, self.bias).to(dt)


def w4_layers(arm):
    """{'{l}.{ln}': (R, d)} of a w4a4 quantized arm (ckpt/w4a4/q_<arm>.pt)."""
    qs = torch.load(os.path.join(WC.CK, f"q_{arm}.pt"), map_location="cpu")
    return {k: (q["R"], q["d"]) for k, q in qs.items() if k.split(".", 1)[1] in QLAYERS}


def static_path(weights):
    return os.path.join(PCK, f"static_{weights}.pt")


def install(tr, weights, dither="none"):
    """Replace QLAYERS with A8Linear (static tables of `weights`, offsets of `dither`)."""
    tabs = torch.load(static_path(weights), map_location="cpu")
    offs = offsets(weights, dither, tabs)
    w4 = None if weights == "w16" else w4_layers(weights)
    dev = next(tr.parameters()).device
    for l, blk in enumerate(tr.transformer_blocks):
        for ln in QLAYERS:
            key = f"{l}.{ln}"
            lin = WC.block_linear(blk, ln)
            W, d = (lin.weight.data, None) if w4 is None else w4[key]
            o = None if offs is None else offs[key]
            set_module(blk, ln, A8Linear(W.to(dev), lin.bias, None if d is None else d.to(dev), tabs[key], o).to(dev))


def offsets(weights, dither, tabs):
    """o_t per layer: golden frac(t·0.6180339)·Δ_t; random U[0, Δ_t) (generator seed 0, keys in sorted order);
    calibrated from dsolve."""
    if dither == "none":
        return None
    keys = sorted(tabs)
    if dither == "golden":
        fr = torch.tensor([math.modf(t * 0.6180339)[0] for t in range(NSTEP)])
        return {k: fr * tabs[k]["D"] for k in keys}
    if dither == "period2":                                                   # exploratory: o_t = (t mod 2)·Δ_t/2
        fr = torch.tensor([(t % 2) / 2 for t in range(NSTEP)])
        return {k: fr * tabs[k]["D"] for k in keys}
    if dither == "random":
        g = torch.Generator().manual_seed(0)
        return {k: torch.rand(NSTEP, generator=g) * tabs[k]["D"] for k in keys}
    if dither == "calibrated":
        sol = torch.load(os.path.join(PCK, f"dither_{weights}.pt"), map_location="cpu")["k"]
        return {k: sol[k].float() / N_OFF * tabs[k]["D"] for k in keys}
    raise ValueError(dither)


# --------------------------------------------------------------------------------------------------------------- setup
@torch.no_grad()
def sample_lat(pipe, e, idx, dev):
    return WC.sample(pipe, e, idx, dev, output_type="latent")


@torch.no_grad()
def decode(pipe, lat):
    img = pipe.vae.decode(lat.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    img = pipe.image_processor.postprocess(img, output_type="pt")
    return (img.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu()


@torch.no_grad()
def design20(dev, bs=8):
    path = os.path.join(PCK, "design20_cache.pt")
    if os.path.exists(path):
        return
    e = WC.load_embeds("design")
    pipe = WC.load_pipe(dev)
    rec = {k: [] for k in ("latent", "t", "prompt", "branch", "step", "eps")}
    for i in range(0, len(e["prompts"]), bs):
        idx = list(range(i, min(len(e["prompts"]), i + bs)))
        sc = StepCounter(pipe.transformer)

        def hook(mod, args, kwargs, out, idx=idx):
            hs = args[0] if args else kwargs["hidden_states"]
            B = hs.shape[0] // 2
            o = out[0] if isinstance(out, tuple) else out.sample
            rec["latent"].append(hs.to(torch.bfloat16).cpu()); rec["t"].append(kwargs["timestep"].float().cpu())
            rec["eps"].append(o[:, :4].to(torch.bfloat16).cpu())
            rec["prompt"].append(torch.tensor([idx[r % B] for r in range(2 * B)]))
            rec["branch"].append(torch.tensor([r // B for r in range(2 * B)]))
            rec["step"].append(torch.full((2 * B,), STATE.step))
        h = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        try:
            sample_lat(pipe, e, idx, dev)
        finally:
            h.remove(); sc.remove()
    out = {k: torch.cat(v) for k, v in rec.items()}
    order = torch.argsort(out["step"] * 100000 + out["prompt"] * 2 + out["branch"])   # batches share a step
    atomic_torch_save({k: v[order] for k, v in out.items()}, path)


def img_dir(arm):
    return os.path.join(PCK, "img", arm)


@torch.no_grad()
def gen_arm(arm, pipe, dev, n=64):
    """MJHQ prompts mjhq1k[:n], batch 8, seed = index: uint8 images + final latents, chunked and resumable."""
    d = img_dir(arm)
    os.makedirs(d, exist_ok=True)
    e = WC.load_embeds("mjhq1k")
    for i in range(0, n, CHUNK):
        f = os.path.join(d, f"{i:05d}.pt")
        if os.path.exists(f):
            continue
        idx = list(range(i, min(n, i + CHUNK)))
        sc = StepCounter(pipe.transformer)
        try:
            lat = sample_lat(pipe, e, idx, dev)
        finally:
            sc.remove()
        atomic_torch_save({"img": decode(pipe, lat), "lat": lat.float().cpu()}, f)
        if arm == "bf16":                                                     # w4a4.gen layout (dtype_control reads it)
            from boa_dit.w4a4 import gen
            os.makedirs(gen.img_dir("bf16"), exist_ok=True)
            atomic_torch_save(torch.load(f)["img"], os.path.join(gen.img_dir("bf16"), f"{i:05d}.pt"))


def load_gen(arm, n=64):
    ch = [torch.load(os.path.join(img_dir(arm), f"{i:05d}.pt")) for i in range(0, n, CHUNK)]
    return torch.cat([c["img"] for c in ch])[:n], torch.cat([c["lat"] for c in ch])[:n]


@torch.no_grad()
def setup(dev):
    with Run(STAGE, "setup_pixart_embeds_design20", dev, {}):
        WC.embed_all(dev)
        design20(dev)
    with Run(STAGE, "setup_pixart_bf16_refs64", dev, {}):
        pipe = WC.load_pipe(dev)
        gen_arm("bf16", pipe, dev)
        del pipe
        torch.cuda.empty_cache()
    from boa_dit.w4a4 import dtype_control as DC                              # unchanged control, logged here
    import sys
    DC.Run = lambda stage, name, d, cfg: Run(STAGE, f"setup_pixart_{name}", d, cfg)
    DC.RESULTS = os.path.join(OUT, "control")
    sys.argv = [sys.argv[0], "--n", "64", "--device", dev]
    DC.main()


# ---------------------------------------------------------------------------------------------------------- calibration
def tok_idx(T, n, seed, dev):
    return torch.randperm(T, generator=torch.Generator().manual_seed(seed))[:n].to(dev)


def layer_inputs(tr, fn, w4=None):
    """Pre-hooks on the bf16 QLAYERS calling fn(key, x̃ fp32 [rows, T, c], lin)."""
    hs = []
    for l, blk in enumerate(tr.transformer_blocks):
        for ln in QLAYERS:
            key = f"{l}.{ln}"
            lin = WC.block_linear(blk, ln)
            d = None if w4 is None else w4[key][1].to(lin.weight.device).float()

            def pre(mod, inp, key=key, d=d):
                x = inp[0].float()
                fn(key, x if d is None else x / d, mod)
            hs.append(lin.register_forward_pre_hook(pre))
    return hs


@torch.no_grad()
def run_design(pipe, dev, prompts, fn, w4=None, bs=4):
    e = WC.load_embeds("design")
    hs = layer_inputs(pipe.transformer, fn, w4)
    try:
        for i in range(0, len(prompts), bs):
            idx = list(prompts[i:i + bs])
            sc = StepCounter(pipe.transformer)
            try:
                sample_lat(pipe, e, idx, dev)
            finally:
                sc.remove()
            fn("__end_batch__", None, None)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def calib(weights, dev):
    """Per (layer, step): range [α·min, α·max] over all design tokens (32 prompts, both branches), α in ALPHAS minimizing
    the summed squared error on 256 fixed random tokens per row; Δ = (hi - lo)/254, z = round(-lo/Δ)."""
    w4 = None if weights == "w16" else w4_layers(weights)
    pipe = WC.load_pipe(dev)
    mn, mx = {}, {}

    def p1(key, x, mod):
        if x is None:
            return
        t = STATE.step
        a, b = float(x.amin()), float(x.amax())
        k = (key, t)
        mn[k], mx[k] = min(mn.get(k, a), a), max(mx.get(k, b), b)
    with Run(STAGE, f"calib_static_{weights}", dev, {"weights": weights, "alphas": ALPHAS, "n_tok": N_ALPHA_TOK}) as run:
        run_design(pipe, dev, range(32), p1, w4)
        al = torch.tensor(ALPHAS, device=dev).view(-1, 1, 1)
        err = {}
        idx_cache = {}
        kidx = {k: i for i, k in enumerate(sorted({k for k, _ in mn}))}

        def p2(key, x, mod):
            if x is None:
                return
            t = STATE.step
            if key not in idx_cache:
                idx_cache[key] = tok_idx(x.shape[1], N_ALPHA_TOK, kidx[key], dev)
            xs = x[:, idx_cache[key]].reshape(1, -1, x.shape[-1])            # [1, rows*256, c]
            lo, hi = al * mn[(key, t)], al * mx[(key, t)]
            D = (hi - lo) / 254
            Z = torch.round(-lo / D)
            e2 = torch.stack([(q_static(xs, D[i], Z[i], torch.zeros((), device=dev)) - xs).pow(2).sum() for i in range(len(ALPHAS))])
            err[(key, t)] = err.get((key, t), 0) + e2.double().cpu()
        run_design(pipe, dev, range(32), p2, w4)
        tabs = {}
        keys = sorted({k for k, _ in mn})
        for key in keys:
            D, Z, A = [], [], []
            for t in range(NSTEP):
                j = int(torch.argmin(err[(key, t)]))
                lo, hi = ALPHAS[j] * mn[(key, t)], ALPHAS[j] * mx[(key, t)]
                d = (hi - lo) / 254
                D.append(d); Z.append(round(-lo / d)); A.append(ALPHAS[j])
            tabs[key] = {"D": torch.tensor(D, dtype=torch.float64), "Z": torch.tensor(Z, dtype=torch.float64),
                         "alpha": torch.tensor(A), "min": torch.tensor([mn[(key, t)] for t in range(NSTEP)]),
                         "max": torch.tensor([mx[(key, t)] for t in range(NSTEP)])}
        atomic_torch_save(tabs, static_path(weights))
        rec = run.record()
    al_all = torch.cat([v["alpha"] for v in tabs.values()])
    atomic_write_json(os.path.join(OUT, f"calib_static_{weights}.json"),
                      {"weights": weights, "alpha_mean": float(al_all.mean()), "alpha_hist": {str(a): int((al_all == a).sum()) for a in ALPHAS},
                       "token_note": f"min/max over all tokens; alpha MSE on {N_ALPHA_TOK} fixed random tokens per row", **rec})


# ------------------------------------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def net_diag(tr, dev, mode):
    """δ_g = 4.5 δ_c - 3.5 δ_u on the design20 cache vs the cached bf16 output: per-step error, C/I, spectrum."""
    from boa_dit.w4a4.band_gain import band_masks
    from boa_dit.w4a4.screen import text_for
    c = torch.load(os.path.join(PCK, "design20_cache.pt"), map_location="cpu")
    e = WC.load_embeds("design")
    R = len(c["t"])
    err = torch.zeros(R, 4, 128, 128)
    STATE.mode = mode
    for i in range(0, R, 16):
        sl = slice(i, min(R, i + 16))
        steps = set(c["step"][sl].tolist())
        assert len(steps) == 1
        STATE.step = steps.pop()
        pe, pm = text_for(e, c["prompt"][sl], c["branch"][sl], dev)
        o = tr(c["latent"][sl].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][sl].to(dev),
               added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0].float()[:, :4]
        err[sl] = (o - c["eps"][sl].to(dev).float()).cpu()
    STATE.mode = "fp"
    P = int(c["prompt"].max()) + 1
    idx = {(int(c["prompt"][j]), int(c["step"][j]), int(c["branch"][j])): j for j in range(R)}
    ec = torch.stack([torch.stack([err[idx[(p, s, 1)]] for s in range(NSTEP)]) for p in range(P)])
    eu = torch.stack([torch.stack([err[idx[(p, s, 0)]] for s in range(NSTEP)]) for p in range(P)])
    eg = (GW * ec.double() - (GW - 1) * eu.double())                          # [P, 20, 4, 128, 128]
    masks = band_masks()
    spec = []
    for s in SPEC_STEPS:
        Pw = torch.fft.fft2(eg[:, s]).abs().pow(2).sum((0, 1))
        spec.append([float(Pw[m].sum() / Pw.sum()) for m in masks])
    spec = np.mean(spec, axis=0).tolist()
    return {"per_step_err": eg.pow(2).flatten(2).sum(2).mean(0).tolist(), **ci_split(eg),
            "band_spectrum_steps8_10": spec, "band1_share_steps8_10": spec[0]}


@torch.no_grad()
def b1(dev):
    """Layer-level ρ_adj, coherent fraction and output-error share for static A8 (W16 tables) and dynamic per-token
    A8, on the bf16 trajectory of design prompts 0-7 (2 prompts per run), 256 fixed tokens per layer; then network
    C/I on design20 for both."""
    tabs = torch.load(static_path("w16"), map_location="cpu")
    pipe = WC.load_pipe(dev)
    Q = ("static", "dynamic")
    acc = {}                                                                  # (q, key) -> accumulators
    tok = {}

    def get(qn, key, shape):
        if (qn, key) not in acc:
            acc[(qn, key)] = {"S": None, "prev": None, "SS": 0.0, "coh_num": 0.0, "coh_den": 0.0, "out_e": 0.0,
                              "rho": torch.zeros(NSTEP - 1, 6, dtype=torch.float64)}
        return acc[(qn, key)]

    def fn(key, x, mod):
        if key == "__end_batch__":
            for a in acc.values():
                a["coh_num"] += float(a["S"].double().pow(2).sum()); a["coh_den"] += a["SS"]
                a["S"], a["prev"], a["SS"] = None, None, 0.0
            return
        t = STATE.step
        if key not in tok:
            tok[key] = tok_idx(x.shape[1], N_B1_TOK, list(tabs).index(key), dev)
        xs = x[:, tok[key]]
        tb = tabs[key]
        for qn in Q:
            eq = (q_static(xs, tb["D"][t].to(dev).float(), tb["Z"][t].to(dev).float(), torch.zeros((), device=dev)) if qn == "static"
                  else q_dynamic(xs)) - xs
            a = get(qn, key, eq.shape)
            a["S"] = eq.clone() if a["S"] is None else a["S"] + eq
            a["SS"] += float(eq.double().pow(2).sum())
            a["out_e"] += float((eq.to(torch.bfloat16) @ mod.weight.T).float().pow(2).sum())
            if a["prev"] is not None:
                p, c = a["prev"].double().flatten(), eq.double().flatten()
                a["rho"][t - 1] += torch.tensor([float(p.sum()), float(c.sum()), float((p * p).sum()), float((c * c).sum()),
                                                 float((p * c).sum()), float(p.numel())], dtype=torch.float64)
            a["prev"] = eq
    res = {"prompt_set": "design 0-7", "n_tok": N_B1_TOK, "layers": {q: {} for q in Q}}
    with Run(STAGE, "b1_coherence", dev, {"prompts": "design 0-7"}) as run:
        run_design(pipe, dev, range(8), fn, bs=2)
        for (qn, key), a in acc.items():
            r = a["rho"]
            sx, sy, sxx, syy, sxy, n = (r[:, i] for i in range(6))
            rho = ((sxy / n - sx / n * sy / n) / torch.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))).tolist()
            res["layers"][qn][key] = {"rho_adj": rho, "rho_adj_mean": float(np.mean(rho)), "coherent_frac": a["coh_num"] / (NSTEP * a["coh_den"]),
                                      "out_err_energy": a["out_e"]}
        for qn in Q:
            L = res["layers"][qn]
            tot = sum(v["out_err_energy"] for v in L.values())
            for v in L.values():
                v["out_err_share"] = v["out_err_energy"] / tot
            order = sorted(L, key=lambda k: -L[k]["out_err_share"])
            top, cum = [], 0.0
            for k in order:
                top.append(k); cum += L[k]["out_err_share"]
                if cum >= 0.5:
                    break
            w = np.array([L[k]["out_err_share"] for k in top])
            res[f"{qn}_top_half_layers"] = top
            res[f"{qn}_top_half_rho_adj_weighted"] = float((w * np.array([L[k]["rho_adj_mean"] for k in top])).sum() / w.sum())
            res[f"{qn}_top_half_coherent_frac_weighted"] = float((w * np.array([L[k]["coherent_frac"] for k in top])).sum() / w.sum())
            by = {}
            for k, v in L.items():
                by.setdefault(k.split(".", 1)[1], []).append(v)
            res[f"{qn}_by_type"] = {t: {"rho_adj_mean": float(np.mean([v["rho_adj_mean"] for v in vs])),
                                        "coherent_frac": float(np.mean([v["coherent_frac"] for v in vs])),
                                        "out_err_share": float(sum(v["out_err_share"] for v in vs)),
                                        "rho_adj_by_step": np.mean([v["rho_adj"] for v in vs], axis=0).tolist()} for t, vs in by.items()}
        atomic_write_json(os.path.join(OUT, "b1_layers_partial.json"), res)
        install(pipe.transformer, "w16", "none")
        res["net_fp_floor"] = net_diag(pipe.transformer, dev, "fp")             # batch-layout floor, no quantization
        for qn in Q:
            res[f"net_{qn}"] = net_diag(pipe.transformer, dev, qn)
        s = res
        res["stop"] = bool(s["net_static"]["C_over_I"] < 0.05 and s["static_top_half_rho_adj_weighted"] < 0.3)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "b1.json"), res)
    print({k: v for k, v in res.items() if k not in ("layers", "config") and not k.endswith("by_type")}, flush=True)


# ------------------------------------------------------------------------------------------------- calibrated dither
@torch.no_grad()
def dsolve(weights, dev):
    """Per layer: o_t ∈ {kΔ_t/16} minimizing J = Σ_p ||W Σ_t ē_{p,t}(o_t)||², ē = token-mean rounding error of design
    (prompt, branch) row p (prompts 0-15) on 1,024 fixed random tokens per row; coordinate descent from o ≡ 0, 2 sweeps.
    W = the bf16 weight (W16) or the dequantized W4 R (on X̃)."""
    tabs = torch.load(static_path(weights), map_location="cpu")
    w4 = None if weights == "w16" else w4_layers(weights)
    pipe = WC.load_pipe(dev)
    keys = sorted(tabs)
    M = {k: torch.zeros(NSTEP, 32, N_OFF, 0) for k in keys}                  # filled as [20, rows, 16, d_out] on CPU
    tok, rowbase = {}, {"b": 0}
    offs = (torch.arange(N_OFF, device=dev).float() / N_OFF).view(-1, 1, 1)

    def fn(key, x, mod):
        if key == "__end_batch__":
            rowbase["b"] += 8                                                 # bs 4 prompts x 2 branches
            return
        t = STATE.step
        if key not in tok:
            tok[key] = tok_idx(x.shape[1], N_DITHER_TOK, keys.index(key), dev)
        xs = x[:, tok[key]]                                                   # [rows, n, c]
        D, Z = tabs[key]["D"][t].to(dev).float(), tabs[key]["Z"][t].to(dev).float()
        eb = torch.stack([(q_static(xs, D, Z, o * D) - xs).mean(1) for o in offs.flatten()], 1)   # [rows, 16, c]
        W = mod.weight.float() if w4 is None else w4[key][0].to(dev).float()
        m = (eb @ W.T).cpu()                                                  # [rows, 16, d_out]
        if M[key].shape[-1] == 0:
            M[key] = torch.zeros(NSTEP, 32, N_OFF, m.shape[-1])
        M[key][t, rowbase["b"]:rowbase["b"] + m.shape[0]] = m
    res = {"weights": weights, "layers": {}}
    with Run(STAGE, f"dsolve_{weights}", dev, {"weights": weights, "n_tok": N_DITHER_TOK}) as run:
        run_design(pipe, dev, range(16), fn, w4, bs=4)
        sol = {}
        for key in keys:
            Mk = M[key].to(dev).double()                                      # [20, 32, 16, d]
            k = torch.zeros(NSTEP, dtype=torch.long, device=dev)
            cur = lambda: Mk[torch.arange(NSTEP), :, k].sum(0)                # [32, d]
            J0 = float(cur().pow(2).sum())
            for _ in range(2):
                for t in range(NSTEP):
                    rest = cur() - Mk[t, :, k[t]]
                    J = (rest[:, None, :] + Mk[t]).pow(2).sum((0, 2))         # [16]
                    k[t] = int(torch.argmin(J))
            J1 = float(cur().pow(2).sum())
            sol[key] = k.cpu()
            res["layers"][key] = {"J0": J0, "J1": J1, "decrease": 1 - J1 / J0, "k": k.tolist()}
            M[key] = None
        atomic_torch_save({"k": sol}, os.path.join(PCK, f"dither_{weights}.pt"))
        allk = torch.cat(list(sol.values()))
        res["offset_hist"] = torch.bincount(allk, minlength=N_OFF).tolist()
        res["mean_decrease"] = float(np.mean([v["decrease"] for v in res["layers"].values()]))
        res["total_J0"], res["total_J1"] = sum(v["J0"] for v in res["layers"].values()), sum(v["J1"] for v in res["layers"].values())
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, f"dsolve_{weights}.json"), res)
    print({k: v for k, v in res.items() if k not in ("layers", "config")}, flush=True)


# --------------------------------------------------------------------------------------------------------------- arms
def arm_name(weights, dither):
    return f"{weights}_a8s_{dither}"


@torch.no_grad()
def arm(weights, dither, dev, n=64):
    name = arm_name(weights, dither)
    with Run(STAGE, f"arm_{name}", dev, {"weights": weights, "dither": dither, "n": n}) as run:
        pipe = WC.load_pipe(dev)
        install(pipe.transformer, weights, dither)
        STATE.mode = "static"
        gen_arm(name, pipe, dev, n)
        STATE.mode = "fp"
        diag = net_diag(pipe.transformer, dev, "static")
        del pipe
        torch.cuda.empty_cache()
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        ri, rl = load_gen("bf16", n)
        qi, ql = load_gen(name, n)
        psnr, lp = [], []
        for i in range(n):
            a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
            psnr.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
            lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
        rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
        res = {"arm": name, "weights": weights, "dither": dither, "n": n, "prompt_set": "mjhq1k[:64]",
               "psnr_mean": float(np.mean(psnr)), "lpips_mean": float(np.mean(lp)), "latent_rel_mse_mean": float(np.mean(rel)),
               "psnr_per_image": psnr, "lpips_per_image": lp, "latent_rel_mse_per_image": rel, "diag": diag}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "eval", f"pixart_{name}.json"), res)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "calib", "b1", "dsolve", "arm"])
    ap.add_argument("--weights", default="w16", help="w16 | a w4a4 arm name (ckpt/w4a4/q_<arm>.pt)")
    ap.add_argument("--dither", default="none", choices=["none", "golden", "random", "calibrated", "period2"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    os.makedirs(PCK, exist_ok=True)
    {"setup": lambda: setup(a.device), "calib": lambda: calib(a.weights, a.device), "b1": lambda: b1(a.device),
     "dsolve": lambda: dsolve(a.weights, a.device), "arm": lambda: arm(a.weights, a.dither, a.device)}[a.cmd]()


if __name__ == "__main__":
    main()


# ======================================================================================================== B′ (rule.md)
class A8LinearB(A8Linear):
    """A8Linear with a table of any bit-width (qmax buffer)."""

    def __init__(self, W, bias, d, tab, offs):
        super().__init__(W, bias, d, tab, offs)
        self.qmax = int(tab["qmax"])

    def forward(self, x):
        dt = x.dtype
        xs = x.float() if self.d is None else x.float() / self.d
        if STATE.mode == "static":
            t = STATE.step
            xs = ACT.q_static(xs, self.D[t], self.Z[t], self.O[t], self.qmax)
        elif STATE.mode == "dynamic":
            xs = ACT.q_dynamic(xs)
        return F.linear(xs.to(torch.bfloat16), self.W, self.bias).to(dt)


from boa_dit.temporal import act as ACT  # noqa: E402


def bpath(kind):
    return os.path.join(PCK, f"static_b_{kind}.pt")


def install_b(tr, kind, dither="none"):
    tabs = torch.load(bpath(kind), map_location="cpu")
    fr = ACT.offset_frac(dither, NSTEP)
    dev = next(tr.parameters()).device
    for l, blk in enumerate(tr.transformer_blocks):
        for ln in QLAYERS:
            key = f"{l}.{ln}"
            lin = WC.block_linear(blk, ln)
            tb = tabs[key]
            set_module(blk, ln, A8LinearB(lin.weight.data, lin.bias, None, tb, fr * tb["D"]).to(dev))


@torch.no_grad()
def calib_b(dev):
    """step8 / single8 / single6 tables on the design set (32 prompts, bf16 trajectory), α ∈ 1.00..0.20, 256 tokens/row."""
    pipe = WC.load_pipe(dev)
    cal = ACT.Calib(dev)
    with Run(STAGE, "bp_calib_pixart", dev, {"alphas": ACT.ALPHAS, "alpha_search": "65,536-bin histogram, all tokens"}) as run:
        run_design(pipe, dev, range(32), lambda key, x, mod: None if x is None else cal.add_minmax(key, STATE.step, x))
        run_design(pipe, dev, range(32), lambda key, x, mod: None if x is None else cal.add_hist(key, STATE.step, x))
        atomic_torch_save({"mn": cal.mn, "mx": cal.mx, "hist": cal.hist}, os.path.join(PCK, "calib_hist.pt"))
        rec = run.record()
    retable_b(dev, lo_min=False, cal=cal, rec=rec)


GELU_TYPES = ("ff.net.2",)


def retable_b(dev, lo_min, cal=None, rec=None):
    if cal is None:
        st = torch.load(os.path.join(PCK, "calib_hist.pt"), map_location="cpu")
        cal = ACT.Calib(dev)
        cal.mn, cal.mx, cal.hist = st["mn"], st["mx"], st["hist"]
    tabs = cal.tables(NSTEP, lo_min_keys=GELU_TYPES if lo_min else ())
    for kind, tb in tabs.items():
        atomic_torch_save(tb, bpath(kind))
    summ = {kind: {"alpha_hi_mean": float(torch.cat([v["alpha"] for v in tb.values()]).mean()),
                   "alpha_lo_mean": float(torch.cat([v["alpha_lo"] for v in tb.values()]).mean())} for kind, tb in tabs.items()}
    out = {"lo_min_post_gelu": lo_min, "summary": summ, "clip_rate_by_type": ACT.clip_summary(tabs), **(rec or {})}
    atomic_write_json(os.path.join(OUT, f"bp_calib_pixart{'_lomin' if lo_min else ''}.json"), out)
    print(out["clip_rate_by_type"], flush=True)


@torch.no_grad()
def b1p(dev):
    """B1′ on PixArt: layer-level variants on design prompts 0-7 (1 prompt per run, 256 fixed tokens per layer), then
    network C/I on design20 for single8 / step8 / single6 / dynamic8."""
    tabs = {k: torch.load(bpath(k), map_location="cpu") for k in ACT.KINDS}
    pipe = WC.load_pipe(dev)
    coh = ACT.Coherence(tabs, NSTEP, dev)
    keys = sorted(tabs["single8"])
    tok = {}

    def fn(key, x, mod):
        if key == "__end_batch__":
            coh.end_batch()
            return
        if key not in tok:
            tok[key] = tok_idx(x.shape[1], N_B1_TOK, keys.index(key), dev)
        coh.step(key, STATE.step, x[:, tok[key]], mod.weight)
    with Run(STAGE, "bp_b1_pixart", dev, {"prompts": "design 0-7"}) as run:
        run_design(pipe, dev, range(8), fn, bs=1)
        res = {"bed": "pixart", "prompt_set": "design", "layer": coh.result(), "net": {}}
        atomic_write_json(os.path.join(OUT, "bp_b1_pixart_layers_partial.json"), res)
        for kind, mode in (("single8", "static"), ("step8", "static"), ("single6", "static"), ("single8", "dynamic")):
            pipe = WC.load_pipe(dev)
            install_b(pipe.transformer, kind)
            res["net"]["dynamic8" if mode == "dynamic" else kind] = net_diag(pipe.transformer, dev, mode)
            del pipe
            torch.cuda.empty_cache()
        res["gate"] = ACT.gate(res)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "bp_b1_pixart.json"), res)
    print({k: res["net"][k]["C_over_I"] for k in res["net"]}, res["gate"], flush=True)


@torch.no_grad()
def arm_b(kind, dither, dev, n=64):
    name = f"bp_{kind}_{dither}"
    with Run(STAGE, f"arm_{name}", dev, {"kind": kind, "dither": dither, "n": n}) as run:
        pipe = WC.load_pipe(dev)
        install_b(pipe.transformer, kind, dither)
        STATE.mode = "static"
        gen_arm(name, pipe, dev, n)
        STATE.mode = "fp"
        diag = net_diag(pipe.transformer, dev, "static")
        del pipe
        torch.cuda.empty_cache()
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        ri, rl = load_gen("bf16", n)
        qi, ql = load_gen(name, n)
        psnr, lp = [], []
        for i in range(n):
            a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
            psnr.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
            lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
        rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
        res = {"arm": name, "kind": kind, "dither": dither, "n": n, "prompt_set": "mjhq1k[:64]",
               "psnr_mean": float(np.mean(psnr)), "lpips_mean": float(np.mean(lp)), "latent_rel_mse_mean": float(np.mean(rel)),
               "psnr_per_image": psnr, "lpips_per_image": lp, "latent_rel_mse_per_image": rel, "diag": diag}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "eval", f"pixart_{name}.json"), res)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)


def main_b():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "b1", "arm", "retable_lomin", "s3"])
    ap.add_argument("--kind", default="single8", choices=list(ACT.KINDS))
    ap.add_argument("--dither", default="none", choices=list(ACT.DITHERS))
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--tag", default="")
    ap.add_argument("--compress", default=None, choices=[None, "dct4"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    os.makedirs(PCK, exist_ok=True)
    {"calib": lambda: calib_b(a.device), "b1": lambda: b1p(a.device), "arm": lambda: arm_b(a.kind, a.dither, a.device),
     "retable_lomin": lambda: retable_b(a.device, True),
     "s3": lambda: arm_s3(a.kind, a.dither, a.feedback, a.device, a.n, a.tag, a.compress)}[a.cmd]()


# ================================================================================== stage 3 (rule_feedback.md), PixArt
class A8LinearFB(A8LinearB):
    """A8LinearB plus optional full temporal error feedback: per sample state s (fp16), reset at step 0;
    u = x + s, ũ = clip(u) to [-ZΔ, (qmax - Z)Δ], q = Q(ũ) on codes 0..qmax, q̂ = bf16(q) (consumed), r⁰ = ũ - q̂,
    s = fp16(clamp(r⁰, ±Δ/2)). Clip error is never fed back."""

    def __init__(self, W, bias, d, tab, offs, feedback=False, U=None):
        super().__init__(W, bias, d, tab, offs)
        self.feedback, self.r, self.U = feedback, None, U                     # U [tokens, k]: compressed spatial state (block 5)
        self.stats = torch.zeros(4, dtype=torch.float64)                    # clipped, clamped, elements, -
        self.clipE = torch.zeros(NSTEP // 4, 4, dtype=torch.float64)        # per 4-step bucket: nominal n, E, representable n, E

    def forward(self, x):
        if not (self.feedback and STATE.mode == "static"):
            return super().forward(x)
        dt = x.dtype
        t = STATE.step
        xs = x.float()
        shp = xs.shape if self.U is None or xs.shape[1] != self.U.shape[0] else (xs.shape[0], self.U.shape[1], xs.shape[2])
        if t == 0 or self.r is None or tuple(self.r.shape) != tuple(shp):
            self.r = torch.zeros(shp, device=xs.device, dtype=torch.float16)
        D, Z = self.D[t], self.Z[t]
        lo, hi, hi_rep = -Z * D, (self.qmax - 1 - Z) * D, (self.qmax - Z) * D     # nominal top / representable top (code qmax)
        comp = self.U is not None and tuple(shp) != tuple(xs.shape)
        s_prev = torch.einsum("tk,rkc->rtc", self.U, self.r.float()) if comp else self.r.float()
        u = xs + s_prev                                                       # u_t = x_t + s_{t-1}
        uc = torch.minimum(torch.maximum(u, lo), hi_rep)                      # ũ_t = clip to the representable range
        q = ACT.q_static(uc, D, Z, torch.zeros((), device=xs.device), self.qmax, hi_code=self.qmax)
        qh = q.to(torch.bfloat16).float()                                     # q̂_t: what the next linear consumes (bf16)
        r0 = uc - qh                                                          # r⁰_t: residual of the effective quantizer
        r = r0.clamp(-D / 2, D / 2)                                           # guard clamp (binding counted)
        nom = torch.minimum(torch.maximum(u, lo), hi)
        if self.stats.device != xs.device:                                    # counters on the GPU: no per-call sync
            self.stats, self.clipE = self.stats.to(xs.device), self.clipE.to(xs.device)
        self.stats += torch.stack([(u != uc).sum().double(), (r != r0).sum().double(),
                                   torch.tensor(float(u.numel()), device=xs.device, dtype=torch.float64), torch.zeros((), device=xs.device, dtype=torch.float64)])
        b = t // 4
        self.clipE[b] += torch.stack([(u != nom).sum().double(), (nom - u).double().pow(2).sum(),
                                      (u != uc).sum().double(), (uc - u).double().pow(2).sum()])
        self.r = (torch.einsum("tk,rtc->rkc", self.U, r) if comp else r).to(torch.float16)   # s_t: fp16, no damping
        q = qh
        return F.linear(q.to(torch.bfloat16), self.W, self.bias).to(dt)


def dct4_basis(g, dev):
    """orthonormal 2-D DCT-II modes (u, v) in {0,1}^2 on a g x g token grid (row-major), [g*g, 4]."""
    i = torch.arange(g, dtype=torch.float64)
    c = lambda f: (torch.ones(g, dtype=torch.float64) / g ** 0.5 if f == 0 else (2.0 / g) ** 0.5 * torch.cos(torch.pi * (2 * i + 1) * f / (2 * g)))
    return torch.stack([torch.outer(c(u), c(v)).flatten() for u in (0, 1) for v in (0, 1)], 1).float().to(dev)


def install_fb(tr, kind, dither="none", feedback=False, compress=None):
    tabs = torch.load(bpath(kind), map_location="cpu")
    keys = sorted(tabs)
    dev = next(tr.parameters()).device
    U = dct4_basis(64, dev) if compress == "dct4" else None
    mods = []
    for l, blk in enumerate(tr.transformer_blocks):
        for ln in QLAYERS:
            key = f"{l}.{ln}"
            lin = WC.block_linear(blk, ln)
            tb = tabs[key]
            fr = ACT.offset_frac(dither, NSTEP, keys.index(key))
            m = A8LinearFB(lin.weight.data, lin.bias, None, tb, fr * tb["D"], feedback, U).to(dev)
            set_module(blk, ln, m)
            mods.append(m)
    return mods


@torch.no_grad()
def gen_b1(arm, pipe, dev, n=64):
    """mjhq1k[:n], batch 1 (seed = index), chunk files of 8 (images + latents) under img/<arm>."""
    d = img_dir(arm)
    os.makedirs(d, exist_ok=True)
    e = WC.load_embeds("mjhq1k")
    for i in range(0, n, CHUNK):
        f = os.path.join(d, f"{i:05d}.pt")
        if os.path.exists(f):
            continue
        imgs, lats = [], []
        for j in range(i, min(n, i + CHUNK)):
            sc = StepCounter(pipe.transformer)
            try:
                lat = sample_lat(pipe, e, [j], dev)
            finally:
                sc.remove()
            imgs.append(decode(pipe, lat)); lats.append(lat.float().cpu())
        atomic_torch_save({"img": torch.cat(imgs), "lat": torch.cat(lats)}, f)


@torch.no_grad()
def net_diag_seq(tr, dev, mode):
    """δ_g on the design20 cache, one prompt (2 rows) at a time in step order 0..19 (state carries across steps)."""
    from boa_dit.w4a4.band_gain import band_masks
    from boa_dit.w4a4.screen import text_for
    c = torch.load(os.path.join(PCK, "design20_cache.pt"), map_location="cpu")
    e = WC.load_embeds("design")
    P = int(c["prompt"].max()) + 1
    idx = {(int(c["prompt"][j]), int(c["step"][j]), int(c["branch"][j])): j for j in range(len(c["t"]))}
    err = torch.zeros(P, NSTEP, 2, 4, 128, 128)
    STATE.mode = mode
    for p in range(P):
        for s in range(NSTEP):
            STATE.step = s
            rows = [idx[(p, s, 0)], idx[(p, s, 1)]]
            pe, pm = text_for(e, c["prompt"][rows], c["branch"][rows], dev)
            o = tr(c["latent"][rows].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][rows].to(dev),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0].float()[:, :4]
            err[p, s] = (o - c["eps"][rows].to(dev).float()).cpu()
    STATE.mode = "fp"
    eg = GW * err[:, :, 1].double() - (GW - 1) * err[:, :, 0].double()
    masks = band_masks()
    spec = []
    for s in SPEC_STEPS:
        Pw = torch.fft.fft2(eg[:, s]).abs().pow(2).sum((0, 1))
        spec.append([float(Pw[m].sum() / Pw.sum()) for m in masks])
    spec = np.mean(spec, axis=0).tolist()
    return {"per_step_err": eg.pow(2).flatten(2).sum(2).mean(0).tolist(), **ci_split(eg),
            "temporal_mean_err_energy": float(eg.mean(1).pow(2).flatten(1).sum(1).mean()),
            "band_spectrum_steps8_10": spec, "band1_share_steps8_10": spec[0]}


@torch.no_grad()
def arm_s3(kind, dither, feedback, dev, n=64, tag="", compress=None):
    name = f"s3_{kind}_{'feedback' if feedback else dither}" + (f"_{compress}" if compress else "") + (f"_{tag}" if tag else "")
    with Run(STAGE, f"arm_{name}", dev, {"kind": kind, "dither": dither, "feedback": feedback, "n": n, "batch": 1}) as run:
        ref = "bf16_b1" + (f"_{tag}" if tag else "")
        if not os.path.exists(os.path.join(img_dir(ref), f"{(n - 1) // CHUNK * CHUNK:05d}.pt")):
            gen_b1(ref, WC.load_pipe(dev), dev, n)                            # references at the same batch layout
        pipe = WC.load_pipe(dev)
        mods = install_fb(pipe.transformer, kind, dither, feedback, compress)
        keys_all = [f"{l}.{ln}" for l in range(28) for ln in QLAYERS]
        STATE.mode = "static"
        gen_b1(name, pipe, dev, n)
        st = sum(m.stats.cpu() for m in mods)
        for m in mods:
            m.stats.zero_()
        diag = net_diag_seq(pipe.transformer, dev, "static")
        del pipe
        torch.cuda.empty_cache()
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        ri, rl = load_gen(ref, n)
        qi, ql = load_gen(name, n)
        psnr, lp = [], []
        for i in range(n):
            a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
            psnr.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
            lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
        rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
        res = {"arm": name, "kind": kind, "dither": dither, "feedback": feedback, "n": n, "batch": 1, "prompt_set": f"mjhq1k[:{n}]",
               "psnr_mean": float(np.mean(psnr)), "lpips_mean": float(np.mean(lp)), "latent_rel_mse_mean": float(np.mean(rel)),
               "psnr_per_image": psnr, "lpips_per_image": lp, "latent_rel_mse_per_image": rel, "diag": diag,
               "feedback_clip_rate": float(st[0] / st[2]) if feedback else None,
               "feedback_residual_clamp_rate": float(st[1] / st[2]) if feedback else None,
               "feedback_clip_energy_by_type_bucket": ({ln: sum(m.clipE.cpu() for m, k in zip(mods, keys_all) if k.endswith(ln)).tolist()
                                                        for ln in QLAYERS} if feedback else None)}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "stage3", f"pixart_{name}.json"), res)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)
