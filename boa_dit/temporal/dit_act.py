"""B′ on DiT-XL/2 (rule.md): static activation quantization of the 112 block-linear inputs (weights fp32 under the eval
protocol's fp16 autocast), step index = sample.CTX.step (50 DDIM steps).
  calib : step8 / single8 / single6 tables on calibration seeds 1000-1031 (FP trajectory, both CFG halves, all tokens)
  b1    : B1′ layer-level variants (same trajectory, 2 seeds per run) + network C/I on the FP trajectory of 1000-1031
  arm   : latent rel. MSE on eval seeds 0-511 (--kind, --dither) + C/I diagnostics on the eval diagnostic seeds 0-31
Run: python -m boa_dit.temporal.dit_act calib | b1 | arm --kind single8 --dither sym2
"""
import argparse
import math
import os

import torch

from boa_dit.common import CKPT_DIR, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import LINEARS, block_linear, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.objective.proxy import step_outputs
from boa_dit.sample import CTX, ddim_sample, sample_seeds
from boa_dit.temporal import act as ACT
from boa_dit.temporal.common import CK, OUT, STAGE, ci_split
from boa_dit.temporal.dit import diagnostics

T = 50
CAL_SEEDS = list(range(1000, 1032))
CAL_TRAJ = os.path.join(CK, "fp_traj_calib1000_all.pt")


SMOOTH_TYPES = ("mlp.fc1", "mlp.fc2")
SMOOTH_PATH = os.path.join(CK, "dit_smooth_s.pt")


def load_smooth(dev=None):
    """{key: s [c_in]} for the smoothed sites (x̃ = x/s, W̃ = W·diag(s)); empty if not computed."""
    if not os.path.exists(SMOOTH_PATH):
        return {}
    sm = torch.load(SMOOTH_PATH, map_location="cpu")
    return {k: (v.to(dev) if dev else v) for k, v in sm.items()}


@torch.no_grad()
def compute_smooth(m, diffusion, dev):
    """s_c = sqrt(max|X_c|) / sqrt(max_i |W_ic|) on the calibration trajectory (all steps, both halves, all tokens)."""
    amax = {}

    def fn(key, x, mod):
        if x is None or key.split(".", 1)[1] not in SMOOTH_TYPES:
            return
        a = x.abs().amax((0, 1))
        amax[key] = a if key not in amax else torch.maximum(amax[key], a)
    run_cal(m, diffusion, dev, fn)
    sm = {}
    for key, a in amax.items():
        l, ln = key.split(".", 1)
        w = block_linear(m.blocks[int(l[1:])], ln).weight.float().abs().amax(0)
        sm[key] = (a.clamp_min(1e-5).sqrt() / w.clamp_min(1e-5).sqrt()).cpu()
    atomic_torch_save(sm, SMOOTH_PATH)
    return sm


def bpath(kind):
    return os.path.join(CK, f"dit_static_b_{kind}.pt")


def keys_of():
    return [f"b{l}.{ln}" for l in range(28) for ln in LINEARS]


def input_hooks(model, fn):
    """Pre-hooks on every block linear: x = fn(key, x, mod) (return None to leave the input unchanged)."""
    hs = []
    for key in keys_of():
        l, ln = key.split(".", 1)
        lin = block_linear(model.blocks[int(l[1:])], ln)

        def pre(mod, inp, key=key):
            y = fn(key, inp[0], mod)
            return None if y is None else (y,)
        hs.append(lin.register_forward_pre_hook(pre))
    return hs


class Quantize:
    """Static quantizer of every block-linear input from a table set, with offsets o_t = frac_t · Δ_t, or (feedback=True)
    full temporal error feedback: per site and per row, r (fp16, the activation's shape), reset at step 0;
    u = x + s, ũ = clip(u) to the representable range [-ZΔ, (qmax - Z)Δ], q = Q(ũ) on codes 0..qmax, q̂ = fp16(q) (the
    consumed value), r⁰ = ũ - q̂, s = fp16(clamp(r⁰, ±Δ/2)). Clip error is never fed back. Logs clip counts/energies for
    nominal-interval and representable-range events per site and 5-step bucket, and the clamp binding rate."""

    IDENT_KEY = "b13.mlp.fc1"

    def __init__(self, model, kind, dither, dev, mode="static", feedback=False, guard=True):
        tabs = torch.load(bpath(kind), map_location="cpu")
        self.sm, self.guard = load_smooth(dev), guard
        self.disc = {}                                                        # guard: discarded residual energy per site
        keys = sorted(tabs)
        self.p = {k: (tb["D"].float().to(dev), tb["Z"].float().to(dev),
                      (ACT.offset_frac(dither, T, keys.index(k)) * tb["D"]).float().to(dev), tb["qmax"]) for k, tb in tabs.items()}
        self.mode, self.feedback = mode, feedback
        self.r, self.clip, self.clamp = {}, torch.zeros(T, 2, dtype=torch.float64), torch.zeros(T, 2, dtype=torch.float64)
        self.clipE = {}
        self.ident, self.batches = None, 0
        self.hs = input_hooks(model, self._fn)

    def _fn(self, key, x, mod):
        t = CTX.step
        s_c = self.sm.get(key)                                                # smoothing: quantize x/s, feed q·s (= W·s on q)
        xs = x.float() if s_c is None else x.float() / s_c
        unsm = (lambda v: v) if s_c is None else (lambda v: v * s_c)
        if self.mode == "dynamic":
            return unsm(ACT.q_dynamic(xs)).to(x.dtype)
        D, Z, O, qm = self.p[key]
        lo, hi = -Z[t] * D[t], (qm - 1 - Z[t]) * D[t]
        if not self.feedback:
            self.clip[t] += torch.tensor([float(((xs < lo) | (xs > hi)).sum()), float(xs.numel())], dtype=torch.float64)
            return unsm(ACT.q_static(xs, D[t], Z[t], O[t], qm)).to(x.dtype)
        st = self.fb_step(key, t, xs, D, Z, qm, lo, s_c)
        self._log_fb(key, t, xs, st["u"], st["uc"], lo, hi, st["hi_rep"], st["r"], st["r0"])
        return st["consumed"].to(x.dtype)

    def fb_step(self, key, t, xs, D, Z, qm, lo, s_c=None):
        """One feedback quantization step at site `key` (the only place the feedback arithmetic lives). xs is in the
        smoothed space (x/s) when the site is smoothed; the state lives in that space too."""
        if t == 0 or key not in self.r or self.r[key].shape != xs.shape:
            self.r[key] = torch.zeros_like(xs, dtype=torch.float16)            # state reset at step 0 (per row: per sample, per branch)
        s_prev = self.r[key].float()
        hi_rep = (qm - Z[t]) * D[t]                                           # representable top: code qmax (spare code usable)
        if self.guard:                                                        # feedback-induced clipping guard
            xcl = torch.minimum(torch.maximum(xs, lo), hi_rep)                # x clipped exactly as the baseline would
            r_fed = torch.minimum(torch.maximum(s_prev, lo - xcl), hi_rep - xcl)   # feed only what keeps u in range
            d = s_prev - r_fed                                                # discarded part (energy logged)
            de = self.disc.setdefault(key, torch.zeros(T // 5, 2, dtype=torch.float64))
            de[t // 5] += torch.tensor([float((d != 0).sum()), float(d.double().pow(2).sum())], dtype=torch.float64)
            u = xcl + r_fed                                                   # u_t inside the representable range
        else:
            d, u = torch.zeros_like(xs), xs + s_prev                          # unguarded (diagnostic only): u_t = x_t + s_{t-1}
        uc = torch.minimum(torch.maximum(u, lo), hi_rep)                      # ũ_t (guarded: no-op beyond the baseline clip)
        q = ACT.q_static(uc, D[t], Z[t], torch.zeros((), device=xs.device), qm, hi_code=qm)   # q_t on codes 0..qmax
        consumed = (q if s_c is None else q * s_c).to(torch.float16).float()  # what the next linear consumes (fp16 autocast)
        qh = consumed if s_c is None else consumed / s_c                      # q̂_t in the quantizer's (smoothed) space
        r0 = uc - qh                                                          # r⁰_t: rounding residual of the effective quantizer
        r = r0.clamp(-D[t] / 2, D[t] / 2)                                     # guard clamp (checked: binding counted)
        s = r.to(torch.float16)                                               # s_t: stored state, fp16, no damping
        self.r[key] = s
        return {"s_prev": s_prev, "u": u, "uc": uc, "q": q, "qh": qh, "r0": r0, "r": r, "s": s.float(), "hi_rep": hi_rep,
                "d": d, "consumed": consumed, "xs": xs}

    def _log_fb(self, key, t, xs, u, uc, lo, hi, hi_rep, r, r0):
        nom = torch.minimum(torch.maximum(u, lo), hi)                          # nominal interval (254 intervals)
        b = t // 5
        cl = self.clipE.setdefault(key, torch.zeros(T // 5, 4, dtype=torch.float64))
        cl[b] += torch.tensor([float((u != nom).sum()), float((nom - u).double().pow(2).sum()),
                               float((u != uc).sum()), float((uc - u).double().pow(2).sum())], dtype=torch.float64)
        self.clip[t] += torch.tensor([float((u != uc).sum()), float(u.numel())], dtype=torch.float64)
        self.clamp[t] += torch.tensor([float((r != r0).sum()), float(r.numel())], dtype=torch.float64)

    def clip_energy(self):
        """per site and 5-step bucket: [nominal-interval events, their energy, representable-range events, their energy]."""
        return {k: v.tolist() for k, v in self.clipE.items()}

    def discarded(self):
        """guard: per site and 5-step bucket [elements with a discarded part, discarded energy Σ||d||²]."""
        return {k: v.tolist() for k, v in self.disc.items()}

    def clamp_rate(self):
        return float(self.clamp[:, 0].sum() / self.clamp[:, 1].sum().clamp_min(1))

    def clip_rate(self):
        return (self.clip[:, 0] / self.clip[:, 1].clamp_min(1)).tolist()

    def remove(self):
        for h in self.hs:
            h.remove()


def run_cal(model, diffusion, dev, fn, bs=32):
    """FP trajectory of CAL_SEEDS with input hooks fn(key, x fp32 [rows, 256, c], mod); fn('__end_batch__', ...) after each."""
    hs = input_hooks(model, lambda key, x, mod: fn(key, x.float(), mod))
    try:
        for i in range(0, len(CAL_SEEDS), bs):
            s = CAL_SEEDS[i:i + bs]
            ddim_sample(model, diffusion, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5)
            fn("__end_batch__", None, None)
    finally:
        for h in hs:
            h.remove()


@torch.no_grad()
def calib(dev):
    m, diffusion = load_dit(dev), make_diffusion(50)
    cal = ACT.Calib(dev)
    with Run(STAGE, "bp_calib_dit_smoothed", dev, {"alphas": ACT.ALPHAS, "seeds": "1000-1031", "smooth": SMOOTH_TYPES,
                                                   "one_sided": GELU_TYPES}) as run:
        sm = {k: v.to(dev) for k, v in compute_smooth(m, diffusion, dev).items()}
        sx = lambda key, x: x / sm[key] if key in sm else x
        run_cal(m, diffusion, dev, lambda key, x, mod: None if x is None else cal.add_minmax(key, CTX.step, sx(key, x)))
        run_cal(m, diffusion, dev, lambda key, x, mod: None if x is None else cal.add_hist(key, CTX.step, sx(key, x)))
        atomic_torch_save({"mn": cal.mn, "mx": cal.mx, "hist": cal.hist}, os.path.join(CK, "dit_calib_hist.pt"))
        rec = run.record()
    retable(dev, lo_min=True, cal=cal, rec=rec)


GELU_TYPES = ("mlp.fc2",)


def retable(dev, lo_min, cal=None, rec=None):
    """Tables from the saved histograms; lo_min=True: post-GELU inputs one-sided (lo = min, only hi searched)."""
    if cal is None:
        st = torch.load(os.path.join(CK, "dit_calib_hist.pt"), map_location="cpu")
        cal = ACT.Calib(dev)
        cal.mn, cal.mx, cal.hist = st["mn"], st["mx"], st["hist"]
    tabs = cal.tables(T, lo_min_keys=GELU_TYPES if lo_min else ())
    for kind, tb in tabs.items():
        atomic_torch_save(tb, bpath(kind))
    summ = {kind: {"alpha_hi_mean": float(torch.cat([v["alpha"] for v in tb.values()]).mean()),
                   "alpha_lo_mean": float(torch.cat([v["alpha_lo"] for v in tb.values()]).mean())} for kind, tb in tabs.items()}
    out = {"lo_min_post_gelu": lo_min, "summary": summ, "clip_rate_by_type": ACT.clip_summary(tabs), **(rec or {})}
    atomic_write_json(os.path.join(OUT, "bp_calib_dit.json"), out)
    print(out["clip_rate_by_type"], flush=True)


@torch.no_grad()
def cal_traj(m, diffusion, dev):
    if not os.path.exists(CAL_TRAJ):
        labels = labels_for_seeds(CAL_SEEDS)
        _, states = ddim_sample(m, diffusion, noise_for_seeds(CAL_SEEDS, dev), labels, 1.5, record_states=True)
        ref = {i: step_outputs(m, diffusion, states[i], labels, i)[0][:, :4].cpu() for i in range(T)}
        atomic_torch_save({"x": {i: s.cpu() for i, s in states.items()}, "ref": ref}, CAL_TRAJ)
    return torch.load(CAL_TRAJ, map_location="cpu")


@torch.no_grad()
def net_ci(m, diffusion, dev, tr):
    labels = labels_for_seeds(CAL_SEEDS)
    E = torch.stack([step_outputs(m, diffusion, tr["x"][i].to(dev), labels, i)[0][:, :4].cpu() - tr["ref"][i] for i in range(T)], 1)
    from boa_dit.objective.d2 import band_masks, band_spectrum
    spec = [band_spectrum(E[:, i], band_masks()) for i in range(20, 26)]
    import numpy as np
    spec = np.mean(spec, axis=0).tolist()
    return {"per_step_err": E.double().pow(2).flatten(2).sum(2).mean(0).tolist(), **ci_split(E),
            "band_spectrum_steps20_25": spec, "band1_share_steps20_25": spec[0]}


@torch.no_grad()
def b1(dev):
    tabs = {k: torch.load(bpath(k), map_location="cpu") for k in ACT.KINDS}
    m, diffusion = load_dit(dev), make_diffusion(50)
    coh = ACT.Coherence(tabs, T, dev)
    sm = load_smooth(dev)

    def fn(key, x, mod):
        if key == "__end_batch__":
            coh.end_batch()
            return None
        if key in sm:                                                         # smoothed space: x/s with W·s
            coh.step(key, CTX.step, x / sm[key], mod.weight.float() * sm[key])
        else:
            coh.step(key, CTX.step, x, mod.weight)
        return None
    with Run(STAGE, "bp_b1_dit", dev, {"seeds": "1000-1031"}) as run:
        run_cal(m, diffusion, dev, fn, bs=2)
        res = {"bed": "dit", "seed_set": "calibration 1000-1031", "layer": coh.result(), "net": {}}
        atomic_write_json(os.path.join(OUT, "bp_b1_dit_layers_partial.json"), res)
        tr = cal_traj(m, diffusion, dev)
        for name, kind, mode in (("single8", "single8", "static"), ("step8", "step8", "static"), ("single6", "single6", "static"),
                                 ("dynamic8", "single8", "dynamic")):
            q = Quantize(m, kind, "none", dev, mode)
            res["net"][name] = net_ci(m, diffusion, dev, tr)
            q.remove()
        res["gate"] = ACT.gate(res)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "bp_b1_dit.json"), res)
    print({k: res["net"][k]["C_over_I"] for k in res["net"]}, res["gate"], flush=True)


@torch.no_grad()
def arm(kind, dither, dev, n=512, feedback=False):
    name = f"bp_{kind}_{'feedback' if feedback else dither}"
    xf, _ = fp_reference(dev)
    xf = xf[:n]
    with Run(STAGE, f"arm_dit_{name}", dev, {"kind": kind, "dither": dither, "n": n}) as run:
        m, diffusion = load_dit(dev), make_diffusion(50)
        q = Quantize(m, kind, dither, dev, feedback=feedback)
        part = os.path.join(CK, f"dit_{name}.partial")
        xq = sample_seeds(m, diffusion, list(range(n)), 1.5, 32, partial=part, resume=True)
        rel, _, _ = per_image(xq, xf)
        res = {"arm": name, "kind": kind, "dither": dither, "n": n, "rel_mse_mean": float(rel.mean()),
               "rel_mse_sem": float(rel.std() / math.sqrt(n)), "rel_mse_per_image": rel.tolist(),
               "clip_rate_per_step": q.clip_rate(), "feedback_residual_clamp_rate": q.clamp_rate() if feedback else None,
               "feedback_clip_energy_by_site_bucket": q.clip_energy() if feedback else None,
               "feedback_guard_discarded_by_site_bucket": q.discarded() if feedback else None}
        q.clip.zero_()
        res["diag"] = diagnostics(m, diffusion, dev)
        res["diag"]["clip_rate_per_step"] = q.clip_rate()
        q.remove()
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "eval", f"dit_{name}.json"), res)
    os.remove(part)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "b1", "arm", "retable_lomin"])
    ap.add_argument("--kind", default="single8", choices=list(ACT.KINDS))
    ap.add_argument("--dither", default="none", choices=list(ACT.DITHERS))
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    {"calib": lambda: calib(a.device), "b1": lambda: b1(a.device), "retable_lomin": lambda: retable(a.device, True),
     "arm": lambda: arm(a.kind, a.dither, a.device, feedback=a.feedback)}[a.cmd]()


if __name__ == "__main__":
    main()
