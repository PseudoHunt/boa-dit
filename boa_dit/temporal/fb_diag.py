"""Matched diagnostics, baseline vs feedback (reviewer request; stage3/implementation.md). DiT-XL/2, W16A8 single table,
eval seeds 0-31 (one batch of 32), both arms on the same seeds and batch layout. The quantizer is exactly
dit_act.Quantize; this subclass only adds bookkeeping:
  * per-step error energy Σ||e_t||² at all 112 insertion sites (total and per layer type), e_t = q_t - x_t;
  * cumulative local residual ||Σ_{s<=t} e_s|| over t on TRACK sites (4 layer types x blocks 2, 13, 26);
  * on IDENT sites (one seed): with e_t = q̂_t - x_t, c_t = ũ_t - u_t, η_t = s_t - r⁰_t (fp16 storage + guard clamp),
    κ_t = q̂_t - q_t (cast), and r⁰_t = ũ_t - q̂_t (the residual tracks the effective quantizer, so κ is absorbed):
      Σ_t e_t = -s_last + Σ_t c_t + Σ_t η_t      (s_last = r⁰_last + η_last)
    reported: each sum's norm (‖Σκ‖ too), the absolute discrepancy, and the discrepancy / sqrt(Σ_t ||e_t||²);
  * max |r_t| over the trajectory per site and the clamp binding rate;
  * endpoint latent rel. MSE vs FP; prediction-error C/I from dit.diagnostics (FP trajectory, seeds 0-31).
Run: python -m boa_dit.temporal.fb_diag
"""
import os

import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.eval_latent_mse import fp_reference
from boa_dit.model import LINEARS, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample
from boa_dit import temporal  # noqa: F401
from boa_dit.temporal import act as ACT
from boa_dit.temporal.common import OUT, STAGE
from boa_dit.temporal.dit import diagnostics
from boa_dit.temporal.dit_act import T, Quantize

SEEDS = list(range(32))
TRACK = [f"b{l}.{ln}" for l in (2, 13, 26) for ln in LINEARS]
IDENT = ["b13.mlp.fc2", "b13.attn.qkv", "b13.attn.proj"]


class Tracked(Quantize):
    """dit_act.Quantize with bookkeeping only; the feedback arithmetic is Quantize.fb_step, unchanged."""

    def __init__(self, *a, ident_rows=None, **k):
        super().__init__(*a, **k)
        self.e_step = torch.zeros(T, dtype=torch.float64)
        self.e_type = {ln: torch.zeros(T, dtype=torch.float64) for ln in LINEARS}
        self.cum = {key: torch.zeros(T, dtype=torch.float64) for key in TRACK}
        self.S, self.terms, self.rmax, self.active = {}, {}, {}, True

    def _fn(self, key, x, mod):
        t = CTX.step
        if not self.active:
            return super()._fn(key, x, mod)
        s_c = self.sm.get(key)
        xs = x.float() if s_c is None else x.float() / s_c                    # quantizer (smoothed) space
        D, Z, O, qm = self.p[key]
        lo, hi = -Z[t] * D[t], (qm - 1 - Z[t]) * D[t]
        if not self.feedback:
            q = ACT.q_static(xs, D[t], Z[t], O[t], qm)
            consumed = (q if s_c is None else q * s_c).to(torch.float16).float()   # baseline consumes fp16 too (autocast)
            qh = consumed if s_c is None else consumed / s_c
            self.clip[t] += torch.tensor([float(((xs < lo) | (xs > hi)).sum()), float(xs.numel())], dtype=torch.float64)
            self._book(key, t, qh - xs, None)
            return consumed.to(x.dtype)
        st = self.fb_step(key, t, xs, D, Z, qm, lo, s_c)
        self._log_fb(key, t, xs, st["u"], st["uc"], lo, hi, st["hi_rep"], st["r"], st["r0"])
        self._book(key, t, st["qh"] - xs, st)
        return st["consumed"].to(x.dtype)

    def _book(self, key, t, e, st):
        en = float(e.double().pow(2).sum())
        self.e_step[t] += en
        self.e_type[key.split(".", 1)[1]][t] += en
        if key in TRACK:
            self.S[key] = e.clone() if t == 0 else self.S[key] + e
            self.cum[key][t] = float(self.S[key].double().norm())
        if st is None:
            return
        self.rmax[key] = max(self.rmax.get(key, 0.0), float(st["s"].abs().max()))
        if key not in IDENT:
            return
        if t == 0:
            self.terms[key] = {n: torch.zeros_like(e) for n in ("sum_e", "c", "eta", "kappa", "dsc")}
            self.terms[key]["sum_e2"] = 0.0
        tm = self.terms[key]
        tm["sum_e"] += e                                                      # e_t = q̂_t - x_t (quantizer space)
        tm["sum_e2"] += float(e.double().pow(2).sum())
        tm["c"] += st["uc"] - st["xs"] - st["s_prev"] + st["d"]               # clip change: (x_c - x) guarded, (ũ - u) unguarded
        tm["eta"] += st["s"] - st["r0"]                                       # stored vs exact residual (fp16 + guard clamp)
        tm["kappa"] += st["qh"] - st["q"]                                     # cast error (absorbed: r⁰ = ũ - q̂)
        tm["dsc"] += st["d"]                                                  # guard: discarded residual
        if t == T - 1:
            d = {n: v.double() for n, v in tm.items() if n != "sum_e2"}
            rhs = -st["s"].double() + d["c"] + d["eta"] - d["dsc"]            # Σ(q̂ - x) = -s_last + Σc + Ση - Σd
            disc = float((d["sum_e"] - rhs).norm())
            norm = float(tm["sum_e2"]) ** 0.5
            self.terms[key] = {"norm_sum_e": float(d["sum_e"].norm()), "sqrt_sum_e_energy": norm,
                               "norm_s_last": float(st["s"].norm()), "norm_r0_last": float(st["r0"].norm()),
                               "norm_sum_c": float(d["c"].norm()), "norm_sum_eta": float(d["eta"].norm()),
                               "norm_sum_kappa": float(d["kappa"].norm()), "norm_sum_discarded": float(d["dsc"].norm()),
                               "abs_discrepancy": disc,
                               "discrepancy_over_sqrt_sum_e_energy": disc / norm,
                               "plain_telescoping_ratio": float(d["sum_e"].norm() / norm)}


@torch.no_grad()
def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    xf, _ = fp_reference(dev)
    xf = xf[:32]
    res = {"seeds": "0-31", "track_sites": TRACK, "ident_sites": IDENT, "arms": {}}
    with Run(STAGE, "fb_matched_diag", dev, {"seeds": "0-31"}) as run:
        m, d = load_dit(dev), make_diffusion(50)
        res["identity_one_seed"] = {}
        for guard in (True, False):                                           # identity check: one seed (seed 0)
            q = Tracked(m, "single8", "none", dev, feedback=True, guard=guard)
            ddim_sample(m, d, noise_for_seeds([0], dev), labels_for_seeds([0]), 1.5)
            res["identity_one_seed"]["guarded" if guard else "unguarded"] = {"seed": 0, "layers": q.terms, "clamp_rate": q.clamp_rate()}
            q.remove()
        for fb, guard in ((False, True), (True, True), (True, False)):
            q = Tracked(m, "single8", "none", dev, feedback=fb, guard=guard)
            x = ddim_sample(m, d, noise_for_seeds(SEEDS, dev), labels_for_seeds(SEEDS), 1.5).cpu()
            rel = ((x - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1))
            q.active = False
            dg = diagnostics(m, d, dev)
            res["arms"]["baseline" if not fb else ("feedback" if guard else "feedback_unguarded_DIAGNOSTIC")] = {
                "endpoint_rel_mse": float(rel.mean()), "endpoint_rel_mse_per_image": rel.tolist(),
                "site_err_energy_per_step": q.e_step.tolist(),
                "site_err_energy_per_step_by_type": {k: v.tolist() for k, v in q.e_type.items()},
                "cum_residual_norm": {k: v.tolist() for k, v in q.cum.items()},
                "clip_rate_per_step": q.clip_rate(), "clamp_rate": q.clamp_rate() if fb else None,
                "clip_energy_by_site_bucket": q.clip_energy() if fb else None,
                "guard_discarded_by_site_bucket": q.discarded() if fb else None,
                "identity": q.terms if fb else None, "max_abs_r": q.rmax if fb else None,
                "pred_C_over_I": dg["C_over_I"], "pred_C_share": dg["C_share"], "pred_per_step_err": dg["per_step_err"],
                "pred_band1_share_steps20_25": dg["band1_share_steps20_25"]}
            q.remove()
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "stage3", "fb_matched_diag.json"), res)
    for k, v in res["arms"].items():
        print(k, "endpoint", round(v["endpoint_rel_mse"], 5), "pred C/I", round(v["pred_C_over_I"], 4), flush=True)
    print("identity (seed 0)", res["identity_one_seed"], flush=True)


if __name__ == "__main__":
    main()
