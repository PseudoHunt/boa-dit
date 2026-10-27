"""Pre-stage-3 checks (review): (1) final-cast error and (2) histogram-scorer consistency, on saved activations from
design/calibration data, single-table A8 tables (as calibrated).
  (1) per layer type (blocks 2, 13, 26), per step: rounding error ||q - x||² vs cast error ||q̂ - q||², q̂ = the value
      the next linear consumes (DiT: fp16, the autocast dtype of every block linear; PixArt: bf16); ratio per step and
      accumulated over steps (sum of energies, and ||Σ_t κ_t|| vs ||Σ_t (q - x)||).
  (2) per layer type, a fixed 4,096-token subset at steps {first, middle, last}: the exact inference-path error
      (q_static with the table's nudged Δ, Z, fp32; and with the cast) vs the histogram scorer's estimate for the same
      tokens and the same (Δ, Z); worst relative mismatch.
DiT: calibration seeds 1000-1001 (one batch). PixArt: design prompt 0 (batch 1).
Run: python -m boa_dit.temporal.precheck --bed dit|pixart
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.temporal import act as ACT
from boa_dit.temporal.common import OUT, STAGE

BLOCKS = (2, 13, 26)


def scorer_estimate(xs, D, Z, qmax):
    """The calibration scorer on these tokens: histogram over their own [min, max], error at bin centres, same (Δ, Z)."""
    cal = ACT.Calib(xs.device)
    mn, mx = float(xs.min()), float(xs.max())
    h = torch.histc(xs.float(), bins=ACT.NBINS, min=mn, max=mx).double().view(1, -1)
    w = (mx - mn) / ACT.NBINS
    c = (mn + w * (torch.arange(ACT.NBINS, device=xs.device, dtype=torch.float64) + 0.5)).view(1, -1)
    lo = torch.tensor([-Z * D], device=xs.device, dtype=torch.float64)
    hi = lo + (qmax - 1) * D
    return float(cal._err_rows(c, h.to(xs.device), lo, hi, qmax)[0])


class Probe:
    def __init__(self, tabs, T, cast, ltype):
        self.tabs, self.T, self.cast, self.ltype = tabs, T, cast, ltype
        self.acc, self.score, self.tok = {}, {}, {}

    def __call__(self, key, t, x):
        ln = self.ltype(key)
        tb = self.tabs[key]
        D, Z, qm = float(tb["D"][t]), float(tb["Z"][t]), tb["qmax"]
        Dt, Zt = torch.tensor(D, device=x.device), torch.tensor(Z, device=x.device)
        q = ACT.q_static(x, Dt, Zt, torch.zeros((), device=x.device), qm)
        qh = q.to(self.cast).float()
        e, k = q - x, qh - q
        a = self.acc.setdefault(ln, {"round": np.zeros(self.T), "cast": np.zeros(self.T), "Se": None, "Sk": None, "keys": set()})
        a["keys"].add(key)
        a["round"][t] += float(e.double().pow(2).sum()); a["cast"][t] += float(k.double().pow(2).sum())
        if key.startswith(f"{BLOCKS[1]}.") or key.startswith(f"b{BLOCKS[1]}."):        # accumulation on the middle block
            a["Se"] = e.double().clone() if t == 0 else a["Se"] + e.double()
            a["Sk"] = k.double().clone() if t == 0 else a["Sk"] + k.double()
        if t in (0, self.T // 2, self.T - 1):
            flat = x.reshape(-1, x.shape[-1])
            if key not in self.tok:
                g = torch.Generator().manual_seed(len(self.tok))
                self.tok[key] = torch.randperm(flat.shape[0], generator=g)[:4096].to(x.device)
            xs = flat[self.tok[key]]
            qx = ACT.q_static(xs, Dt, Zt, torch.zeros((), device=x.device), qm)
            exact = float((qx - xs).double().pow(2).sum())
            exact_cast = float((qx.to(self.cast).float() - xs).double().pow(2).sum())
            est = scorer_estimate(xs, D, Z, qm)
            self.score.setdefault(ln, []).append({"key": key, "t": t, "exact": exact, "exact_with_cast": exact_cast, "scorer": est,
                                                  "rel_mismatch": abs(est - exact) / exact, "rel_mismatch_vs_cast": abs(est - exact_cast) / exact_cast})

    def result(self):
        out = {"cast_dtype": str(self.cast), "by_type": {}, "scorer": {}}
        for ln, a in self.acc.items():
            ratio = a["cast"] / np.maximum(a["round"], 1e-30)
            r = {"ratio_per_step": ratio.tolist(), "ratio_total": float(a["cast"].sum() / a["round"].sum()), "max_step_ratio": float(ratio.max()),
                 "sites": sorted(a["keys"])}
            if a["Se"] is not None:
                r["accum_norm_sum_cast_over_norm_sum_round"] = float(a["Sk"].norm() / a["Se"].norm())
                r["accum_norm_sum_cast_over_sqrt_sum_round_energy"] = float(a["Sk"].norm() / np.sqrt(a["round"].sum() / len(a["keys"])))
            out["by_type"][ln] = r
        for ln, v in self.score.items():
            w = max(v, key=lambda z: z["rel_mismatch"])
            out["scorer"][ln] = {"max_rel_mismatch": w["rel_mismatch"], "worst": w, "mean_rel_mismatch": float(np.mean([z["rel_mismatch"] for z in v])),
                                 "max_rel_mismatch_vs_cast_path": max(z["rel_mismatch_vs_cast"] for z in v)}
        out["scorer_worst_layer"] = max(out["scorer"].items(), key=lambda kv: kv[1]["max_rel_mismatch"])[0]
        out["max_cast_ratio_any_type"] = max(r["ratio_total"] for r in out["by_type"].values())
        out["tracking_required"] = bool(out["max_cast_ratio_any_type"] > 0.01)
        return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bed", required=True, choices=["dit", "pixart"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    with Run(STAGE, f"precheck_{a.bed}", dev, vars(a)) as run:
        if a.bed == "dit":
            from boa_dit.model import load_dit, make_diffusion, labels_for_seeds, noise_for_seeds
            from boa_dit.sample import CTX, ddim_sample
            from boa_dit.temporal.dit_act import T, bpath, input_hooks, load_smooth
            tabs = torch.load(bpath("single8"), map_location="cpu")
            sm = load_smooth(dev)
            pr = Probe(tabs, T, torch.float16, lambda k: k.split(".", 1)[1])
            m, d = load_dit(dev), make_diffusion(50)
            sel = {k for k in tabs if int(k.split(".")[0][1:]) in BLOCKS}
            hs = input_hooks(m, lambda key, x, mod: pr(key, CTX.step, x.float() / sm[key] if key in sm else x.float()) if key in sel else None)
            ddim_sample(m, d, noise_for_seeds([1000, 1001], dev), labels_for_seeds([1000, 1001]), 1.5)
        else:
            from boa_dit.temporal import pixart as PX
            from boa_dit.w4a4 import common as WC
            tabs = torch.load(PX.bpath("single8"), map_location="cpu")
            pr = Probe(tabs, PX.NSTEP, torch.bfloat16, lambda k: k.split(".", 1)[1])
            pipe = WC.load_pipe(dev)
            sel = {k for k in tabs if int(k.split(".")[0]) in BLOCKS}
            hs = PX.layer_inputs(pipe.transformer, lambda key, x, mod: pr(key, PX.STATE.step, x) if key in sel else None)
            sc = PX.StepCounter(pipe.transformer)
            PX.sample_lat(pipe, WC.load_embeds("design"), [0], dev)
            sc.remove()
        for h in hs:
            h.remove()
        res = {"bed": a.bed, **pr.result(), **run.record()}
    atomic_write_json(os.path.join(OUT, "stage3", f"precheck_{a.bed}.json"), res)
    print({k: v for k, v in res.items() if k in ("cast_dtype", "max_cast_ratio_any_type", "tracking_required", "scorer_worst_layer")})
    for ln, r in res["by_type"].items():
        print(ln, "cast/round energy", round(r["ratio_total"], 5), "max step", round(r["max_step_ratio"], 5),
              "accum", {k: round(v, 5) for k, v in r.items() if k.startswith("accum")})
    for ln, r in res["scorer"].items():
        print(ln, "scorer max rel mismatch", round(r["max_rel_mismatch"], 5), "(vs cast path", round(r["max_rel_mismatch_vs_cast_path"], 5), ")")


if __name__ == "__main__":
    main()
