"""Residual-kernel support of the saved V3 kernels (CPU). For each (branch, window, block, head) kernel kappa:
floor = min(kappa) (primary) or median(kappa) (robustness check), floor share = floor * 961 / sum(kappa),
kappa_res = (kappa - floor) clipped at 0, and the side of the smallest centered square holding 95% of kappa_res's mass.
Verdict: "residual compact" iff median side <= 9 and >= 90% of kernels have side <= 15.
-> results/attn_share/kernel_residual_support.json
"""
import os

import numpy as np
import torch

from boa_dit.attn_share.kernel_support import support
from boa_dit.common import RESULTS, atomic_write_json

OUT = os.path.join(RESULTS, "attn_share")


def analyse(kap, how):
    flat = kap.flatten(-2)
    floor = flat.min(-1).values if how == "min" else flat.median(-1).values
    share = floor * flat.shape[-1] / flat.sum(-1)
    res = (kap - floor[..., None, None]).clamp_min(0)
    side = support(res)
    s = side.flatten().numpy()
    vals, counts = np.unique(s, return_counts=True)
    verdict = "residual compact" if (np.median(s) <= 9 and (s <= 15).mean() >= 0.9) else "residual not compact"
    return {"floor": how, "floor_share_median": float(share.median()), "floor_share_p10": float(torch.quantile(share.flatten(), 0.1)),
            "floor_share_p90": float(torch.quantile(share.flatten(), 0.9)),
            "side_median": float(np.median(s)), "frac_side_le_9": float((s <= 9).mean()),
            "frac_side_le_15": float((s <= 15).mean()), "frac_side_31": float((s == 31).mean()),
            "side_histogram": {int(v): int(c) for v, c in zip(vals, counts)},
            "side_median_per_window": [float(side[w].float().median()) for w in range(side.shape[0])],
            "verdict": verdict, "sides": side.tolist()}


def main():
    a = torch.load(os.path.join(OUT, "artifacts", "V3_kernels_k20.pt"), map_location="cpu")
    res = {"source": "results/attn_share/artifacts/V3_kernels_k20.pt", "k": a["k"], "windows": a["windows"],
           "rule": "residual compact iff median side <= 9 and >= 90% of (block, head, window) kernels have side <= 15; "
                   "side = smallest centered square holding 95% of (kappa - floor)+ mass; full kernel is 31x31",
           "per_branch": {br: {how: analyse(a["kappa"][br], how) for how in ("min", "median")} for br in ("cond", "uncond")}}
    v = {br: res["per_branch"][br]["min"]["verdict"] for br in res["per_branch"]}
    res["verdict"] = v["cond"] if len(set(v.values())) == 1 else f"cond: {v['cond']}; uncond: {v['uncond']}"
    atomic_write_json(os.path.join(OUT, "kernel_residual_support.json"), res)
    for br, d in res["per_branch"].items():
        for how, r in d.items():
            print(br, how, {k: v for k, v in r.items() if k not in ("sides", "side_histogram")})
    print("VERDICT:", res["verdict"])


if __name__ == "__main__":
    main()
