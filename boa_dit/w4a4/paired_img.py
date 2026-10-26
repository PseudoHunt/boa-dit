"""Paired MJHQ-n image comparison of an arm vs a reference arm (rule_addendum_screen.md): mean per-image PSNR and LPIPS
differences with paired bootstrap 95% CIs (2,000 resamples, seed 0). Usage: python -m boa_dit.w4a4.paired_img ARM REF [n]"""
import json
import os
import sys

import numpy as np

from boa_dit.common import RESULTS


def load(arm, n):
    for p in (f"img{n}_{arm}.json", f"img256_{arm}.json"):
        f = os.path.join(RESULTS, "w4a4", "eval", p)
        if os.path.exists(f):
            d = json.load(open(f))
            return np.array(d["psnr_per_image"][:n]), np.array(d["lpips_per_image"][:n])
    raise FileNotFoundError(arm)


def paired(arm, ref, n=64):
    pa, la = load(arm, n)
    pr, lr = load(ref, n)
    rng = np.random.default_rng(0)
    idx = [rng.integers(0, n, n) for _ in range(2000)]
    dp, dl = pa - pr, la - lr
    return {"arm": arm, "ref": ref, "n": n, "dpsnr": float(dp.mean()), "dpsnr_ci": np.percentile([dp[i].mean() for i in idx], [2.5, 97.5]).tolist(),
            "dlpips": float(dl.mean()), "dlpips_ci": np.percentile([dl[i].mean() for i in idx], [2.5, 97.5]).tolist()}


if __name__ == "__main__":
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    print(json.dumps(paired(sys.argv[1], sys.argv[2], n)))
