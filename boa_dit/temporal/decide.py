"""Queue helpers (rule.md).
  budget EST : exit 1 if logged stage-temporal GPU-hours + EST would exceed the 4.0 h cap
  b2         : print 'PASS <winner>' or 'FAIL <winner>' for the B2 read (best dither by latent MSE vs no dither:
               >= 10% latent reduction with CI > 0 and >= +0.3 dB PSNR)
Run: python -m boa_dit.temporal.decide budget 0.5 | b2
"""
import csv
import json
import os
import sys

from boa_dit.common import RESULTS
from boa_dit.temporal.common import OUT, STAGE, boot_reduction

CAP = float(os.environ.get("TEMPORAL_CAP", "4.0"))


def spent():
    with open(os.path.join(RESULTS, "gpu_hours.csv")) as f:
        return sum(float(r[5]) for r in csv.reader(f) if r and r[0] == STAGE)


def b2(weights="w16"):
    load = lambda d: json.load(open(os.path.join(OUT, "eval", f"pixart_{weights}_a8s_{d}.json")))
    base = load("none")
    best = None
    for d in ("golden", "random", "calibrated"):
        r = load(d)
        br = boot_reduction(r["latent_rel_mse_per_image"], base["latent_rel_mse_per_image"])
        dp = r["psnr_mean"] - base["psnr_mean"]
        if best is None or br["reduction"] > best[1]["reduction"]:
            best = (d, br, dp)
    d, br, dp = best
    ok = br["reduction"] >= 0.10 and br["ci"][0] > 0 and dp >= 0.3
    return ok, d


if __name__ == "__main__":
    if sys.argv[1] == "budget":
        s, est = spent(), float(sys.argv[2])
        print(f"spent {s:.3f} h + est {est:.2f} h vs cap {CAP}", file=sys.stderr)
        sys.exit(0 if s + est <= CAP else 1)
    ok, d = b2()
    print(("PASS " if ok else "FAIL ") + d)
