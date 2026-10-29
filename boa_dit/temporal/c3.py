"""Consolidated plan (2026-10-28): shared 3.0 A30-hour budget, stage `c3`, counted across both branches' gpu_hours.csv.
Run: python -m boa_dit.temporal.c3 [est_hours]   (exit 1 if spent + est would cross the cap)"""
import csv
import os
import sys

STAGE = "c3"
CAP = 3.0
CSVS = ["/home/boa-dit/results/gpu_hours.csv", "/home/boa-dit-wj/results/gpu_hours.csv"]


def spent():
    s = 0.0
    for f in CSVS:
        if os.path.exists(f):
            for r in csv.reader(open(f)):
                if r and r[0] == STAGE:
                    s += float(r[-1])
    return s


def guard(est, label=""):
    s = spent()
    ok = s + est <= CAP
    print(f"[c3 budget] {label}: spent {s:.3f} h + est {est:.2f} h vs cap {CAP} -> {'OK' if ok else 'STOP'}", flush=True)
    return ok


if __name__ == "__main__":
    sys.exit(0 if guard(float(sys.argv[1]) if len(sys.argv) > 1 else 0.0, "check") else 1)
