"""Print gain2 per step across rho values: ratio-of-means vs median-of-ratios, plus floor comparison."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tags = sys.argv[1:] or ["gain_iso_rho0.0001", "gain_iso_rho0.003", "gain_iso_rho0.01", "gain_iso_rho0.03"]
D = {}
for t in tags:
    p = os.path.join(ROOT, "results", "stage1", f"{t}.json")
    if os.path.exists(p):
        D[t] = json.load(open(p))
print("cols: ratio-of-means / median-of-ratios")
print("step | " + " | ".join(f"{t[9:]:>15s}" for t in D))
for i in range(50):
    row = []
    for t, d in D.items():
        r = np.array(d["gain2_per_seed"][i])
        row.append(f"{d['gain2'][i]:7.3g} {np.median(r):7.3g}")
    print(f"{i:4d} | " + " | ".join(row))
for t, d in D.items():
    g = np.array(d["gain2"]); m = np.array([np.median(x) for x in d["gain2_per_seed"]])
    print(f"{t}: mean-ratio max/med {g.max() / np.median(g):.2f}; median-ratio max/med {m.max() / np.median(m):.2f}")
