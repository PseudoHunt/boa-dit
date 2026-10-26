"""Gate 1 queue helper (pre-registered choices, rule.md). 'e6' -> quantize_ts flags for E6 (best of E4/E5 by latent MSE,
+ act_order); 'best' -> name of the best engine arm (E1-E6) by latent MSE (for FID-5k). Exits 1 if a needed JSON is missing."""
import json
import os
import sys

G = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gate1", "eval")
BASE = "uniform"   # read-1 fallback (read 1 failed before Section 4): E2-E6 on the uniform-W4 base
E = {"E1": ("ts_dc8_ao_w4", "--act_order"),
     "E2": (f"ts_{BASE}_vv1_w4", f"--base {BASE} --v v1"),
     "E3": (f"ts_{BASE}_fc1v1_w4", f"--base {BASE} --fc1 v1"),
     "E4": (f"ts_{BASE}_vv1_fc1v1_w4", f"--base {BASE} --v v1 --fc1 v1"),
     "E5": (f"ts_{BASE}_vv2_fc1v1_wov2_w4", f"--base {BASE} --v v2 --fc1 v1 --wo v2")}


def mse(name):
    p = os.path.join(G, f"latent_{name}.json")
    if not os.path.exists(p):
        sys.exit(1)
    return json.load(open(p))["rel_mse_mean"]


if sys.argv[1] == "e6":
    k = min(("E4", "E5"), key=lambda k: mse(E[k][0]))
    print(E[k][1] + " --act_order")
else:
    names = [v[0] for v in E.values()]
    e6 = min((E["E4"][0], E["E5"][0]), key=mse).replace("_w4", "_ao_w4")
    print(min(names + [e6], key=mse))
