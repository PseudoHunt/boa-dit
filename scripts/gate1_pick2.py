"""Gate 1 follow-up: best heig-8 combination by latent MSE among H4, H2, fc1 row groups and the downstream-first arm (for FID-5k). Exit 1 if any is missing."""
import json
import os
import sys

G = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "gate1", "eval")
ARMS = ["ts_heig8_vv1_fc1v1_w4", "ts_heig8_vv1_w4", "ts_heig8_fc1rg_w4", "ts_heig8_vv1_fc1v1_dsf_w4"]  # H5gs dropped in the follow-up
m = {}
for a in ARMS:
    p = os.path.join(G, f"latent_{a}.json")
    if not os.path.exists(p):
        sys.exit(1)
    m[a] = json.load(open(p))["rel_mse_mean"]
print(min(m, key=m.get))
