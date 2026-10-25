"""results/stage2/fid_anchors.json from the FID-2k JSONs, paired with each model's latent MSE."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import atomic_write_json  # noqa: E402

E = os.path.join(ROOT, "results", "stage2", "eval")
rows = {}
for label, name in [("FP", "fp"), ("uniform W4", "uniform_w4_K1_lam1_brcfg"), ("uniform W3", "uniform_w3_K1_lam1_brcfg")]:
    f = json.load(open(os.path.join(E, f"fid2000_{name}.json")))
    lat = os.path.join(E, f"latent_{name}.json")
    lat = json.load(open(lat)) if os.path.exists(lat) else None
    rows[label] = {"name": name, "fid2k": f["fid"], "gpu_hours": f["gpu_hours"],
                   "rel_latent_mse_512": lat["rel_mse_mean"] if lat else 0.0}
atomic_write_json(os.path.join(ROOT, "results", "stage2", "fid_anchors.json"),
                  {"note": "FID-2k vs the ADM 256 reference batch (10k), seeds 0-1999, cfg 1.5, DDIM-50, pytorch-fid; "
                           "same seeds for all three. FID-2k is biased upward and noisy by ~1-2 points.",
                   "anchors": rows})
for k, v in rows.items():
    print(f"{k:11s} FID-2k {v['fid2k']:.2f}   rel latent MSE {v['rel_latent_mse_512']:.4f}")
