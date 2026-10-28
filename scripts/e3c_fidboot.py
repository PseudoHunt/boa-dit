"""Paired FID bootstrap for the E3C FID-2k arms (results/e3c/PROTOCOL.md): 1,000 resamples of the 2,000 generated images (same indices
for every arm) against the fixed ADM reference statistics. CPU. Run: python scripts/e3c_fidboot.py"""
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import atomic_write_json
from boa_dit.fidboot import FIDRef, paired_boot
CK = "/home/boa-dit/ckpt/e3c/fid"
arms = [a for a in ["C0", "H16", "C1t_s0", "C2_s0", "C2M_s0", "C2L_s0"] if os.path.exists(os.path.join(CK, f"feats_{a}.npy"))]
feats = {a: np.load(os.path.join(CK, f"feats_{a}.npy")).astype(np.float64) for a in arms}
pairs = [(a, "C0") for a in arms if a != "C0"] + [p for p in [("C2_s0", "H16"), ("C2_s0", "C1t_s0"), ("C2M_s0", "C2_s0"), ("C2L_s0", "C2_s0"), ("C2L_s0", "H16")] if p[0] in arms and p[1] in arms]
ref = np.load(os.path.join(ROOT, "results", "stage0", "ref_stats.npz"))
t = time.time()
res = paired_boot(FIDRef(ref["mu"], ref["sigma"]), feats, pairs)
res.update({"n_boot": 1000, "arms": arms, "cpu_seconds": time.time() - t, "note": "CI conditional on the reference statistics and n = 2000"})
atomic_write_json(os.path.join(ROOT, "results", "e3c", "fid_boot.json"), res)
print(json.dumps({k: v for k, v in res.items() if k in ("fid", "pairs")}, indent=1))
