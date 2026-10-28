"""Paired FID bootstrap, DNS side (results/dns/PROTOCOL.md §3): baseline = E3C C0 features (reused), DNS, Q-Drift, C-DNS arms if present.
CPU. Run: python scripts/dns_fidboot.py"""
import json, os, sys, time
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import atomic_write_json
from boa_dit.fidboot import FIDRef, paired_boot
feats = {"base": np.load("/home/boa-dit/ckpt/e3c/fid/feats_C0.npy").astype(np.float64)}
for a in ("dns", "qdrift", "cdns_ind", "cdns_full"):
    f = f"/home/boa-dit/ckpt/dns/fid/feats_{a}.npy"
    if os.path.exists(f):
        feats[a] = np.load(f).astype(np.float64)
pairs = [(a, "base") for a in feats if a != "base"] + [p for p in [("dns", "qdrift"), ("cdns_ind", "dns"), ("cdns_full", "cdns_ind"), ("cdns_full", "dns")] if p[0] in feats and p[1] in feats]
ref = np.load(os.path.join(ROOT, "results", "stage0", "ref_stats.npz"))
t = time.time()
res = paired_boot(FIDRef(ref["mu"], ref["sigma"]), feats, pairs)
res.update({"n_boot": 1000, "cpu_seconds": time.time() - t, "baseline": "E3C C0 (reused: same artifacts, sampler, seeds, labels, metric)", "note": "CI conditional on the reference statistics and n = 2000"})
atomic_write_json(os.path.join(ROOT, "results", "dns", "fid_boot.json"), res)
print(json.dumps({k: v for k, v in res.items() if k in ("fid", "pairs")}, indent=1))
