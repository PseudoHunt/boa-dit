"""Proper H_K filters (user follow-up, after the band-1 oracle recovered 73%): a spectral filter on the 16x16 token grid
whose per-frequency gain is sqrt(G) interpolated from the band-1 / band-2 part of D2's table (not a Gaussian).
Token frequency f_tok (cycles/token) maps to latent radius r = |f_tok| / 2 (cycles/latent px). sqrt(G) is linear in r
between the band-1 and band-2 centroid radii, flat below the band-1 centroid, and flat at the band-2 value above the
band-2 centroid; normalized to the band-1 value.
  pooled : G = mean over steps 0-20 of G(t, band)/G(t, band 1); one filter for all buckets
  bucket : bucket b uses its own probed step 5b (one probed step per bucket), normalized to its band 1 (frequency shape
           only, so the bucket weighting stays uniform as in heig-8)
Writes configs/objective_hk_{pooled,bucket}.json = {"gain": [B][16][16]}. Run: python -m boa_dit.objective.hk_spectral
"""
import json
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, ROOT, atomic_write_json
from boa_dit.objective.d2 import band_masks


def centroids():
    f = torch.fft.fftfreq(32)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    return [float(r[m].mean()) for m in band_masks()[:2]]


def gain_grid(g1, g2, c1, c2):
    f = torch.fft.fftfreq(16, dtype=torch.float64)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2) / 2                      # token cycles -> latent cycles/px
    s1, s2 = np.sqrt(g1), np.sqrt(g2)
    t = ((r - c1) / (c2 - c1)).clamp(0, 1)
    return ((s1 + (s2 - s1) * t) / s1).numpy()


def main():
    d = json.load(open(os.path.join(RESULTS, "objective", "d2.json")))
    G = {int(t): [np.mean(d["G"][t][f"band{b}"]) for b in (1, 2)] for t in d["G"]}
    c1, c2 = centroids()
    pooled_ratio = np.mean([G[t][1] / G[t][0] for t in (0, 5, 10, 15, 20)])
    pooled = gain_grid(1.0, pooled_ratio, c1, c2)
    atomic_write_json(os.path.join(ROOT, "configs", "objective_hk_pooled.json"),
                      {"gain": [pooled.tolist()] * 10, "band2_over_band1": pooled_ratio, "centroids": [c1, c2]})
    per = [gain_grid(G[5 * b][0], G[5 * b][1], c1, c2) for b in range(10)]
    atomic_write_json(os.path.join(ROOT, "configs", "objective_hk_bucket.json"),
                      {"gain": [g.tolist() for g in per], "band2_over_band1": [G[5 * b][1] / G[5 * b][0] for b in range(10)],
                       "centroids": [c1, c2]})
    res = {"centroid_r_band1_band2": [c1, c2], "pooled_G2_over_G1": pooled_ratio,
           "bucket_G2_over_G1": [G[5 * b][1] / G[5 * b][0] for b in range(10)],
           "pooled_gain_min_max": [float(pooled.min()), float(pooled.max())]}
    atomic_write_json(os.path.join(RESULTS, "objective", "hk_spectral.json"), res)
    print(res)


if __name__ == "__main__":
    main()
