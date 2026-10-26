"""PixArt K filter (user follow-up): spectral gain on the 64x64 token grid, sqrt(G) interpolated linearly in latent radius
between the band-1 and band-2 centroids of PixArt's band-gain table (band_gain.json, design set), flat outside, pooled
over the first half of the steps (probed steps 0-8), normalized so the least-amplified frequency = 1 (scale-free for
the solver). Token frequency f (cycles/token) -> latent radius |f|/2. Writes configs/w4a4_kfilter.json.
"""
import json
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, ROOT, atomic_write_json
from boa_dit.w4a4.band_gain import band_masks


def main():
    bg = json.load(open(os.path.join(RESULTS, "w4a4", "band_gain.json")))
    f = torch.fft.fftfreq(128)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    c1, c2 = (float(r[m].mean()) for m in band_masks()[:2])
    steps = [k for k in bg["G"] if int(k) <= 8]
    ratio = float(np.mean([np.mean(bg["G"][k]["band2"]) / np.mean(bg["G"][k]["band1"]) for k in steps]))
    ft = torch.fft.fftfreq(64, dtype=torch.float64)
    rt = torch.sqrt(ft[:, None] ** 2 + ft[None, :] ** 2) / 2
    t = ((rt - c1) / (c2 - c1)).clamp(0, 1)
    g = (1.0 + (np.sqrt(ratio) - 1.0) * t).numpy()
    g = g / g.min()
    out = {"prompt_set": bg["prompt_set"], "gain": g.tolist(), "band2_over_band1_steps0_8": ratio,
           "centroids": [c1, c2], "min": float(g.min()), "max": float(g.max())}
    atomic_write_json(os.path.join(ROOT, "configs", "w4a4_kfilter.json"), out)
    print({k: v for k, v in out.items() if k != "gain"})


if __name__ == "__main__":
    main()
