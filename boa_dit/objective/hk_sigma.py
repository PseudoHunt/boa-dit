"""H_K sigma (rule.md D2.4): least-squares fit of log|B(f)|^2 = c - 4 pi^2 sigma^2 f_tok^2 (Gaussian blur over the 16x16
token grid, f_tok in cycles/token = 2 x cycles/latent-px) to log G-hat(band) = log mean over steps 0-20 of G(t, band)/G(t,
band 1). Each band is weighted by the fraction of its DFT bins the token grid can represent (|f_x|, |f_y| <= 0.25 cycles
per latent px) and placed at the centroid radius of that representable part. Bands 3-4 have no representable part, so
the fit is set by bands 1-2: a token blur cannot act on the within-patch frequencies (recorded in REPORT).
"""
import json
import math
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, ROOT, atomic_write_json
from boa_dit.objective.d2 import band_masks


def main():
    d = json.load(open(os.path.join(RESULTS, "objective", "d2.json")))
    G = np.array([[np.mean(d["G"][str(t)][f"band{b + 1}"]) for b in range(4)] for t in (0, 5, 10, 15, 20)])
    Ghat = (G / G[:, :1]).mean(0)
    f = torch.fft.fftfreq(32)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    inside = (f[:, None].abs() <= 0.25) & (f[None, :].abs() <= 0.25)
    w, ft = [], []
    for m in band_masks():
        mi = m & inside
        w.append(float(mi.sum() / m.sum()))
        ft.append(2 * float(r[mi].mean()) if mi.any() else float("nan"))
    use = [b for b in range(4) if w[b] > 0]
    # weighted LS in (c, s) for y = c - s x, x = 4 pi^2 f^2, s = sigma^2
    x = np.array([4 * math.pi ** 2 * ft[b] ** 2 for b in use]); y = np.log(Ghat[use]); ww = np.array([w[b] for b in use])
    A = np.stack([np.ones_like(x), -x], 1) * np.sqrt(ww)[:, None]
    c, s = np.linalg.lstsq(A, y * np.sqrt(ww), rcond=None)[0]
    sigma = float(math.sqrt(max(s, 0.0)))
    res = {"G_hat_norm_band1_steps0_20": Ghat.tolist(), "band_weight_representable": w, "band_ftok_centroid": ft,
           "sigma_tokens": sigma, "fit_c": float(c),
           "fitted_response_sq": [float(math.exp(-4 * math.pi ** 2 * sigma ** 2 * ft[b] ** 2)) if w[b] > 0 else None for b in range(4)]}
    atomic_write_json(os.path.join(RESULTS, "objective", "hk_sigma.json"), res)
    atomic_write_json(os.path.join(ROOT, "configs", "objective_hk.json"), {"sigma_tokens": sigma})
    print(res)


if __name__ == "__main__":
    main()
