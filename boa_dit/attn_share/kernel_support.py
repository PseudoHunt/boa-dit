"""Effective support of the V3 offset kernel: for each (branch, window, block, head), the smallest centered square
(side 2r+1) holding >= 95% of kappa's mass (kappa = mean of the window's mean map over all (i, j) with the same 2-D
offset; 31 x 31, taken as a distribution over offsets). CPU only. -> results/attn_share/kernel_support.json + .png
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.attn_share.gate1 import offsets
from boa_dit.common import RESULTS, atomic_savefig, atomic_write_json

OUT = os.path.join(RESULTS, "attn_share")


def kernels(M, k):
    """M [S, L, H, N, N] -> kappa [W, L, H, 31, 31] from 5-step window means, s < k."""
    offs = offsets().flatten()
    cnt = torch.zeros(961).index_add_(0, offs, torch.ones(offs.shape[0]))
    out = []
    for w in range(0, k, 5):
        Mw = M[w:w + 5].float().mean(0)                                            # [L, H, N, N]
        kap = torch.zeros(*Mw.shape[:2], 961).index_add_(2, offs, Mw.flatten(2)) / cnt
        out.append(kap.reshape(*Mw.shape[:2], 31, 31))
    return torch.stack(out)


def support(kap, q=0.95):
    """Smallest side 2r+1 of a centered square holding >= q of the mass, for every leading index."""
    kap = kap / kap.sum((-1, -2), keepdim=True)
    side = torch.full(kap.shape[:-2], 31)
    for r in range(15, -1, -1):
        m = kap[..., 15 - r:16 + r, 15 - r:16 + r].sum((-1, -2))
        side = torch.where(m >= q, torch.full_like(side, 2 * r + 1), side)
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=20)
    a = ap.parse_args()
    res = {"k": a.k, "q": 0.95, "windows": [f"s{w}-{w + 4}" for w in range(0, a.k, 5)],
           "note": "side of the smallest centered square holding 95% of kappa's mass; 31 = full support; "
                   "uniform attention gives a near-uniform kappa, i.e. side ~ 31", "per_branch": {}}
    sides = {}
    for br in ("cond", "uncond"):
        M = torch.load(os.path.join(OUT, "maps", f"M_{br}.pt"), map_location="cpu", mmap=True)[:a.k]
        kap = kernels(M, a.k)
        s = support(kap)                                                           # [W, L, H]
        sides[br] = s
        # secondary: split kappa = floor + excess, floor = 10th percentile of kappa over offsets (a constant floor is
        # a global mean of V, O(N)); support of the excess and the floor's share of the mass
        flat = kap.flatten(-2)
        floor = torch.quantile(flat, 0.10, dim=-1)[..., None, None]
        excess = (kap - floor).clamp_min(0)
        floor_share = (floor[..., 0, 0] * 961 / flat.sum(-1))
        se = support(excess)
        vals, counts = np.unique(s.numpy(), return_counts=True)
        res["per_branch"][br] = {
            "side_histogram": {int(v): int(c) for v, c in zip(vals, counts)},
            "median_side": float(s.float().median()), "frac_side_le_7": float((s <= 7).float().mean()),
            "frac_side_le_15": float((s <= 15).float().mean()), "frac_full_31": float((s == 31).float().mean()),
            "median_side_per_window": [float(s[w].float().median()) for w in range(s.shape[0])],
            "median_side_per_block": [float(s[:, l].float().median()) for l in range(s.shape[1])],
            "sides": s.tolist(),
            "excess_over_floor": {"median_side": float(se.float().median()),
                                  "side_histogram": {int(v): int(c) for v, c in zip(*np.unique(se.numpy(), return_counts=True))},
                                  "frac_side_le_7": float((se <= 7).float().mean()),
                                  "frac_side_le_15": float((se <= 15).float().mean()),
                                  "median_floor_mass_share": float(floor_share.median()),
                                  "note": "floor = 10th percentile of kappa; secondary diagnostic, not requested"}}
    atomic_write_json(os.path.join(OUT, "kernel_support.json"), res)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(13, 4))
    for br, c in (("cond", "C0"), ("uncond", "C1")):
        s = sides[br].flatten().numpy()
        axs[0].hist(s, bins=np.arange(0, 33, 2), alpha=0.6, label=br, color=c)
    axs[0].set_xlabel("side of centered square holding 95% of κ (31 = full)"); axs[0].set_ylabel("(window, block, head)")
    axs[0].legend()
    im = axs[1].imshow(sides["cond"].float().median(0).values.T.numpy(), aspect="auto", origin="lower",
                       cmap="viridis", vmin=1, vmax=31)
    axs[1].set_xlabel("block"); axs[1].set_ylabel("head"); axs[1].set_title("cond: median side over windows")
    fig.colorbar(im, ax=axs[1]); fig.tight_layout()
    atomic_savefig(fig, os.path.join(OUT, "kernel_support.png")); plt.close(fig)
    print({br: {k: v for k, v in d.items() if k != "sides"} for br, d in res["per_branch"].items()})


if __name__ == "__main__":
    main()
