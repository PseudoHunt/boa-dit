"""Matched-seed W3 grid for G2: seeds 0-7; rows FP, uniform W3, then each W3 deflate arm with an FID run.
FP images: ckpt/fp_eval_pixels64.pt (seeds 0-63). Uniform W3: generated once (16 seeds, ~20 s) into ckpt/deflate_grids.
Arm images: ckpt/deflate_grids/<arm>.pt, saved by the FID run (seeds 0-15, same noise and labels)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import CKPT_DIR, RESULTS, atomic_savefig, atomic_torch_save  # noqa: E402

GD = os.path.join(CKPT_DIR, "deflate_grids")
ARMS = [("DC-r*", "deflate_dc_rstar_w3"), ("DC-1", "deflate_dc_r1_w3"), ("C-r*", "deflate_c_rstar_w3"),
        ("WSVD-r*", "deflate_wsvd_rstar_w3")]


@torch.no_grad()
def uniform_w3(dev="cuda:0"):
    p = os.path.join(GD, "uniform_w3_K1_lam1_brcfg.pt")
    if not os.path.exists(p):
        from boa_dit.model import decode, load_dit, load_quant_state, load_vae, make_diffusion
        from boa_dit.sample import sample_seeds
        torch.cuda.set_device(dev)
        m = load_dit(dev)
        load_quant_state(m, os.path.join(CKPT_DIR, "ckpt_uniform_w3_K1_lam1_brcfg", "model.pt"))
        x = sample_seeds(m, make_diffusion(50), list(range(32)), 1.5, 32, log=False)
        atomic_torch_save(decode(load_vae(dev), x[:16].to(dev)), p)
    return torch.load(p, map_location="cpu")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [("FP", torch.load(os.path.join(CKPT_DIR, "fp_eval_pixels64.pt"), map_location="cpu")),
            ("uniform W3", uniform_w3())]
    rows += [(lab, torch.load(os.path.join(GD, f"{n}.pt"), map_location="cpu")) for lab, n in ARMS
             if os.path.exists(os.path.join(GD, f"{n}.pt"))]
    fig, axs = plt.subplots(len(rows), 8, figsize=(16, 2.1 * len(rows)), squeeze=False)
    for r, (lab, g) in enumerate(rows):
        for c in range(8):
            axs[r, c].imshow(g[c].permute(1, 2, 0).numpy()); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(lab, fontsize=11)
    fig.tight_layout(); atomic_savefig(fig, os.path.join(RESULTS, "deflate", "g2_grid_w3.png")); plt.close(fig)
    print("grid rows:", [lab for lab, _ in rows])


if __name__ == "__main__":
    main()
