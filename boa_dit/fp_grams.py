"""Per-bucket input Grams of every block linear on the FP model (Gate A and the guard selection).

Calibration seeds 1000-1031, both CFG branches, 50 steps, B = 10 buckets. Inputs propagate through FP blocks.
Saves ckpt/fp_grams/block{l}.pt = {layer: [B, d, d] fp32}, each the branch-weighted (s^2 : (s-1)^2, normalized)
mean of x x^T over the bucket's tokens.
"""
import argparse
import os

import torch

from boa_dit.calib import GramAccumulator, capture_block0_inputs, run_block
from boa_dit.common import CKPT_DIR, Run, atomic_torch_save
from boa_dit.model import LINEARS, load_dit, make_diffusion
from boa_dit.quantize import CALIB_SEEDS

OUT = os.path.join(CKPT_DIR, "fp_grams")


def load_block_grams(l):
    return torch.load(os.path.join(OUT, f"block{l}.pt"), map_location="cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--cfg", type=float, default=1.5)
    a = ap.parse_args()
    dev, B = a.device, 10
    wc, wu = a.cfg ** 2, (a.cfg - 1) ** 2
    wc, wu = wc / (wc + wu), wu / (wc + wu)
    os.makedirs(OUT, exist_ok=True)
    with Run("gateA", "fp_grams", dev, vars(a)):
        model, diffusion = load_dit(dev), make_diffusion(50)
        S, N = diffusion.num_timesteps, len(CALIB_SEEDS)
        X, C = capture_block0_inputs(model, diffusion, CALIB_SEEDS, a.cfg, dev)
        for l, blk in enumerate(model.blocks):
            path = os.path.join(OUT, f"block{l}.pt")
            if os.path.exists(path):
                run_block(blk, X, C, S, N, B, write_back=True)
                continue
            acc = GramAccumulator(blk, B, dev)
            try:
                run_block(blk, X, C, S, N, B, acc=acc)
            finally:
                acc.remove()
            means = acc.means()
            del acc
            out = {ln: torch.stack([wc * means[(ln, b, 0)] + wu * means[(ln, b, 1)] for b in range(B)]) for ln in LINEARS}
            atomic_torch_save(out, path)
            del means, out
            torch.cuda.empty_cache()
            run_block(blk, X, C, S, N, B, write_back=True)   # FP outputs -> next block's inputs
            print(f"block {l} grams saved", flush=True)


if __name__ == "__main__":
    main()
