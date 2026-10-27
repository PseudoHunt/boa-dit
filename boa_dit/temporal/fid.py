"""FID-2k of a Part A arm (rule.md A2): eval_fid.py protocol (seeds 0..n-1, cfg 1.5, DDIM-50, bs 32, pytorch-fid vs the
ADM reference batch) with the arm's step-indexed tables. Computes the reference statistics first if missing.
Run: python -m boa_dit.temporal.fid --arm b --n 2000
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.eval_fid import REF_STATS, compute_ref, fid_from_stats, inception_stats
from boa_dit.model import decode, load_vae, make_diffusion
from boa_dit.sample import sample_seeds
from boa_dit.temporal.common import CK, OUT, STAGE
from boa_dit.temporal.dit import load_arm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    if not os.path.exists(REF_STATS):
        with Run(STAGE, "fid_ref_stats", dev, {}):
            compute_ref(dev)
    with Run(STAGE, f"fid{a.n}_{a.arm}", dev, vars(a)) as run:
        m, _ = load_arm(a.arm, dev)
        part = os.path.join(CK, f"fid{a.n}_{a.arm}.partial")
        x = sample_seeds(m, make_diffusion(50), list(range(a.n)), 1.5, 32, partial=part, resume=True)
        del m
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev)).permute(0, 2, 3, 1).numpy()
        del vae
        mu, sigma = inception_stats(imgs, dev)
        ref = np.load(REF_STATS)
        fid = fid_from_stats(mu, sigma, ref["mu"], ref["sigma"])
        rec = {"arm": a.arm, "n": a.n, "fid": fid, **run.record()}
    atomic_write_json(os.path.join(OUT, "eval", f"fid{a.n}_{a.arm}.json"), rec)
    os.remove(part)
    print(rec, flush=True)


if __name__ == "__main__":
    main()
