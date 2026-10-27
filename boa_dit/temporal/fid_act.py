"""FID-2k for a stage-3 DiT activation-quantization arm (rule_feedback.md): eval_fid.py protocol (seeds 0..n-1, cfg 1.5,
DDIM-50, bs 32, pytorch-fid vs the ADM reference batch), quantizer dit_act.Quantize with log=False (no counters; the
sampled latents are bit-identical to log=True, verified on a 32-seed batch). Resumable per batch.
Run: python -m boa_dit.temporal.fid_act --kind single8 --dither none [--feedback] --n 2000
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.eval_fid import REF_STATS, compute_ref, fid_from_stats, inception_stats
from boa_dit.model import decode, load_dit, load_vae, make_diffusion
from boa_dit.sample import sample_seeds
from boa_dit.temporal import act as ACT
from boa_dit.temporal.common import CK, OUT, STAGE
from boa_dit.temporal.dit_act import Quantize


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="single8", choices=list(ACT.KINDS))
    ap.add_argument("--dither", default="none", choices=list(ACT.DITHERS))
    ap.add_argument("--feedback", action="store_true")
    ap.add_argument("--n", type=int, default=2000)
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    name = f"bp_{a.kind}_{'feedback' if a.feedback else a.dither}"
    if not os.path.exists(REF_STATS):
        with Run(STAGE, "fid_ref_stats", dev, {}):
            compute_ref(dev)
    with Run(STAGE, f"fid{a.n}_dit_{name}", dev, vars(a)) as run:
        m = load_dit(dev)
        q = Quantize(m, a.kind, a.dither, dev, feedback=a.feedback, log=False)
        part = os.path.join(CK, f"fid{a.n}_dit_{name}.partial")
        x = sample_seeds(m, make_diffusion(50), list(range(a.n)), 1.5, 32, partial=part, resume=True)
        q.remove()
        del m
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev)).permute(0, 2, 3, 1).numpy()
        del vae
        mu, sigma = inception_stats(imgs, dev)
        ref = np.load(REF_STATS)
        rec = {"arm": name, "n": a.n, "fid": fid_from_stats(mu, sigma, ref["mu"], ref["sigma"]), **run.record()}
    atomic_write_json(os.path.join(OUT, "stage3", f"fid{a.n}_dit_{name}.json"), rec)
    os.remove(part)
    print(rec, flush=True)


if __name__ == "__main__":
    main()
