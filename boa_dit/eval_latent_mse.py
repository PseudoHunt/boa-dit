"""Stage 2 / 5.4: matched-seed latent MSE of a quantized model vs FP (primary metric).

Seeds 0..511 (disjoint from calibration 1000..1031 and Stage 1 2000..2015), cfg 1.5, ddim50, fixed batch
layout (bs 32, seeds in order) for FP and quantized runs. Relative MSE = ||x0_q - x0_fp||^2 / ||x0_fp||^2 per
image, averaged; PSNR in latent space uses the FP latent's peak-to-peak range. Pixel metrics on the first 64.
"""
import argparse
import math
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import decode, load_dit, load_quant_state, load_vae, make_diffusion
from boa_dit.sample import sample_seeds

EVAL_SEEDS = list(range(512))
HELDOUT_START = 1032   # held-out calibration seeds 1032..1095 (used for calibration-time choices only)


def fp_paths(seed_start):
    tag = "" if seed_start == 0 else f"_s{seed_start}"
    return (os.path.join(CKPT_DIR, f"fp_eval_latents{tag}.pt"), os.path.join(CKPT_DIR, f"fp_eval_pixels64{tag}.pt"))


def per_image(xq, xf):
    err = (xq - xf).flatten(1).pow(2)
    rel = err.sum(1) / xf.flatten(1).pow(2).sum(1)
    mse = err.mean(1)
    rng = (xf.flatten(1).max(1).values - xf.flatten(1).min(1).values)
    psnr = 10 * torch.log10(rng ** 2 / mse)
    return rel, mse, psnr


@torch.no_grad()
def fp_reference(dev, bs=32, n_pix=64, seed_start=0, n=512):
    FP_LAT, FP_PIX = fp_paths(seed_start)
    if not os.path.exists(FP_LAT):
        model, diffusion = load_dit(dev), make_diffusion(50)
        x = sample_seeds(model, diffusion, list(range(seed_start, seed_start + n)), 1.5, bs,
                         partial=FP_LAT + ".partial", resume=True)
        atomic_torch_save(x, FP_LAT)
        os.remove(FP_LAT + ".partial")
        del model
    xf = torch.load(FP_LAT, map_location="cpu")
    if not os.path.exists(FP_PIX):
        vae = load_vae(dev)
        atomic_torch_save(decode(vae, xf[:n_pix].to(dev)), FP_PIX)
    return xf, torch.load(FP_PIX, map_location="cpu")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="quantized run name (ckpt/ckpt_<name>/model.pt) or 'fp'")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--stage", default="stage2")
    ap.add_argument("--seed_start", type=int, default=0, help="0 = eval seeds; 1032 = held-out calibration seeds")
    ap.add_argument("--carry", action="store_true", help="attach results/deflate/quant/<name>/carry.pt (deflate arms)")
    a = ap.parse_args()
    dev = a.device
    seeds = list(range(a.seed_start, a.seed_start + a.n))
    xf, pf = fp_reference(dev, seed_start=a.seed_start, n=a.n if a.seed_start else 512)
    xf = xf[:a.n]
    tag = "" if a.seed_start == 0 else f"_heldout{a.seed_start}n{a.n}"
    out_json = os.path.join(RESULTS, a.stage, "eval", f"latent_{a.name}{tag}.json")
    with Run(a.stage, f"eval_latent_{a.name}", dev, vars(a)) as run:
        model, diffusion = load_dit(dev), make_diffusion(50)
        load_quant_state(model, os.path.join(CKPT_DIR, f"ckpt_{a.name}", "model.pt"))
        if a.carry:
            from boa_dit.deflate.carry import install_from
            install_from(model, a.name)
        part = os.path.join(CKPT_DIR, f"ckpt_{a.name}", f"eval_latents_{a.seed_start}_{a.n}.partial")
        xq = sample_seeds(model, diffusion, seeds, 1.5, 32, partial=part, resume=True)
        rel, mse, psnr = per_image(xq, xf)
        del model
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        n_pix = min(64, a.n)
        pq = decode(vae, xq[:n_pix].to(dev)).float()
        pmse = (pq - pf[:n_pix].float()).flatten(1).pow(2).mean(1)
        ppsnr = 10 * torch.log10(255.0 ** 2 / pmse)
        sem = float(rel.std() / math.sqrt(len(rel)))
        res = {"name": a.name, "n": a.n, "seed_start": a.seed_start, "rel_mse_mean": float(rel.mean()), "rel_mse_sem": sem,
               "rel_mse_median": float(rel.median()), "latent_psnr_mean": float(psnr.mean()),
               "latent_mse_mean": float(mse.mean()),
               "pixel_mse_mean_64": float(pmse.mean()), "pixel_psnr_mean_64": float(ppsnr.mean()),
               "rel_mse_per_image": rel.tolist()}
        res.update(run.record())
    atomic_write_json(out_json, res)
    os.remove(part)
    print({k: v for k, v in res.items() if k != "rel_mse_per_image"})


if __name__ == "__main__":
    main()
