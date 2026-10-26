"""Protocol control: PSNR / LPIPS sensitivity floor at 1024^2. The unquantized pipeline in fp16 vs the bf16 references
(same MJHQ prompts, seeds, scheduler): how much PSNR a near-zero perturbation costs. Batch 2 (memory).
Run: python -m boa_dit.w4a4.dtype_control --n 64 --device cuda:0
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.w4a4 import gen
from boa_dit.w4a4.common import MODEL_DIR, load_embeds, sample


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    from diffusers import PixArtSigmaPipeline
    arm = "fp16_unquantized"
    with Run("w4a4", f"dtype_control_{arm}", dev, vars(a)) as run:
        pipe = PixArtSigmaPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, text_encoder=None).to(dev)
        pipe.set_progress_bar_config(disable=True)
        e = load_embeds("mjhq1k")
        d = gen.img_dir(arm)
        os.makedirs(d, exist_ok=True)
        for i in range(0, a.n, gen.CHUNK):
            if os.path.exists(os.path.join(d, f"{i:05d}.pt")):
                continue
            chunk = []
            for j in range(i, min(a.n, i + gen.CHUNK), 2):
                chunk.append(sample(pipe, e, list(range(j, min(a.n, j + 2))), dev, output_type="pt"))
            imgs = torch.cat(chunk)
            atomic_torch_save((imgs.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu(), os.path.join(d, f"{i:05d}.pt"))
        del pipe
        torch.cuda.empty_cache()
        psnr, lp = gen.metrics(arm, a.n, dev)
        res = {"arm": arm, "n": a.n, "psnr_mean": float(np.mean(psnr)), "lpips_mean": float(np.mean(lp)),
               "note": "fp16 pipeline, no quantization, vs bf16 references: the sensitivity floor of the image metric",
               "psnr_per_image": psnr, "lpips_per_image": lp, **run.record()}
    atomic_write_json(os.path.join(RESULTS, "w4a4", "debug", f"control_{arm}.json"), res)
    print(f"fp16 unquantized vs bf16, MJHQ-{a.n}: PSNR {res['psnr_mean']:.2f} dB, LPIPS {res['lpips_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()
