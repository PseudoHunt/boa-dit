"""Image eval (rule.md): MJHQ prompts mjhq1k[:n] at 1024^2, seed = prompt index, the model's default sampler; uint8
images saved in chunks of 8 (resumable) under ckpt/w4a4/img/<arm>/; then PSNR and LPIPS (VGG) vs the bf16 images.
  --arm bf16 | <quantized arm name>
Run: python -m boa_dit.w4a4.gen --arm hsvd_r4_b0.5 --n 256 --device cuda:0
"""
import argparse
import math
import os

import torch

from boa_dit.common import RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.w4a4.common import CK, load_embeds, load_pipe, sample

CHUNK = 8


def img_dir(arm):
    return os.path.join(CK, "img", arm)


@torch.no_grad()
def generate(arm, n, dev, pipe=None):
    """pipe: an externally built (e.g. deepcompressor) pipeline; else bf16 + our QLinear install for `arm`."""
    d = img_dir(arm)
    os.makedirs(d, exist_ok=True)
    todo = [i for i in range(0, n, CHUNK) if not os.path.exists(os.path.join(d, f"{i:05d}.pt"))]
    if not todo:
        return
    if pipe is None:
        pipe = load_pipe(dev)
        if arm != "bf16":
            from boa_dit.w4a4.screen import install
            install(pipe.transformer, arm)
    e = load_embeds("mjhq1k")
    for i in todo:
        idx = list(range(i, min(n, i + CHUNK)))
        imgs = sample(pipe, e, idx, dev, output_type="pt")                    # [b, 3, H, W] in [0, 1]
        atomic_torch_save((imgs.float().clamp(0, 1) * 255).round().to(torch.uint8).cpu(), os.path.join(d, f"{i:05d}.pt"))
    del pipe
    torch.cuda.empty_cache()


def load_imgs(arm, n):
    return torch.cat([torch.load(os.path.join(img_dir(arm), f"{i:05d}.pt")) for i in range(0, n, CHUNK)])[:n]


@torch.no_grad()
def metrics(arm, n, dev):
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    ref, q = load_imgs("bf16", n), load_imgs(arm, n)
    psnr, lp = [], []
    for i in range(n):
        a, b = ref[i:i + 1].to(dev).float(), q[i:i + 1].to(dev).float()
        mse = (a - b).pow(2).mean()
        psnr.append(float(10 * torch.log10(255.0 ** 2 / mse.clamp_min(1e-10))))
        lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
    return psnr, lp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    with Run("w4a4", f"gen{a.n}_{a.arm}", a.device, vars(a)) as run:
        generate(a.arm, a.n, a.device)
        res = {"arm": a.arm, "n": a.n}
        if a.arm != "bf16":
            psnr, lp = metrics(a.arm, a.n, a.device)
            res.update({"psnr_mean": sum(psnr) / len(psnr), "lpips_mean": sum(lp) / len(lp), "lpips_net": "vgg (lpips 0.1.4)",
                        "psnr_per_image": psnr, "lpips_per_image": lp})
        res.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "w4a4", "eval", f"img{a.n}_{a.arm}.json"), res)
    print({k: v for k, v in res.items() if not k.endswith("per_image") and k != "config"}, flush=True)


if __name__ == "__main__":
    main()
