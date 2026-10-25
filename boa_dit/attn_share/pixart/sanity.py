"""PixArt-Σ §4 (log-only): 8 prompts, seed 0, FP vs self-attention replaced at steps 0-7 by the kernel form
out_i = (floor * sum_j v_j + (kres * v)_i) / (floor * N + (kres * 1)_i), floor = min(kappa), kres = kappa - floor, from the
§2 cond / uncond kernels of that (block, step, head). FFT convolution; nothing larger than one kernel is stored.
Checks the FFT path against the dense kernel-rebuilt map once. CLIP score (ViT-L/14, 100 * cos) for both.
-> results/attn_share/pixart/sanity.json, sanity_grid.jpg
"""
import argparse
import os

import torch

from boa_dit.attn_share.gate1 import offsets
from boa_dit.attn_share.pixart.common import (CK, CTX, G, OUT, PROMPTS, decode_images, embeds_for, generators,
                                              kernel_fft, load_pipe, patch_self_attention, prompt_embeds, share_out)
from boa_dit.common import Run, atomic_write_json

K_SHARE = 8
IMGS = [(p, 0) for p in range(len(PROMPTS))]


def kernels(dev):
    kap = torch.load(os.path.join(CK, "kappa.pt"), map_location="cpu")["kappa"]
    out = {}
    for s in range(K_SHARE):
        for l in range(kap["cond"].shape[1]):
            kk = torch.stack([kap["uncond"][s, l], kap["cond"][s, l]]).to(dev)        # [2 (uncond, cond), H, 127, 127]
            fl = kk.flatten(-2).min(-1).values                                         # [2, H]
            out[(l, s)] = (fl, kernel_fft(kk - fl[..., None, None]))
    return out


@torch.no_grad()
def check(dev, kf):
    """FFT kernel form vs the dense row-normalized kappa map at N = 4096, one (block, step), all heads."""
    kap = torch.load(os.path.join(CK, "kappa.pt"), map_location="cpu")["kappa"]
    l, s = 13, 3
    kk = torch.stack([kap["uncond"][s, l], kap["cond"][s, l]]).to(dev)
    v = torch.randn(2, G * G, 16, 72, device=dev)
    out = share_out(v, *kf[(l, s)])                                                    # [2, N, H, hd]
    offs = offsets(G).to(dev)
    worst = 0.0
    for h in (0, 7, 15):
        for r in (0, 1):
            P = kk[r, h].flatten()[offs]
            P = P / P.sum(-1, keepdim=True)
            ref = P @ v[r, :, h]
            worst = max(worst, float((out[r, :, h] - ref).abs().max() / ref.abs().max()))
    return worst


@torch.no_grad()
def clip_scores(images, dev):
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    x = proc(text=[PROMPTS[p] for p, _ in IMGS], images=images, return_tensors="pt", padding=True).to(dev)
    out = m(**x)
    ie, te = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True), out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    return (100 * (ie * te).sum(-1).clamp_min(0)).tolist()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    emb = prompt_embeds(dev)
    with Run("attn_share", "pixart_sanity", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        procs = patch_self_attention(pipe)
        kf = kernels(dev)
        err = check(dev, kf)
        print("fft kernel form vs dense map, max rel", err, flush=True)
        lat = {}
        for name in ("fp", "shared"):
            for p in procs:
                p.mode, p.k, p.kfn = ("share", K_SHARE, (lambda l, s: kf[(l, s)])) if name == "shared" else (None, 0, None)
            CTX.step = -1
            lat[name] = pipe(prompt=None, negative_prompt=None, num_inference_steps=20, guidance_scale=4.5,
                             generator=generators(IMGS, dev), output_type="latent", max_sequence_length=300,
                             **embeds_for(IMGS, emb, dev)).images
        imgs = {n: decode_images(pipe, z) for n, z in lat.items()}
        rel = float(((lat["shared"] - lat["fp"]).float().pow(2).flatten(1).sum(1) / lat["fp"].float().pow(2).flatten(1).sum(1)).mean())
        del pipe
        torch.cuda.empty_cache()
        clip = {n: clip_scores(im, dev) for n, im in imgs.items()}
        rec = run.record()
    from PIL import Image
    grid = Image.new("RGB", (8 * 256, 2 * 256))
    for c in range(8):
        grid.paste(imgs["fp"][c].resize((256, 256)), (c * 256, 0))
        grid.paste(imgs["shared"][c].resize((256, 256)), (c * 256, 256))
    grid.save(os.path.join(OUT, "sanity_grid.jpg"), quality=90)
    atomic_write_json(os.path.join(OUT, "sanity.json"), {
        "shared_steps": f"0-{K_SHARE - 1}", "form": "floor=min(kappa) global-mean term + FFT convolution with kres, row-normalized",
        "fft_vs_dense_max_rel": err, "clip_vit_l14": clip,
        "clip_mean": {n: sum(v) / len(v) for n, v in clip.items()}, "rel_latent_mse_vs_fp": rel,
        "grid": "top row FP, bottom row shared; prompts 1-8 left to right, seed 0", **rec})
    print({n: sum(v) / len(v) for n, v in clip.items()}, "rel latent mse", rel)


if __name__ == "__main__":
    main()
