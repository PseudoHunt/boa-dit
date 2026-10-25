"""Attention sharing, Gate 2: FID-2k (primary), latent MSE / IS / ResNet-50 class change (information) per arm.

One arm per process, generation + scoring on one GPU. Seeds 0-1999, cfg 1.5, DDIM-50, bs 32 (same layout as every
earlier eval, so seeds 0-511 are directly comparable to ckpt/fp_eval_latents.pt). Maps come from Gate 1
(results/attn_share/maps/M_{cond,uncond}.pt), loaded on CPU and copied to the arm's GPU with a tensor check.

Arms: k5 k10 k15 k20 (all heads, steps s < k), k15-sel (heads with Gate 1 median within >= 0.8 at that (block,
step)), k10-onemap (M[., ., 5] for all s < 10); extra, outside the verdict: k10-uniform (uniform 1/256 maps, the
near-uniform control) and fp (patched model, sharing off: checks bit-identity with the FP reference, FP IS/grid).
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.attn_share.attn import CTRL, patch
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_fid import REF_STATS, fid_from_stats, inception_stats
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import decode, load_dit, load_vae, make_diffusion
from boa_dit.sample import sample_seeds

OUT = os.path.join(RESULTS, "attn_share")
MAPS = os.path.join(OUT, "maps")
ARMS = {"k5": (5, "all"), "k10": (10, "all"), "k15": (15, "all"), "k20": (20, "all"), "k15-sel": (15, "sel"),
        "k10-onemap": (10, "onemap"), "k10-uniform": (10, "uniform"), "fp": (0, "off")}
N_FID, N_LAT, N_GRID = 2000, 512, 16


def to_dev_checked(t, dev):
    g = t.to(dev)
    assert torch.equal(g.cpu(), t), "host->device copy mismatch"
    return g


def load_maps(k, dev):
    out = []
    for br in ("cond", "uncond"):
        m = torch.load(os.path.join(MAPS, f"M_{br}.pt"), map_location="cpu", mmap=True)[:k].clone()
        out.append(to_dev_checked(m, dev))
    return out


def configure(arm, dev):
    k, kind = ARMS[arm]
    CTRL.k, CTRL.heads = k, None
    if kind == "off":
        CTRL.mode = None
        return
    CTRL.mode = "share"
    if kind == "uniform":
        U = torch.full((16, 256, 256), 1 / 256, dtype=torch.float16, device=dev)
        CTRL.maps = lambda b, s: (U, U)
        return
    Mc, Mu = load_maps(6 if kind == "onemap" else k, dev)
    if kind == "onemap":
        CTRL.maps = lambda b, s: (Mc[5, b], Mu[5, b])
    else:
        CTRL.maps = lambda b, s: (Mc[s, b], Mu[s, b])
    if kind == "sel":
        w = torch.tensor(json.load(open(os.path.join(OUT, "gate1_stats.json")))["stats"]["within_median"])
        sel = to_dev_checked((w >= 0.8).contiguous(), dev)                       # [25, 28, 16]
        CTRL.heads = lambda b, s: sel[s, b]


@torch.no_grad()
def resnet_top1(imgs_u8_nchw, dev, bs=64):
    import torchvision as tv
    wts = tv.models.ResNet50_Weights.IMAGENET1K_V2
    net, tf = tv.models.resnet50(weights=wts).to(dev).eval(), wts.transforms()
    out = []
    for i in range(0, len(imgs_u8_nchw), bs):
        x = tf(imgs_u8_nchw[i:i + bs].to(dev))
        out.append(net(x).argmax(-1).cpu())
    return torch.cat(out)


@torch.no_grad()
def inception_score(imgs_u8_nchw, dev, splits=10, bs=100):
    import torchvision as tv
    wts = tv.models.Inception_V3_Weights.IMAGENET1K_V1
    net, tf = tv.models.inception_v3(weights=wts).to(dev).eval(), wts.transforms()
    p = torch.cat([net(tf(imgs_u8_nchw[i:i + bs].to(dev))).softmax(-1).double().cpu()
                   for i in range(0, len(imgs_u8_nchw), bs)])
    scores = []
    for c in p.chunk(splits):
        kl = (c * (c.clamp_min(1e-12).log() - c.mean(0, keepdim=True).log())).sum(1).mean()
        scores.append(float(kl.exp()))
    return float(np.mean(scores)), float(np.std(scores))


def fp_top1(dev):
    path = os.path.join(OUT, "fp_resnet_top1_512.json")
    if not os.path.exists(path):
        xf, _ = fp_reference(dev)
        vae = load_vae(dev)
        imgs = decode(vae, xf[:N_LAT].to(dev))
        del vae
        atomic_write_json(path, {"top1": resnet_top1(imgs, dev).tolist(), "source": "ckpt/fp_eval_latents.pt"})
    return torch.tensor(json.load(open(path))["top1"])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    out_json = os.path.join(OUT, "eval", f"gate2_{a.arm}.json")
    if os.path.exists(out_json):
        print("exists", out_json)
        return
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with Run("attn_share", f"gate2_{a.arm}", dev, vars(a)) as run:
        ref_top1 = fp_top1(dev)
        model, diffusion = patch(load_dit(dev)), make_diffusion(50)
        configure(a.arm, dev)
        part = os.path.join(CKPT_DIR, f"attn_share_{a.arm}_fid2k.partial")
        try:
            x = sample_seeds(model, diffusion, list(range(N_FID)), 1.5, 32, partial=part, resume=True)
        finally:
            CTRL.mode, CTRL.maps, CTRL.heads = None, None, None
        del model
        torch.cuda.empty_cache()
        xf, _ = fp_reference(dev)
        rel, _, _ = per_image(x[:N_LAT], xf)
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev))                                            # [N, 3, 256, 256] uint8 (CPU)
        del vae
        mu, sigma = inception_stats(imgs.permute(0, 2, 3, 1).numpy(), dev)
        ref = np.load(REF_STATS)
        fid = fid_from_stats(mu, sigma, ref["mu"], ref["sigma"])
        is_mean, is_std = inception_score(imgs, dev)
        top1 = resnet_top1(imgs[:N_LAT], dev)
        atomic_torch_save(imgs[:N_GRID].clone(), os.path.join(MAPS, f"grid_{a.arm}.pt"))
        res = {"arm": a.arm, "k": ARMS[a.arm][0], "kind": ARMS[a.arm][1], "n_fid": N_FID, "fid2k": fid,
               "is_mean": is_mean, "is_std": is_std,
               "is_note": "torchvision Inception-v3, 10 splits of 200; not comparable to published IS",
               "rel_mse_mean_512": float(rel.mean()), "rel_mse_median_512": float(rel.median()),
               "class_changed_frac_512": float((top1 != ref_top1).float().mean()),
               "rel_mse_per_image": rel.tolist(), **run.record()}
    atomic_write_json(out_json, res)
    os.remove(part)
    print({k: v for k, v in res.items() if k != "rel_mse_per_image"}, flush=True)


if __name__ == "__main__":
    main()
