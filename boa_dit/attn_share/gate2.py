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
        "k10-onemap": (10, "onemap"), "k10-uniform": (10, "uniform"), "fp": (0, "off"),
        # Gate 3 storage variants at k = k_pass (set by --k); V1 = the Gate 2 arm k{k_pass} itself
        "V2": (None, "windowed"), "V3": (None, "kernel"), "V4": (None, "sparse32"), "V5": (None, "condonly")}
GATE3 = ("V2", "V3", "V4", "V5")
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


def storage_bytes(kind, k, L=28, H=16, N=256):
    """Bytes stored for both CFG branches (fp16 values; uint8 column indices for the sparse variant)."""
    W = -(-k // 5)
    per = {"perstep": k * L * H * N * N * 2, "windowed": W * L * H * N * N * 2, "kernel": W * L * H * 961 * 2,
           "sparse32": k * L * H * N * 32 * 3, "condonly": k * L * H * N * N * 2}[kind]
    return per if kind == "condonly" else 2 * per


def gate3_maps(kind, k, dev):
    from boa_dit.attn_share.gate1 import offsets
    Mc, Mu = load_maps(k, dev)
    if kind == "condonly":
        return Mc, Mc
    out = []
    for M in (Mc, Mu):
        if kind in ("windowed", "kernel"):
            Mw = torch.stack([M[w:w + 5].float().mean(0) for w in range(0, k, 5)])        # [W, L, H, N, N]
            if kind == "kernel":
                offs = offsets().to(dev)
                flat = offs.flatten()
                cnt = torch.zeros(961, device=dev).index_add_(0, flat, torch.ones_like(flat, dtype=torch.float))
                kap = torch.zeros(*Mw.shape[:3], 961, device=dev).index_add_(3, flat, Mw.flatten(3)) / cnt
                Mw = kap[..., offs]
                Mw = Mw / Mw.sum(-1, keepdim=True)
            out.append(Mw.half())
        else:  # sparse32: top-32 entries per row of the per-step map, renormalized
            S = torch.empty_like(M)
            for s in range(k):
                v, i = M[s].float().topk(32, dim=-1)
                S[s] = torch.zeros_like(M[s], dtype=torch.float).scatter_(-1, i, v / v.sum(-1, keepdim=True)).half()
            out.append(S)
    return out


def configure(arm, dev, k_override=None):
    k, kind = ARMS[arm]
    if arm in GATE3:
        k = k_override
        CTRL.k, CTRL.heads, CTRL.mode = k, None, "share"
        Mc, Mu = gate3_maps(kind, k, dev)
        if kind in ("windowed", "kernel"):
            CTRL.maps = lambda b, s: (Mc[s // 5, b], Mu[s // 5, b])
        else:
            CTRL.maps = lambda b, s: (Mc[s, b], Mu[s, b])
        return
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
    ap.add_argument("--k", type=int, default=None, help="k_pass, Gate 3 variants only")
    a = ap.parse_args()
    dev = a.device
    g3 = a.arm in GATE3
    assert not g3 or a.k, "--k (k_pass) is required for Gate 3 variants"
    out_json = os.path.join(OUT, "eval", f"gate3_{a.arm}_k{a.k}.json" if g3 else f"gate2_{a.arm}.json")
    if os.path.exists(out_json):
        print("exists", out_json)
        return
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with Run("attn_share", f"gate3_{a.arm}_k{a.k}" if g3 else f"gate2_{a.arm}", dev, vars(a)) as run:
        ref_top1 = fp_top1(dev)
        model, diffusion = patch(load_dit(dev)), make_diffusion(50)
        configure(a.arm, dev, a.k)
        part = os.path.join(CKPT_DIR, f"attn_share_{a.arm}{a.k or ''}_fid2k.partial")
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
        atomic_torch_save(imgs[:N_GRID].clone(), os.path.join(MAPS, f"grid_{a.arm}{a.k or ''}.pt"))
        k = a.k if g3 else ARMS[a.arm][0]
        res = {"arm": a.arm, "k": k, "kind": ARMS[a.arm][1], "n_fid": N_FID, "fid2k": fid,
               "storage_bytes": storage_bytes(ARMS[a.arm][1], k) if g3 else
               (storage_bytes("perstep", k) if ARMS[a.arm][1] in ("all", "sel") else None),
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
