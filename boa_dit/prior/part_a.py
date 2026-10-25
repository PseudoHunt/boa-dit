"""Part A: how much of the W4 error is routing (Q, K) error? See results/prior/rule.md.

W4 uniform model; per arm, the FP rows of attn.qkv (Q,K = rows 0-2303, V = rows 2304-3455; weight and bias) are swapped in
by a forward pre-hook reading CTX.step (pointer swap between two resident tensors, no copies per call).
  --what latent : rel. latent MSE vs FP on seeds 0-127 (base also checked against the stored per-image values)
  --what fid    : FID-2k on seeds 0-1999 + ResNet-50 label accuracy on 0-511 (the base arm: 512 images, labels only)
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.attn_share.gate2 import fp_top1, label_acc, resnet_top1
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.eval_fid import REF_STATS, fid_from_stats, inception_stats
from boa_dit.eval_latent_mse import per_image
from boa_dit.model import decode, load_dit, load_quant_state, load_vae, make_diffusion
from boa_dit.sample import CTX, sample_seeds

OUT = os.path.join(RESULTS, "prior")
W4 = os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt")
C = 1152
QK, V = slice(0, 2 * C), slice(2 * C, 3 * C)
ALL = tuple(range(28))
ARMS = {"base": (None, (), None), "A1": (QK, ALL, None), "A2": (QK, ALL, "lt14"), "A3": (QK, ALL, "ge14"),
        "A4": (V, ALL, None), "A5": (QK, (2,), None), "A6": (QK, tuple(range(5)), None), "A7": (QK, tuple(range(5)), "lt14")}


def build(arm, dev):
    """W4 model with the arm's FP rows switched in per step. Returns (model, hooks)."""
    rows, blocks, when = ARMS[arm]
    fp = load_dit("cpu")
    m = load_dit(dev)
    load_quant_state(m, W4)
    hooks = []
    for l in blocks:
        lin = m.blocks[l].attn.qkv
        Wq, bq = lin.weight.data, lin.bias.data
        Wm, bm = Wq.clone(), bq.clone()
        Wm[rows] = fp.blocks[l].attn.qkv.weight.data[rows].to(dev)
        bm[rows] = fp.blocks[l].attn.qkv.bias.data[rows].to(dev)
        assert torch.equal(Wm[rows].cpu(), fp.blocks[l].attn.qkv.weight.data[rows])

        def pre(mod, inp, Wq=Wq, bq=bq, Wm=Wm, bm=bm):
            s = CTX.step
            on = s is not None and (when is None or (when == "lt14" and s < 14) or (when == "ge14" and s >= 14))
            mod.weight.data, mod.bias.data = (Wm, bm) if on else (Wq, bq)
        hooks.append(lin.register_forward_pre_hook(pre))
    return m, hooks


@torch.no_grad()
def latent(arm, dev):
    out = os.path.join(OUT, "eval", f"latent_{arm}.json")
    if os.path.exists(out):
        return
    xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:128]
    with Run("prior", f"latent_{arm}", dev, {"arm": arm}) as run:
        m, _ = build(arm, dev)
        x = sample_seeds(m, make_diffusion(50), list(range(128)), 1.5, 32, log=False)
        rel = per_image(x, xf)[0]
        res = {"arm": arm, "seeds": "0-127", "rel_mse_mean": float(rel.mean()), "rel_mse_per_image": rel.tolist(), **run.record()}
    if arm == "base":
        stored = torch.tensor(json.load(open(os.path.join(RESULTS, "stage2", "eval",
                                                          "latent_uniform_w4_K1_lam1_brcfg.json")))["rel_mse_per_image"][:128])
        res["integrity_max_abs_diff_vs_stored"] = float((rel - stored).abs().max())
    atomic_write_json(out, res)
    print(arm, {k: v for k, v in res.items() if k not in ("rel_mse_per_image", "config")}, flush=True)


@torch.no_grad()
def fid(arm, dev):
    out = os.path.join(OUT, "eval", f"fid_{arm}.json")
    if os.path.exists(out):
        return
    n = 512 if arm == "base" else 2000
    with Run("prior", f"fid_{arm}", dev, {"arm": arm, "n": n}) as run:
        m, _ = build(arm, dev)
        part = os.path.join(CKPT_DIR, "prior", f"fid_{arm}.partial")
        x = sample_seeds(m, make_diffusion(50), list(range(n)), 1.5, 32, partial=part, resume=True, log=False)
        del m
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev))
        del vae
        t1 = resnet_top1(imgs[:512], dev)
        res = {"arm": arm, "n": n, "label_acc_512": label_acc(t1), "class_changed_frac_512": float((t1 != fp_top1(dev)).float().mean())}
        if n == 2000:
            mu, sig = inception_stats(imgs.permute(0, 2, 3, 1).numpy(), dev)
            ref = np.load(REF_STATS)
            res["fid2k"] = fid_from_stats(mu, sig, ref["mu"], ref["sigma"])
        res.update(run.record())
    atomic_write_json(out, res)
    os.remove(part)
    print(arm, res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["latent", "fid"], required=True)
    ap.add_argument("--arms", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    os.makedirs(os.path.join(CKPT_DIR, "prior"), exist_ok=True)
    for arm in a.arms.split(","):
        (latent if a.what == "latent" else fid)(arm, a.device)


if __name__ == "__main__":
    main()
