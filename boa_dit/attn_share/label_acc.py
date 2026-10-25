"""ResNet-50 label accuracy for Gate 2 arms that ran before predictions were stored.

Regenerates seeds 0-511 only (same bs-32 layout as the arm's FID run, so the latents are the same images; checked
against the arm's stored per-image rel. latent MSE), decodes, classifies. -> results/attn_share/eval/label_acc_<arm>.json
"""
import argparse
import json
import os

import torch

from boa_dit.attn_share.attn import CTRL, patch
from boa_dit.attn_share.gate2 import N_LAT, OUT, configure, fp_top1, label_acc, resnet_top1
from boa_dit.common import CKPT_DIR, Run, atomic_write_json
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import decode, load_dit, load_vae, make_diffusion
from boa_dit.sample import sample_seeds


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    out_json = os.path.join(OUT, "eval", f"label_acc_{a.arm}.json")
    if os.path.exists(out_json):
        return
    stored = json.load(open(os.path.join(OUT, "eval", f"gate2_{a.arm}.json")))
    with Run("attn_share", f"label_acc_{a.arm}", dev, vars(a)) as run:
        ref_top1 = fp_top1(dev)
        model, diffusion = patch(load_dit(dev)), make_diffusion(50)
        configure(a.arm, dev)
        part = os.path.join(CKPT_DIR, f"attn_share_{a.arm}_lab512.partial")
        try:
            x = sample_seeds(model, diffusion, list(range(N_LAT)), 1.5, 32, partial=part, resume=True, log=False)
        finally:
            CTRL.mode, CTRL.maps, CTRL.heads = None, None, None
        del model
        xf, _ = fp_reference(dev)
        rel = per_image(x, xf)[0]
        diff = float((rel - torch.tensor(stored["rel_mse_per_image"])).abs().max())
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev))
        del vae
        top1 = resnet_top1(imgs, dev)
        res = {"arm": a.arm, "label_acc_512": label_acc(top1), "top1_512": top1.tolist(),
               "class_changed_frac_512": float((top1 != ref_top1).float().mean()),
               "class_changed_frac_512_stored": stored["class_changed_frac_512"],
               "max_abs_diff_rel_mse_vs_stored": diff, **run.record()}
    atomic_write_json(out_json, res)
    os.remove(part)
    print({k: v for k, v in res.items() if k != "top1_512"}, flush=True)


if __name__ == "__main__":
    main()
