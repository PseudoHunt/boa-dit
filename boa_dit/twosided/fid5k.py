"""Gate 1 §5: FID-5k (seeds 0-4999, cfg 1.5, DDIM-50, bs 32, same protocol as eval_fid) + ResNet-50 label accuracy.

  --arm fp             : the FP model (reference for the W4 numbers)
  --arm uniform        : ckpt_uniform_w4_K1_lam1_brcfg
  --arm static-8       : uniform W4 + Gate A static correction, rank 8 (lowrank.Corrector)
  --arm <name>         : ckpt_<name>/model.pt + its carry (deflate_dc_r8_w4, ts_* engine arms)
Run: python -m boa_dit.twosided.fid5k --arm deflate_dc_r8_w4 --device cuda:0
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.attn_share.gate2 import fp_top1, label_acc, resnet_top1
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_fid import REF_STATS, fid_from_stats, inception_stats
from boa_dit.model import decode, load_dit, load_quant_state, load_vae, make_diffusion
from boa_dit.sample import sample_seeds

UNIFORM = "uniform_w4_K1_lam1_brcfg"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    out = os.path.join(RESULTS, "gate1", "eval", f"fid{a.n}_{a.arm}.json")
    if os.path.exists(out):
        return
    with Run("gate1", f"fid{a.n}_{a.arm}", dev, vars(a)) as run:
        model, diffusion = load_dit(dev), make_diffusion(50)
        corr = None
        if a.arm == "fp":
            pass
        elif a.arm in ("uniform", "static-8"):
            load_quant_state(model, os.path.join(CKPT_DIR, f"ckpt_{UNIFORM}", "model.pt"))
            if a.arm == "static-8":
                from boa_dit import lowrank
                corr = lowrank.Corrector(model, lowrank.build(4, dev, ["static"]), "static", 8, dev)
        else:
            load_quant_state(model, os.path.join(CKPT_DIR, f"ckpt_{a.arm}", "model.pt"))
            from boa_dit.deflate.carry import install_from
            install_from(model, a.arm)
        part = os.path.join(CKPT_DIR, "gate1_fid", f"fid{a.n}_{a.arm}.partial")
        os.makedirs(os.path.dirname(part), exist_ok=True)
        x = sample_seeds(model, diffusion, list(range(a.n)), 1.5, 32, partial=part, resume=True, log=False)
        if corr is not None:
            corr.remove()
        del model
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev))
        del vae
        atomic_torch_save(imgs[:16].clone(), os.path.join(CKPT_DIR, "gate1_fid", f"grid_{a.arm}.pt"))
        mu, sig = inception_stats(imgs.permute(0, 2, 3, 1).numpy(), dev)
        ref = np.load(REF_STATS)
        t1 = resnet_top1(imgs, dev)
        res = {"arm": a.arm, "n": a.n, "fid": fid_from_stats(mu, sig, ref["mu"], ref["sigma"]),
               "label_acc": label_acc(t1), "label_acc_512": label_acc(t1[:512]),
               "class_changed_vs_fp_512": float((t1[:512] != fp_top1(dev)[:len(t1[:512])]).float().mean()),
               "top1": t1.tolist(), **run.record()}
    atomic_write_json(out, res)
    os.remove(part)
    print({k: v for k, v in res.items() if k not in ("top1", "config")}, flush=True)


if __name__ == "__main__":
    main()
