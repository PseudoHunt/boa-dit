"""G3: timestep-bucket sensitivity of W3 quantization, by identity (latent MSE) and by quality (FID-2k). See results/tsens/G3.md.

  traj  : cache the FP trajectory for seeds 0-511 (x before each step + final), fp16 -> ckpt/tsens/fp_traj.pt
  arm X : P0..P4 (W3 in bucket b, FP elsewhere), I0 / I4 (W3 in B0 / B4, W4 elsewhere): 2,000 images (seeds 0-1999,
          FID-2k), first 512 with per-step states (deviation vs the FP trajectory), labels, late-anchored restart at 25
  noise : FP FID-2k on seeds 2000-3999
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

from boa_dit.attn_share.gate2 import fp_top1, label_acc, resnet_top1
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_fid import REF_STATS, fid_from_stats, inception_stats
from boa_dit.model import decode, labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample

OUT = os.path.join(RESULTS, "tsens")
CK = os.path.join(CKPT_DIR, "tsens")
TRAJ = os.path.join(CK, "fp_traj.pt")
BUCKET = 10
ARMS = {f"P{b}": ("fp", b) for b in range(5)}
ARMS.update({"I0": ("w4", 0), "I4": ("w4", 4)})


class Router(nn.Module):
    """Sends each sampler step (CTX.step) to models[route[step]]."""

    def __init__(self, models, route):
        super().__init__()
        self.models = nn.ModuleDict(models)
        self.route = route

    def forward(self, *args, **kw):
        return self.models[self.route[CTX.step]](*args, **kw)


def models(dev, need):
    out = {}
    for k in need:
        m = load_dit(dev)
        if k != "fp":
            load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_uniform_{k}_K1_lam1_brcfg", "model.pt"))
        out[k] = m
    return out


def route_for(arm):
    base, b = ARMS[arm]
    return [("w3" if s // BUCKET == b else base) for s in range(50)]


def rel(a, b):
    return (a.float() - b.float()).pow(2).flatten(1).sum(1) / b.float().pow(2).flatten(1).sum(1)


@torch.no_grad()
def traj(dev):
    if os.path.exists(TRAJ):
        return
    os.makedirs(CK, exist_ok=True)
    m, diff = load_dit(dev), make_diffusion(50)
    seeds = list(range(512))
    with Run("tsens", "fp_traj", dev, {}):
        states, finals = [], []
        for i in range(0, 512, 32):
            x, st = ddim_sample(m, diff, noise_for_seeds(seeds[i:i + 32], dev), labels_for_seeds(seeds[i:i + 32]), 1.5,
                                record_states=True)
            states.append(torch.stack([st[s].half().cpu() for s in range(50)]))
            finals.append(x.cpu())
        atomic_torch_save({"states": torch.cat(states, 1), "final": torch.cat(finals)}, TRAJ)   # [50, 512, 4, 32, 32]


def score_fid(x, dev):
    vae = load_vae(dev)
    imgs = decode(vae, x.to(dev))
    del vae
    mu, sig = inception_stats(imgs.permute(0, 2, 3, 1).numpy(), dev)
    ref = np.load(REF_STATS)
    return fid_from_stats(mu, sig, ref["mu"], ref["sigma"]), imgs


@torch.no_grad()
def arm(name, dev):
    out_json = os.path.join(OUT, "eval", f"{name}.json")
    if os.path.exists(out_json):
        return
    T = torch.load(TRAJ, map_location="cpu")
    route = route_for(name)
    diff = make_diffusion(50)
    with Run("tsens", f"arm_{name}", dev, {"route": route}) as run:
        R = Router(models(dev, sorted(set(route) | {"fp"})), route)
        part = os.path.join(CK, f"{name}.partial")
        done = torch.load(part, map_location="cpu") if os.path.exists(part) else {}
        seeds = list(range(2000))
        for i in range(0, 2000, 32):
            if i in done:
                continue
            nz, lb = noise_for_seeds(seeds[i:i + 32], dev), labels_for_seeds(seeds[i:i + 32])
            if i < 512:
                x, st = ddim_sample(R, diff, nz, lb, 1.5, record_states=True)
                dev_steps = torch.stack([rel(st[s].cpu(), T["states"][s, i:i + 32]) for s in range(50)])   # [50, 32]
                # late-anchored: arm and FP both restart from the cached FP latent at step 25
                x25 = T["states"][25, i:i + 32].to(dev).float()
                xa = ddim_sample(R, diff, nz, lb, 1.5, start_step=25, x_init=x25)
                R.route = ["fp"] * 50
                xf = ddim_sample(R, diff, nz, lb, 1.5, start_step=25, x_init=x25)
                R.route = route
                done[i] = {"x": x.cpu(), "dev": dev_steps, "late": rel(xa.cpu(), xf.cpu())}
            else:
                done[i] = {"x": ddim_sample(R, diff, nz, lb, 1.5).cpu()}
            atomic_torch_save(done, part)
        del R
        torch.cuda.empty_cache()
        x = torch.cat([done[i]["x"] for i in range(0, 2000, 32)])
        devs = torch.cat([done[i]["dev"] for i in range(0, 512, 32)], 1)                 # [50, 512]
        late = torch.cat([done[i]["late"] for i in range(0, 512, 32)])
        idr = rel(x[:512], T["final"])
        fid, imgs = score_fid(x, dev)
        t1 = resnet_top1(imgs[:512], dev)
        res = {"arm": name, "route": route, "identity_rel_mse": float(idr.mean()), "identity_rel_mse_median": float(idr.median()),
               "fid2k": fid, "label_acc_512": label_acc(t1), "class_changed_frac_512": float((t1 != fp_top1(dev)).float().mean()),
               "late_anchored_rel_mse": float(late.mean()), "per_step_deviation_mean": devs.mean(1).tolist(),
               "per_step_deviation_median": devs.median(1).values.tolist(), "rel_mse_per_image": idr.tolist(), **run.record()}
    atomic_write_json(out_json, res)
    os.remove(part)
    print({k: v for k, v in res.items() if k not in ("rel_mse_per_image", "per_step_deviation_mean", "per_step_deviation_median", "route")}, flush=True)


@torch.no_grad()
def noise(dev):
    out_json = os.path.join(OUT, "eval", "noise_fp_s2000.json")
    if os.path.exists(out_json):
        return
    from boa_dit.sample import sample_seeds
    with Run("tsens", "noise_fp_s2000", dev, {}) as run:
        m = load_dit(dev)
        x = sample_seeds(m, make_diffusion(50), list(range(2000, 4000)), 1.5, 32,
                         partial=os.path.join(CK, "noise.partial"), resume=True, log=False)
        del m
        fid, _ = score_fid(x, dev)
        res = {"seeds": "2000-3999", "fid2k": fid, **run.record()}
    atomic_write_json(out_json, res)
    os.remove(os.path.join(CK, "noise.partial"))
    print(res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", help="traj | noise | arm name (P0..P4, I0, I4)")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    if a.what == "traj":
        traj(a.device)
    elif a.what == "noise":
        noise(a.device)
    else:
        traj(a.device)
        arm(a.what, a.device)


if __name__ == "__main__":
    main()
