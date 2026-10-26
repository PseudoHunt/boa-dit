"""Per-arm error spectra (user follow-up): for each DiT arm, the band spectrum of its per-step guided error (4 eps
channels) on the FP trajectory (eval seeds 0-31 x steps 0,5,..,45), and the band-1 share at steps 20-25 as the secondary
metric next to latent MSE. Also the H_K filter gains in the 'least-amplified frequency = 1' convention.
Run: python -m boa_dit.objective.arm_spectra --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, ROOT, Run, atomic_write_json
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion
from boa_dit.objective.d2 import band_masks, band_spectrum
from boa_dit.objective.proxy import STEPS, TRAJ, load_arm, step_outputs

SEEDS = list(range(32))
EXTRA = {lab: (ck, True, 0, ("objective", "eval", f"latent_{ck}.json")) for lab, ck in (
    ("CFG-solve", "deflate_heig_r8_w4_cfgsolve"), ("CFG-full", "deflate_heig_r8_w4_cfgfull"),
    ("H_K sigma0.39", "deflate_heig_r8_w4_hk"), ("H_K pooled", "deflate_heig_r8_w4_hkpooled"),
    ("H_K bucket", "deflate_heig_r8_w4_hkbucket"), ("heig-latent-alloc", "deflate_heig_ralloclatent_w4"),
    ("heig-alloc (energy)", "deflate_heig_rallocenergy_w4"))}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    masks = band_masks()
    traj = torch.load(TRAJ, map_location="cpu")
    labels = labels_for_seeds(SEEDS)
    out = {"arms": {}, "filters": {}}
    for v in ("pooled", "bucket"):
        g = np.array(json.load(open(os.path.join(ROOT, "configs", f"objective_hk_{v}.json")))["gain"])
        out["filters"][v] = [{"min": 1.0, "max": float(gb.max() / gb.min())} for gb in g]
    with Run("objective", "arm_spectra", dev, vars(a)) as run:
        diffusion = make_diffusion(50)
        fp = load_dit(dev)
        ref = {s: step_outputs(fp, diffusion, traj["x"][s][:32].to(dev), labels, s)[0][:, :4].cpu() for s in STEPS}
        del fp
        arms = ["heig-8"] + [k for k in EXTRA if os.path.exists(os.path.join(RESULTS, *EXTRA[k][3]))]
        for name in arms:
            m, _ = load_arm(name, dev, EXTRA)
            S, E = {}, {}
            for s in STEPS:
                q = step_outputs(m, diffusion, traj["x"][s][:32].to(dev), labels, s)[0][:, :4].cpu()
                e = q - ref[s]
                S[s] = band_spectrum(e, masks)
                E[s] = float(e.double().pow(2).flatten(1).sum(1).mean())
            spec = (EXTRA.get(name) or (None, None, None, ("gate1", "eval", "latent_deflate_heig_r8_w4.json")))[3]
            lat = json.load(open(os.path.join(RESULTS, *spec)))["rel_mse_mean"]
            out["arms"][name] = {"S": S, "energy_per_sample": E, "latent_rel_mse": lat,
                                 "band1_share_steps20_25": float(np.mean([S[20][0], S[25][0]])),
                                 "band1_energy_steps20_25": float(np.mean([S[20][0] * E[20], S[25][0] * E[25]]))}
            print(f"{name}: latent {lat:.5f}; band-1 share at steps 20-25 {out['arms'][name]['band1_share_steps20_25']:.3f}; "
                  f"band-1 energy {out['arms'][name]['band1_energy_steps20_25']:.4g}", flush=True)
            del m
            torch.cuda.empty_cache()
        out.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "objective", "arm_spectra.json"), out)


if __name__ == "__main__":
    main()
