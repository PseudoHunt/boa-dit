"""D2(b) band oracle on DiT-XL/2 (heig-8, eval seeds 0-63, the eval sampler): at each step the quantized model's guided
prediction (4 eps channels) keeps its out-of-band part and takes FP's in-band part (FP evaluated on the same current x).
recovery = 1 - relMSE(oracle)/relMSE(heig-8), both vs the FP latents, paired bootstrap CI over the 64 seeds.
Variants: band 1 at all steps; bands 1+2 at all steps; band 1 at steps 0-24 only. The ceiling for any objective that
targets low-frequency layer error. Run: python -m boa_dit.objective.oracle_band --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.eval_latent_mse import per_image
from boa_dit.model import NULL_CLASS, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks
from boa_dit.objective.proxy import load_arm
from boa_dit.sample import ddim_sample

SEEDS = list(range(64))
VARIANTS = {"sanity_all_bands": ((0, 1, 2, 3), 50), "band1_all": ((0,), 50), "band12_all": ((0, 1), 50), "band1_first_half": ((0,), 25)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    masks = band_masks()
    xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:len(SEEDS)]
    stored = json.load(open(os.path.join(RESULTS, "gate1", "eval", "latent_deflate_heig_r8_w4.json")))["rel_mse_per_image"][:len(SEEDS)]
    out = {"seeds": "0-63", "heig8_rel_mse_seeds0_63": float(np.mean(stored)), "variants": {}}
    with Run("objective", "d2b_band_oracle", dev, vars(a)) as run, torch.no_grad():
        fp = load_dit(dev)
        q, _ = load_arm("heig-8", dev)
        diffusion = make_diffusion(50)
        for name, (bands, last_step) in VARIANTS.items():
            M = torch.zeros(32, 32, dtype=torch.bool)
            for b in bands:
                M |= masks[b]
            M = M.to(dev)
            xs = []
            for b0 in range(0, len(SEEDS), 32):
                sd = SEEDS[b0:b0 + 32]
                lab = torch.as_tensor(labels_for_seeds(sd), device=dev)
                n = len(sd)
                y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])

                def hook(i, xx, res):
                    if i >= last_step:
                        return res
                    ts = torch.full((n,), list(range(diffusion.num_timesteps))[::-1][i], device=dev, dtype=torch.long)
                    tmap = diffusion._wrap_model(lambda x_, t_: t_)(xx, ts)              # respaced -> original timesteps
                    with torch.autocast("cuda", dtype=torch.float16):
                        o = fp(torch.cat([xx, xx]), torch.cat([tmap, tmap]), y).float()
                    eps, rest = o[:, :3], o[:, 3:]
                    rf = torch.cat([eps[n:] + 1.5 * (eps[:n] - eps[n:]), rest[:n]], 1)[:, :4]
                    Fq, Ff = torch.fft.fft2(res[:, :4].float()), torch.fft.fft2(rf)
                    mixed = torch.fft.ifft2(torch.where(M, Ff, Fq)).real
                    res = res.clone()
                    res[:, :4] = mixed
                    return res
                xs.append(ddim_sample(q, diffusion, noise_for_seeds(sd, dev), labels_for_seeds(sd), 1.5, eps_hook=hook).cpu())
            rel = per_image(torch.cat(xs), xf)[0].numpy()
            base = np.array(stored)
            rng = np.random.default_rng(0)
            bs = [1 - rel[i].mean() / base[i].mean() for i in (rng.integers(0, len(rel), len(rel)) for _ in range(2000))]
            rec = float(1 - rel.mean() / base.mean())
            out["variants"][name] = {"rel_mse": float(rel.mean()), "recovery": rec, "ci": np.percentile(bs, [2.5, 97.5]).tolist(),
                                     "bands": [b + 1 for b in bands], "steps": f"0-{last_step - 1}", "rel_mse_per_image": rel.tolist()}
            print(f"{name}: rel MSE {rel.mean():.5f} (heig-8 {base.mean():.5f}) -> recovery {100 * rec:.1f}% "
                  f"[{100 * np.percentile(bs, 2.5):.1f}, {100 * np.percentile(bs, 97.5):.1f}]", flush=True)
        out["read_band1_ge_40pct"] = out["variants"]["band1_all"]["recovery"] >= 0.40
        out.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "objective", "d2b_oracle.json"), out)


if __name__ == "__main__":
    main()
