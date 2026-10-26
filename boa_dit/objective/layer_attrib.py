"""Band-1 attribution on heig-8 (rule_addendum_alloc.md): for each of the 112 (block, layer) pairs, restore that layer to
FP (its carry removed) and measure (a) the band-1 guided-error energy at steps 20 and 25 of the FP trajectory and (b)
latent rel. MSE, both on seeds 1100-1163: disjoint from the eval seeds 0-511, calibration 1000-1031 and held-out
1032-1047 (an earlier run on eval seeds 0-63 was voided: `layer_attrib_VOID_eval_seeds0_63.json`). The FP trajectory, FP
final latents and the heig-8 baseline for these seeds are computed here. One resident model; weights and the carry
hook are swapped per arm.
Run: python -m boa_dit.objective.layer_attrib --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.carry import attach, carry_path
from boa_dit.eval_latent_mse import per_image
from boa_dit.model import LINEARS, block_linear, labels_for_seeds, load_dit, load_quant_state, make_diffusion
from boa_dit.objective.d2 import band_masks
from boa_dit.objective.proxy import TRAJ, step_outputs
from boa_dit.sample import sample_seeds

BASE = "deflate_heig_r8_w4"
SEEDS = list(range(1100, 1164))
B1_STEPS = (20, 25)
OUT = os.path.join(RESULTS, "objective", "layer_attrib.json")


@torch.no_grad()
def band1_energy(model, diffusion, traj, ref, mask, dev):
    labels = labels_for_seeds(SEEDS)
    tot = 0.0
    for s in B1_STEPS:
        for b in range(0, len(SEEDS), 32):
            q = step_outputs(model, diffusion, traj["x"][s][b:b + 32].to(dev), labels[b:b + 32], s)[0][:, :4]
            F = torch.fft.fft2((q - ref[s][b:b + 32].to(dev)).double())
            tot += float(F.abs().pow(2)[..., mask].sum())
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    res = json.load(open(OUT)) if os.path.exists(OUT) else {"pairs": {}}
    with Run("objective", "layer_attrib", dev, vars(a)) as run, torch.no_grad():
        diffusion = make_diffusion(50)
        mask = band_masks()[0].to(dev)
        fp = load_dit(dev)
        labels = labels_for_seeds(SEEDS)
        from boa_dit.model import noise_for_seeds
        from boa_dit.sample import ddim_sample
        traj, x0s = {"x": {s: [] for s in B1_STEPS}}, []
        for b in range(0, len(SEEDS), 32):
            x0, st = ddim_sample(fp, diffusion, noise_for_seeds(SEEDS[b:b + 32], dev), labels[b:b + 32], 1.5, record_states=True)
            for s in B1_STEPS:
                traj["x"][s].append(st[s].cpu())
            x0s.append(x0.cpu())
        traj["x"] = {s: torch.cat(v) for s, v in traj["x"].items()}
        xf = torch.cat(x0s)
        ref = {s: torch.cat([step_outputs(fp, diffusion, traj["x"][s][b:b + 32].to(dev), labels[b:b + 32], s)[0][:, :4].cpu()
                             for b in range(0, len(SEEDS), 32)]) for s in B1_STEPS}
        m = load_dit(dev)
        load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{BASE}", "model.pt"))
        carry = torch.load(carry_path(BASE), map_location="cpu")
        handles = {}
        for key, c in carry.items():
            l, ln = key.split(".", 1)
            handles[key] = attach(block_linear(m.blocks[int(l[1:])], ln), c["V"], c["delta"])
        base_rel = per_image(sample_seeds(m, diffusion, SEEDS, 1.5, 32, log=False), xf)[0].numpy()
        if "base" not in res:
            res["base"] = {"band1_energy": band1_energy(m, diffusion, traj, ref, mask, dev), "seeds": "1100-1163",
                           "rel_mse": float(base_rel.mean()), "rel_mse_per_image": base_rel.tolist()}
        E0 = res["base"]["band1_energy"]
        for l in range(28):
            for ln in LINEARS:
                key = f"b{l}.{ln}"
                if key in res["pairs"]:
                    continue
                lin = block_linear(m.blocks[l], ln)
                Wq = lin.weight.data.clone()
                lin.weight.data.copy_(block_linear(fp.blocks[l], ln).weight.data)
                handles[key].remove()
                try:
                    e1 = band1_energy(m, diffusion, traj, ref, mask, dev)
                    x = sample_seeds(m, diffusion, SEEDS, 1.5, 32, log=False)
                finally:
                    lin.weight.data.copy_(Wq)
                    c = carry[key]
                    handles[key] = attach(lin, c["V"], c["delta"])
                rel = per_image(x, xf)[0].numpy()
                rng = np.random.default_rng(0)
                bs = [1 - rel[i].mean() / base_rel[i].mean() for i in (rng.integers(0, len(rel), len(rel)) for _ in range(2000))]
                res["pairs"][key] = {"band1_contribution": 1 - e1 / E0, "latent_share": float(1 - rel.mean() / base_rel.mean()),
                                     "latent_ci": np.percentile(bs, [2.5, 97.5]).tolist(), "rel_mse": float(rel.mean())}
                atomic_write_json(OUT, res)
                print(f"{key}: band-1 contribution {res['pairs'][key]['band1_contribution']:+.3f}, latent share "
                      f"{res['pairs'][key]['latent_share']:+.3f}", flush=True)
        c = {k: v["band1_contribution"] for k, v in res["pairs"].items()}
        pos = sum(max(v, 0) for v in c.values())
        top = sorted(c, key=lambda k: -c[k])[:10]
        res["top10"] = top
        res["concentration_top10"] = sum(c[k] for k in top) / pos
        res["flat_stop"] = res["concentration_top10"] < 0.25
        lat = {k: v["latent_share"] for k, v in res["pairs"].items()}
        res["spearman_band1_vs_latent"] = float(np.corrcoef(np.argsort(np.argsort(list(c.values()))),
                                                           np.argsort(np.argsort([lat[k] for k in c])))[0, 1])
        res.update(run.record())
    atomic_write_json(OUT, res)
    print(f"concentration (top-10 share of band-1 contribution) {res['concentration_top10']:.3f}; flat stop: {res['flat_stop']}; "
          f"Spearman(band-1 contribution, latent share) {res['spearman_band1_vs_latent']:.2f}", flush=True)


if __name__ == "__main__":
    main()
