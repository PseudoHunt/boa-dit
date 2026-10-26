"""D2 (rule.md): sampler frequency response by step. FP model, eval seeds 0-31 from the FP trajectory's x_t (steps
0,5,..,45). At step t, a random-phase error confined to one of 4 equal-area radial DFT bands (or white) is added to the
guided prediction (4 eps channels) with energy = heig-8's mean ||delta_g,t||^2 (D1 network data); FP continues to the
end. G(t, band) = mean_seed ||x0_inj - x0_FP||^2 / injected energy. S(t, band) = heig-8's delta_g,t band spectrum.
Also the Jacobian-free DDIM coefficient product. Run: python -m boa_dit.objective.d2 --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion
from boa_dit.objective.proxy import STEPS, TRAJ
from boa_dit.sample import ddim_sample

SEEDS = list(range(32))
OUT = os.path.join(RESULTS, "objective")


def band_masks(n=32):
    f = torch.fft.fftfreq(n)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    q = torch.quantile(r.flatten(), torch.tensor([0.25, 0.5, 0.75]))
    edges = [-1.0] + q.tolist() + [10.0]
    return [((r > edges[i]) & (r <= edges[i + 1])) for i in range(4)]


def band_noise(shape, mask, energy, g, dev):
    z = torch.randn(shape, generator=g, device="cpu")
    if mask is not None:
        Z = torch.fft.fft2(z)
        z = torch.fft.ifft2(Z * mask).real
    z = z * torch.sqrt(torch.tensor(energy) / z.pow(2).flatten(1).sum(1)).view(-1, 1, 1, 1)
    return z.to(dev)


def band_spectrum(e, masks):
    P = torch.fft.fft2(e.double()).abs().pow(2).sum((0, 1))                   # [32, 32] summed over samples, channels
    tot = float(P.sum())
    return [float(P[m].sum()) / tot for m in masks]


def jacobian_free(diffusion):
    ab = torch.tensor(diffusion.alphas_cumprod, dtype=torch.float64)
    abp = torch.tensor(diffusion.alphas_cumprod_prev, dtype=torch.float64)
    idx = list(range(diffusion.num_timesteps))[::-1]                          # sampling order: i -> respaced t
    out = {}
    for i in STEPS:
        t = idx[i]
        c_eps = torch.sqrt(1 - abp[t]) - torch.sqrt(abp[t]) * torch.sqrt(1 - ab[t]) / torch.sqrt(ab[t])
        prod = 1.0
        for j in range(i + 1, len(idx)):
            tj = idx[j]
            prod *= float(torch.sqrt(abp[tj] / ab[tj]))
        out[i] = float((c_eps * prod) ** 2)
    return out


def boot_ratio(a, b, n=2000):
    rng = np.random.default_rng(0)
    a, b = np.asarray(a), np.asarray(b)
    rs = [a[i].mean() / b[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return float(a.mean() / b.mean()), *np.percentile(rs, [2.5, 97.5]).tolist()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    d1 = json.load(open(os.path.join(OUT, "d1.json")))
    energy = {int(k): v for k, v in d1["network"]["dg4_energy_per_sample_by_step"].items()}
    masks = band_masks()
    traj = torch.load(TRAJ, map_location="cpu")
    fp_b = torch.load(os.path.join(CKPT_DIR, "objective", "branches_fp.pt"))["fp"]
    hq_b = torch.load(os.path.join(CKPT_DIR, "objective", "branches_heig8.pt"))["heig-8"]
    res = {"steps": list(STEPS), "bands": "4 equal-area radial bands of the 32x32 DFT, band 1 lowest", "G": {}, "S": {},
           "energy_per_sample": energy}
    with Run("objective", "d2_frequency", dev, vars(a)) as run:
        m, diffusion = load_dit(dev), make_diffusion(50)
        labels = labels_for_seeds(SEEDS)
        x0_ref = traj["x0"][:len(SEEDS)]
        xl = traj["x"][STEPS[-1]][:len(SEEDS)].to(dev)
        chk = ddim_sample(m, diffusion, xl, labels, 1.5, start_step=STEPS[-1], x_init=xl)
        res["continuation_bitexact"] = bool(torch.equal(chk.cpu(), x0_ref))
        for t in STEPS:
            res["S"][t] = band_spectrum(hq_b[t]["res"] - fp_b[t]["res"], masks)
            row = {}
            for bi, mask in enumerate(masks + [None]):
                g = torch.Generator().manual_seed(1000 * t + bi)
                noise = band_noise((len(SEEDS), 4, 32, 32), mask, energy[t], g, dev)

                def hook(i, xx, out, t=t, noise=noise):
                    if i == t:
                        out = out.clone()
                        out[:, :4] += noise
                    return out
                xt = traj["x"][t][:len(SEEDS)].to(dev)
                x0 = ddim_sample(m, diffusion, xt, labels, 1.5, eps_hook=hook, start_step=t, x_init=xt).cpu()
                gain = ((x0 - x0_ref).flatten(1).pow(2).sum(1) / energy[t]).double().numpy()
                row["white" if mask is None else f"band{bi + 1}"] = gain.tolist()
            res["G"][t] = row
            g1, g4 = row["band1"], row["band4"]
            r = boot_ratio(g1, g4)
            res.setdefault("ratio_b1_b4", {})[t] = r
            print(f"step {t}: G " + " ".join(f"{k}={np.mean(v):.3g}" for k, v in row.items()) + f"; b1/b4 {r[0]:.2f} [{r[1]:.2f}, {r[2]:.2f}]; "
                  f"S " + " ".join(f"{s:.2f}" for s in res["S"][t]), flush=True)
        res["jacobian_free_G"] = jacobian_free(diffusion)
        first5 = STEPS[:5]
        n_dir = sum(1 for t in first5 if (res["ratio_b1_b4"][t][1] > 1 and res["ratio_b1_b4"][t][0] >= 3) or
                    (res["ratio_b1_b4"][t][2] < 1 and res["ratio_b1_b4"][t][0] <= 1 / 3))
        res["read2_directional_steps"] = n_dir
        res["read2_directional"] = n_dir >= 2
        harm = {}
        for t in STEPS:
            h = [np.mean(res["G"][t][f"band{b + 1}"]) * res["S"][t][b] for b in range(4)]
            harm[t] = [x / sum(h) for x in h]
        res["harm"] = harm
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "d2.json"), res)
    print("read2 directional steps:", n_dir, "bitexact continuation:", res["continuation_bitexact"], flush=True)


if __name__ == "__main__":
    main()
