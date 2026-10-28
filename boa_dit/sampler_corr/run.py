"""Sampler-side correction on heig-8 W4 (results/temporal/sampler_corr/PROTOCOL.md).
Q-Drift (arXiv 2603.18095, Eq. 14-15): Euler in sigma coordinates with eps scaled by (1 + c_i), c_i = |dsigma_i|/(2 sigma_i) V_i,
V_i = E[Var(eps_hat - eps | eps_hat)] (scalar per step, isotropic joint-Gaussian, paired on FP-trajectory states). Deterministic
DDIM is exactly Euler in y = x/sqrt(abar), sigma = sqrt((1-abar)/abar), so the correction is the eps hook out[:, :4] *= (1 + c_i).
  check   : explicit Euler-sigma sampler reproduces DDIM (fp32), with and without the factor
  calib   : V_i on seeds 6000-6063 + conditional-residual temporal-covariance diagnostic on held-out 6064-6095
  latent  : heig-8 + Q-Drift endpoint latent rel. MSE on eval seeds 0-511 (fidelity, not the design objective)
  fid --arm base|qdrift : FID-2k (seeds 0-1999, eval_fid protocol)
Run: python -m boa_dit.sampler_corr.run check|calib|latent|fid --arm qdrift
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.eval_fid import REF_STATS, compute_ref, fid_from_stats, inception_stats
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import decode, labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks
from boa_dit.sample import CTX, ddim_sample
from boa_dit.weights_joint.common import step_outputs

STAGE, CAP_H = "sampler_corr", 2.0
OUT = os.path.join(RESULTS, "temporal", "sampler_corr")
CK = os.path.join(CKPT_DIR, "sampler_corr")
HEIG = "deflate_heig_r8_w4"
CARRY = os.path.join(CKPT_DIR, "temporal", "rebuild_quant", HEIG, "carry.pt")
CAL, HELD = list(range(6000, 6064)), list(range(6064, 6096))
T = 50


def spent():
    with open(os.path.join(RESULTS, "gpu_hours.csv")) as f:
        return sum(float(r[5]) for r in csv.reader(f) if r and r[0] == STAGE)


def guard(est, what):
    s = spent()
    print(f"  budget: {s:.3f} h + {est:.3f} h vs cap {CAP_H} ({what})", flush=True)
    if s + est > CAP_H:
        print(f"  UNRUN: {what}", flush=True)
        sys.exit(5)


def sigmas(diffusion):
    """sigma_i at sampling step i (i = 0 noisiest) and sigma_{i+1} (0 at the end), from the respaced schedule."""
    ab, abp = np.asarray(diffusion.alphas_cumprod, dtype=np.float64), np.asarray(diffusion.alphas_cumprod_prev, dtype=np.float64)
    ts = list(range(diffusion.num_timesteps))[::-1]
    s = np.array([math.sqrt((1 - ab[t]) / ab[t]) for t in ts])
    sn = np.array([math.sqrt((1 - abp[t]) / abp[t]) if abp[t] < 1 else 0.0 for t in ts])
    return s, sn, ts, ab, abp


def heig(dev):
    m = load_dit(dev)
    load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    hs = install(m, torch.load(CARRY, map_location="cpu"))
    return m, hs


def qdrift_hook(c):
    def hook(i, xx, out):
        out = out.clone()
        out[:, :4] *= (1.0 + c[i])
        return out
    return hook


@torch.no_grad()
def check(dev):
    """Euler in sigma-coordinates (y = x/sqrt(abar)) vs DDIM, fp32, 4 seeds, with and without a (1 + c) factor."""
    run = Run(STAGE, "check_ddim_euler", dev, {}); run.__enter__()
    m, d = load_dit(dev), make_diffusion(50)
    s, sn, ts, ab, abp = sigmas(d)
    seeds = CAL[:4]
    labels, noise = labels_for_seeds(seeds), noise_for_seeds(seeds, dev)
    res = {}
    for lab, c in (("plain", np.zeros(T)), ("scaled", np.full(T, 0.01))):
        ref = ddim_sample(m, d, noise, labels, 1.5, eps_hook=qdrift_hook(c), amp=False)
        x = noise.clone()
        for i in range(T):
            eps = step_outputs(m, d, x, labels, i, amp=False)[:, :4] * (1 + c[i])
            y = x / math.sqrt(ab[ts[i]])
            y = y + (sn[i] - s[i]) * eps
            x = y * math.sqrt(abp[ts[i]])
        res[lab] = float((x - ref).norm() / ref.norm())
    res["pass"] = all(v < 1e-5 for v in res.values())
    run.__exit__(None, None, None)
    atomic_write_json(os.path.join(OUT, "check_ddim_euler.json"), res)
    print(res, flush=True)
    if not res["pass"]:
        raise SystemExit(6)


@torch.no_grad()
def paired(seeds, dev, d, fp, q):
    """FP trajectory of seeds (batches of 32); at each step the FP and heig-8 guided eps (4 ch) on the same FP state."""
    E, Q = torch.zeros(len(seeds), T, 4, 32, 32), torch.zeros(len(seeds), T, 4, 32, 32)
    for b in range(0, len(seeds), 32):
        s = seeds[b:b + 32]
        lab = labels_for_seeds(s)
        _, st = ddim_sample(fp, d, noise_for_seeds(s, dev), lab, 1.5, record_states=True)
        for i in range(T):
            E[b:b + 32, i] = step_outputs(fp, d, st[i], lab, i)[:, :4].cpu()
            Q[b:b + 32, i] = step_outputs(q, d, st[i], lab, i)[:, :4].cpu()
    return E, Q


def band_split(X, masks):
    """X [..., 32, 32] -> list of band components (real), one per mask."""
    F = torch.fft.fft2(X.double())
    return [torch.fft.ifft2(F * m.double()).real for m in masks]


@torch.no_grad()
def calib(dev):
    guard(0.10, "calibration + diagnostic")
    os.makedirs(OUT, exist_ok=True)
    with Run(STAGE, "calib_diag", dev, {"cal": [CAL[0], CAL[-1]], "held": [HELD[0], HELD[-1]]}) as run:
        d = make_diffusion(50)
        fp = load_dit(dev)
        q, _ = heig(dev)
        E, Q = paired(CAL, dev, d, fp, q)
        Eh, Qh = paired(HELD, dev, d, fp, q)
        s, sn, _, _, _ = sigmas(d)
        V, beta, mD, mQ = [], [], [], []
        for i in range(T):
            D, Qi = (Q[:, i] - E[:, i]).double().flatten(), Q[:, i].double().flatten()
            md, mq = D.mean(), Qi.mean()
            vD, vQ, cov = ((D - md) ** 2).mean(), ((Qi - mq) ** 2).mean(), ((D - md) * (Qi - mq)).mean()
            V.append(float(vD - cov ** 2 / vQ)); beta.append(float(cov / vQ)); mD.append(float(md)); mQ.append(float(mq))
        c = [abs(sn[i] - s[i]) / (2 * s[i]) * V[i] for i in range(T)]
        cal = {"V": V, "c": c, "beta": beta, "mean_delta": mD, "mean_qhat": mQ, "sigma": s.tolist(), "sigma_next": sn.tolist(),
               "unconditional_var": [float((Q[:, i] - E[:, i]).double().var()) for i in range(T)]}
        atomic_write_json(os.path.join(OUT, "qdrift_calibration.json"), cal)
        # diagnostic on held-out seeds: raw Delta and conditional residual xi
        D = (Qh - Eh).double()
        Xi = torch.stack([D[:, i] - (mD[i] + beta[i] * (Qh[:, i].double() - mQ[i])) for i in range(T)], 1)   # [32, T, 4, 32, 32]
        masks = band_masks()
        diag = {}
        for lab, X in (("raw", D), ("residual", Xi)):
            comps = band_split(X, masks)
            comps.append(X.mean((-1, -2), keepdim=True).expand_as(X))                                     # DC (spatial mean)
            names = ["band1", "band2", "band3", "band4", "dc"]
            tot = sum(float(cp.pow(2).sum()) for cp in comps[:4])
            out = {}
            for nm, cp in zip(names, comps):
                lag = {}
                for L in (1, 2, 5, 10):
                    a, b = cp[:, :-L].flatten(2), cp[:, L:].flatten(2)                                   # [32, T-L, n]
                    num = (a * b).sum((0, 2)); den = torch.sqrt(a.pow(2).sum((0, 2)) * b.pow(2).sum((0, 2)))
                    lag[str(L)] = float((num / den).mean())
                S = cp.sum(1)
                coh = float(S.pow(2).sum() / (T * cp.pow(2).sum()))
                out[nm] = {"lag_corr": lag, "coherent_fraction": coh, "energy_share": float(cp.pow(2).sum()) / tot if nm != "dc" else None}
            diag[lab] = out
        diag["energy_ratio_residual_over_raw"] = float(Xi.pow(2).sum() / D.pow(2).sum())
        diag["read_temporal_dependence_remains"] = bool(diag["residual"]["band1"]["lag_corr"]["1"] >= 0.3 and diag["residual"]["band1"]["coherent_fraction"] >= 3 / T)
        diag.update(run.record())
    atomic_write_json(os.path.join(OUT, "residual_temporal_diag.json"), diag)
    print("c_i (first, mid, last):", [round(c[i], 5) for i in (0, 25, 49)], "| V (first, mid, last):", [round(V[i], 5) for i in (0, 25, 49)], flush=True)
    for lab in ("raw", "residual"):
        print(lab, {nm: (round(v["lag_corr"]["1"], 3), round(v["coherent_fraction"], 3)) for nm, v in diag[lab].items()}, flush=True)
    print("residual/raw energy", round(diag["energy_ratio_residual_over_raw"], 3), "| temporal dependence remains:", diag["read_temporal_dependence_remains"], flush=True)


@torch.no_grad()
def sample_many(m, d, seeds, dev, hook, part):
    done = torch.load(part) if os.path.exists(part) else {}
    for b in range(0, len(seeds), 32):
        if b in done:
            continue
        s = seeds[b:b + 32]
        done[b] = ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5, eps_hook=hook).cpu()
        atomic_torch_save(done, part)
    return torch.cat([done[b] for b in sorted(done)])


@torch.no_grad()
def latent(dev):
    guard(0.12, "Q-Drift latent eval (512)")
    c = json.load(open(os.path.join(OUT, "qdrift_calibration.json")))["c"]
    with Run(STAGE, "latent_qdrift", dev, {}) as run:
        xf, _ = fp_reference(dev)
        m, _ = heig(dev)
        part = os.path.join(CK, "latent_qdrift.partial")
        os.makedirs(CK, exist_ok=True)
        x = sample_many(m, make_diffusion(50), list(range(512)), dev, qdrift_hook(c), part)
        rel, _, _ = per_image(x, xf)
        stored = json.load(open(os.path.join(RESULTS, "gate1", "eval", f"latent_{HEIG}.json")))["rel_mse_per_image"]
        from boa_dit.weights_joint.common import boot_reduction
        res = {"rel_mse_mean": float(rel.mean()), "rel_mse_per_image": rel.tolist(), "baseline_heig8_stored": float(np.mean(stored)),
               "vs_baseline": boot_reduction(rel.tolist(), stored), **run.record()}
    os.remove(part)
    atomic_write_json(os.path.join(OUT, "latent_qdrift.json"), res)
    print(f"Q-Drift latent {res['rel_mse_mean']:.5f} vs heig-8 {res['baseline_heig8_stored']:.5f}: {res['vs_baseline']}", flush=True)


@torch.no_grad()
def fid(arm, dev, n=2000):
    guard(0.40, f"FID-2k {arm}")
    if not os.path.exists(REF_STATS):
        with Run(STAGE, "fid_ref_stats", dev, {}):
            compute_ref(dev)
    c = json.load(open(os.path.join(OUT, "qdrift_calibration.json")))["c"] if arm == "qdrift" else None
    with Run(STAGE, f"fid{n}_{arm}", dev, {"arm": arm}) as run:
        m, _ = heig(dev)
        os.makedirs(CK, exist_ok=True)
        part = os.path.join(CK, f"fid{n}_{arm}.partial")
        x = sample_many(m, make_diffusion(50), list(range(n)), dev, qdrift_hook(c) if c else None, part)
        del m
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev)).permute(0, 2, 3, 1).numpy()
        del vae
        mu, sig = inception_stats(imgs, dev)
        ref = np.load(REF_STATS)
        res = {"arm": arm, "n": n, "fid": fid_from_stats(mu, sig, ref["mu"], ref["sigma"]), **run.record()}
    os.remove(part)
    atomic_write_json(os.path.join(OUT, f"fid{n}_{arm}.json"), res)
    print(f"FID-{n} {arm}: {res['fid']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "calib", "latent", "fid"])
    ap.add_argument("--arm", default="base", choices=["base", "qdrift"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    {"check": lambda: check(dev), "calib": lambda: calib(dev), "latent": lambda: latent(dev), "fid": lambda: fid(a.arm, dev)}[a.cmd]()


if __name__ == "__main__":
    main()
