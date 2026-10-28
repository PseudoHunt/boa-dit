"""Weight-space cancellation oracle and joint two-table GPTQ pilot (results/temporal/weights_joint/PROTOCOL.md):
paths, seed sets, the stage-`weights_joint` budget guard (2.0 A30-h hard stop), FP references and diagnostics on the
design seeds."""
import csv
import math
import os
import sys

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import NULL_CLASS, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks, band_spectrum
from boa_dit.sample import CTX, ddim_sample
from boa_dit.temporal.common import ci_split

STAGE = "weights_joint"
CAP_H = 2.0
OUT = os.path.join(RESULTS, "temporal", "weights_joint")
CK = os.path.join(CKPT_DIR, "weights_joint")
DESIGN = list(range(4000, 4032))
ORACLE = DESIGN[:16]
RESERVED = list(range(4100, 4228))                                            # untouched, not used in this pilot
HEIG = "deflate_heig_r8_w4"
HEIG_LAYERS = os.path.join(CKPT_DIR, "temporal", "heig8_layers.pt")


def spent():
    with open(os.path.join(RESULTS, "gpu_hours.csv")) as f:
        return sum(float(r[5]) for r in csv.reader(f) if r and r[0] == STAGE)


def guard(est_h, what):
    """Launch only if spent + est fits the 2.0 h pilot cap (exit 5 otherwise; the caller marks the arm unrun)."""
    s = spent()
    print(f"  budget: {s:.3f} h spent + {est_h:.3f} h est vs cap {CAP_H} ({what})", flush=True)
    if s + est_h > CAP_H:
        print(f"  UNRUN: {what} does not fit the pilot cap", flush=True)
        sys.exit(5)


@torch.no_grad()
def step_outputs(model, diffusion, x, labels, i, amp=True, cfg=1.5):
    """One DDIM step's guided model output at index i from x (the sampler's CFG path); amp=False runs fp32."""
    cap = {}
    n = len(x)
    lab = torch.as_tensor(labels, device=x.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    ti = list(range(diffusion.num_timesteps))[::-1][i]
    CTX.step, CTX.n = i, n

    def model_fn(xx, ts):
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            out = model(torch.cat([xx, xx]), torch.cat([ts, ts]), y).float()
        eps, rest = out[:, :3], out[:, 3:]
        c, u = eps[:n], eps[n:]
        res = torch.cat([u + cfg * (c - u), rest[:n]], dim=1)
        cap["res"] = res
        return res
    t = torch.full((n,), ti, device=x.device, dtype=torch.long)
    diffusion.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)
    CTX.step = None
    return cap["res"]


@torch.no_grad()
def fp_traj(dev, seeds, amp, tag):
    """FP trajectory states (all 50 steps), guided reference outputs (4 eps channels) and endpoint, cached."""
    path = os.path.join(CK, f"fp_traj_{tag}.pt")
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    m, d = load_dit(dev), make_diffusion(50)
    labels = labels_for_seeds(seeds)
    x0, st = ddim_sample(m, d, noise_for_seeds(seeds, dev), labels, 1.5, record_states=True, amp=amp)
    ref = {i: step_outputs(m, d, st[i], labels, i, amp)[:, :4].cpu() for i in range(50)}
    out = {"x": {i: s.cpu() for i, s in st.items()}, "x0": x0.cpu(), "ref": ref, "seeds": seeds, "amp": amp}
    atomic_torch_save(out, path)
    del m
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def diagnostics(m, d, dev, tr, amp=True):
    """Per-step prediction-error energy, coherent C / incoherent I / C/I, band-1 share (steps 20-25) on traj tr."""
    labels = labels_for_seeds(tr["seeds"])
    E = torch.stack([step_outputs(m, d, tr["x"][i].to(dev), labels, i, amp)[:, :4].cpu() - tr["ref"][i] for i in range(50)], 1)
    spec = np.mean([band_spectrum(E[:, i], band_masks()) for i in range(20, 26)], axis=0).tolist()
    c = ci_split(E)
    return {"per_step_err": E.double().pow(2).flatten(2).sum(2).mean(0).tolist(), **c,
            "C_abs_per_seed": c["C"] / len(tr["seeds"]), "band_spectrum_steps20_25": spec}


def rel_mse(x, xf):
    return ((x - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1)).double()


def boot_reduction(a, b, n=2000):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(0)
    rs = [1 - a[i].mean() / b[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return {"reduction": float(1 - a.mean() / b.mean()), "ci": np.percentile(rs, [2.5, 97.5]).tolist(), "n": len(a)}
