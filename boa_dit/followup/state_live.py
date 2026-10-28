"""Follow-up Experiment 1B (results/followup/PROTOCOL.md §1B): once-per-trajectory state correction on live heig-8 sampling.
  live_design  : seeds 8000-8031; arms P and M at τ ∈ {t30, t40, final}; placement frozen by the pre-registered rules
  live_confirm : seeds 8100-8163 (untouched); contemporaneous baseline, P at τ_P, M at τ_M; FP references
Coefficients: results/followup/e1/params_cal.pt (fitted on traces 7000-7063 by state_corr.screen).
"""
import json
import os
import time

import numpy as np
import torch

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.f4 import STAGE, guard
from boa_dit.followup.state_corr import CK, OUT, PLACES, apply, boot_ci, pooled_var, split
from boa_dit.model import decode, labels_for_seeds, load_dit, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample
from boa_dit.sampler_corr.run import heig

DESIGN, CONFIRM = list(range(8000, 8032)), list(range(8100, 8164))
ARMS = ["P", "M"]
STEP = {"t30": 30, "t40": 40}


def corrected(arm, params, x):
    return apply(arm, params, x.cpu()).float().to(x.device)


@torch.no_grad()
def sample_arms(seeds, plan, dev, bs=32):
    """plan: list of (arm, place). Returns FP endpoints, baseline endpoints, {(arm, place): endpoints}, correction seconds."""
    P = torch.load(os.path.join(OUT, "params_cal.pt"))
    d = make_diffusion(50)
    fp = load_dit(dev)
    xf = torch.cat([ddim_sample(fp, d, noise_for_seeds(seeds[i:i + bs], dev), labels_for_seeds(seeds[i:i + bs]), 1.5).cpu()
                    for i in range(0, len(seeds), bs)])
    del fp
    torch.cuda.empty_cache()
    q, _ = heig(dev)
    xb, out, tcorr = [], {k: [] for k in plan}, 0.0
    for i in range(0, len(seeds), bs):
        s = seeds[i:i + bs]
        lab = labels_for_seeds(s)
        x0, st = ddim_sample(q, d, noise_for_seeds(s, dev), lab, 1.5, record_states=True)
        xb.append(x0.cpu())
        for arm, place in plan:
            torch.cuda.synchronize(); t0 = time.time()
            src = x0 if place == "final" else st[STEP[place]]
            xc = corrected(arm, P[place][arm], src)
            torch.cuda.synchronize(); tcorr += time.time() - t0
            if place != "final":                                              # resume the ordinary heig-8 sampler
                xc = ddim_sample(q, d, xc, lab, 1.5, start_step=STEP[place], x_init=xc)
            out[(arm, place)].append(xc.cpu())
    return xf, torch.cat(xb), {k: torch.cat(v) for k, v in out.items()}, tcorr


def rel(x, f):
    return ((x - f).flatten(1).pow(2).sum(1) / f.flatten(1).pow(2).sum(1)).double()


def moment(x, f):
    """endpoint band variance ratio vs FP (bands 1-4) and band-1 DC-mean mismatch."""
    X, F = split(x.double()), split(f.double())
    return {"var_ratio": [pooled_var(X[b]) / pooled_var(F[b]) for b in range(4)],
            "mismatch_low": float(sum(abs(np.log(pooled_var(X[b]) / pooled_var(F[b]))) for b in (0, 1))),
            "dc_mean_mismatch": float((X[0].mean((-1, -2)).mean(0) - F[0].mean((-1, -2)).mean(0)).abs().mean())}


def moment_ci(x, f, n=500):
    g = np.random.default_rng(0)
    r = [moment(x[i], f[i])["mismatch_low"] for i in (torch.as_tensor(g.integers(0, len(x), len(x))) for _ in range(n))]
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def paired(a, b):
    """mean relative change of a vs b with a seed-bootstrap CI."""
    diff = (a - b).numpy()
    m = float(b.mean())
    return {"rel_change": float(diff.mean()) / m, "ci": [c / m for c in boot_ci(diff)]}


def design(dev):
    if not guard(0.08, "e1 live design"):
        raise SystemExit(5)
    plan = [(a, p) for a in ARMS for p in PLACES]
    with Run(STAGE, "e1_live_design", dev, {"seeds": [DESIGN[0], DESIGN[-1]]}) as run:
        xf, xb, out, tc = sample_arms(DESIGN, plan, dev)
        rb = rel(xb, xf)
        res = {"seeds": [DESIGN[0], DESIGN[-1]], "baseline": {"rel_mse": float(rb.mean()), **moment(xb, xf)}, "arms": {}}
        for (arm, place), x in out.items():
            r = rel(x, xf)
            res["arms"][f"{arm}@{place}"] = {"rel_mse": float(r.mean()), "vs_baseline": paired(r, rb), **moment(x, xf)}
        order = {p: i for i, p in enumerate(PLACES)}                          # ties -> the later placement
        pick = lambda arm, key: min(PLACES, key=lambda p: (round(res["arms"][f"{arm}@{p}"][key], 12), -order[p]))
        res["frozen"] = {"P": pick("P", "rel_mse"), "M": pick("M", "mismatch_low")}
        res.update(run.record())
    os.makedirs(CK, exist_ok=True)
    atomic_torch_save({"fp": xf, "baseline": xb, **{f"{a}@{p}": v for (a, p), v in out.items()}}, os.path.join(CK, "design_latents.pt"))
    atomic_write_json(os.path.join(OUT, "live_design.json"), res)
    for k, v in res["arms"].items():
        print(f"  {k}: latent {v['rel_mse']:.5f} ({100 * v['vs_baseline']['rel_change']:+.2f}%), low-band mismatch {v['mismatch_low']:.4f}", flush=True)
    print(f"  baseline latent {res['baseline']['rel_mse']:.5f}, mismatch {res['baseline']['mismatch_low']:.4f}; frozen: {res['frozen']}", flush=True)


def confirm(dev):
    fz = json.load(open(os.path.join(OUT, "live_design.json")))["frozen"]
    if not guard(0.12, "e1 live confirm"):
        raise SystemExit(5)
    plan = [("P", fz["P"]), ("M", fz["M"])]
    with Run(STAGE, "e1_live_confirm", dev, {"seeds": [CONFIRM[0], CONFIRM[-1]], "frozen": fz}) as run:
        xf, xb, out, tc = sample_arms(CONFIRM, plan, dev)
        vae = load_vae(dev)
        imf, imb = decode(vae, xf.to(dev)), decode(vae, xb.to(dev))
        ims = {k: decode(vae, v.to(dev)) for k, v in out.items()}
        del vae
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()

        def fid(im):
            ps, lp = [], []
            for i in range(len(im)):
                a, b = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
            return torch.tensor(ps, dtype=torch.float64), torch.tensor(lp, dtype=torch.float64)
        rb, (pb, lb) = rel(xb, xf), fid(imb)
        res = {"seeds": [CONFIRM[0], CONFIRM[-1]], "frozen": fz,
               "baseline": {"rel_mse": float(rb.mean()), "rel_mse_per_seed": rb.tolist(), "psnr": float(pb.mean()), "lpips": float(lb.mean()),
                            **moment(xb, xf), "mismatch_low_ci": moment_ci(xb, xf)}, "arms": {}}
        for (arm, place), x in out.items():
            r, (p, l) = rel(x, xf), fid(ims[(arm, place)])
            res["arms"][f"{arm}@{place}"] = {
                "rel_mse": float(r.mean()), "rel_mse_per_seed": r.tolist(), "psnr": float(p.mean()), "lpips": float(l.mean()),
                "vs_baseline": {"latent": paired(r, rb),
                                "psnr_diff": {"diff": float((p - pb).mean()), "ci": boot_ci((p - pb).numpy())},
                                "lpips_diff": {"diff": float((l - lb).mean()), "ci": boot_ci((l - lb).numpy())}},
                **moment(x, xf), "mismatch_low_ci": moment_ci(x, xf)}
        res["runtime"] = {"correction_seconds_total": tc, "per_image_ms": 1000 * tc / len(CONFIRM),
                          "extra_state": "P: 2 band gains + 4 DC biases; M: 2 band gains + 4 DC shifts (scalars, no per-image state)"}
        q = np.percentile(rb.numpy(), [10, 50, 90])                           # fixed rule: seeds nearest the 10/50/90th percentile
        res["qualitative_seeds"] = [CONFIRM[int(np.argmin(np.abs(rb.numpy() - v)))] for v in q]
        res.update(run.record())
    atomic_torch_save({"fp": xf, "baseline": xb, **{f"{a}@{p}": v for (a, p), v in out.items()}}, os.path.join(CK, "confirm_latents.pt"))
    atomic_torch_save({"fp": imf, "baseline": imb, **{f"{a}@{p}": v for (a, p), v in ims.items()}}, os.path.join(CK, "confirm_images.pt"))
    save_grid(res["qualitative_seeds"], imf, imb, ims)
    atomic_write_json(os.path.join(OUT, "live_confirm.json"), res)
    for k, v in res["arms"].items():
        vb = v["vs_baseline"]
        print(f"  {k}: latent {100 * vb['latent']['rel_change']:+.2f}% [{100 * vb['latent']['ci'][0]:+.2f}, {100 * vb['latent']['ci'][1]:+.2f}], "
              f"PSNR {vb['psnr_diff']['diff']:+.3f}, LPIPS {vb['lpips_diff']['diff']:+.4f}, mismatch {v['mismatch_low']:.4f} vs base {res['baseline']['mismatch_low']:.4f}", flush=True)


def save_grid(seeds, imf, imb, ims):
    from PIL import Image
    cols = [imf, imb] + list(ims.values())
    rows = []
    for s in seeds:
        i = CONFIRM.index(s)
        rows.append(torch.cat([c[i] for c in cols], 2))
    g = torch.cat(rows, 1).permute(1, 2, 0).numpy()
    Image.fromarray(g).save(os.path.join(OUT, "qualitative_fp_base_" + "_".join(f"{a}{p}".replace("@", "") for a, p in ims) + ".png"))


def run(cmd):
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"live_design": design, "live_confirm": confirm}[cmd](dev)
