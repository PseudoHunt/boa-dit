"""Fresh development validation (results/e3c/PROTOCOL.md, 2026-10-29 15:10, step 4): mjhq1k[400:432], 32 untouched prompts.
Candidates fixed by rule: P0; the low-rate screen's P1 and P2 checkpoints with the lowest design endpoint mean; the exploratory
P2 λ0.5 lr1e-4 at α 0.1. Endpoint latent rel. MSE vs the bf16 teacher; paired per-prompt bootstrap CI, median and wins vs P0.
Run: python -m boa_dit.e3c.px_fresh
"""
import json
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.e3c import pixart_refit as R
from boa_dit.e3c import px_diag as D
from boa_dit.f5 import STAGE
from boa_dit.w4a4 import common as WC

FRESH = list(range(400, 432))
OUTF = os.path.join(R.OUT, "fresh_dev.json")


def candidates():
    base = torch.load(R.QBASE, map_location="cpu")
    out = {"P0": base}
    scr = json.load(open(os.path.join(R.OUT, "screen_lr1e-5.json")))
    for arm in ("P1", "P2"):
        a = scr["arms"][arm]
        best = min((k for k in a if k.startswith("u")), key=lambda k: a[k]["endpoint"]["rel_mse_mean"])
        out[f"{arm}_lr1e-5_{best}"] = torch.load(os.path.join(R.CK, f"q_screen_lr1e-5_{arm}_{best}.pt"), map_location="cpu")
    q = torch.load(D.CKPTS["P2_l0.5_lr1e-4"], map_location="cpu")
    out["P2_l0.5_lr1e-4_alpha0.1"] = D.mix(base, q, alpha=0.1)
    return out


@torch.no_grad()
def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    cands = candidates()
    e = WC.load_embeds("mjhq1k")
    with Run(STAGE, "e3c_px_fresh", dev, {"prompts": [FRESH[0], FRESH[-1] + 1], "candidates": list(cands)}) as run:
        pipe, _ = R.setup(dev, trainable=False)
        tf = os.path.join(R.CK, "fresh_teacher.pt")
        if os.path.exists(tf):
            T = torch.load(tf)
        else:
            tch = R.teacher(dev)
            T = torch.cat([R.run_loop(pipe, tch, R.init_latents(pipe, [j], dev), *R.emb(e, [j], dev), dev).float().cpu() for j in FRESH])
            atomic_torch_save(T, tf)
            del tch
            torch.cuda.empty_cache()
        res = {}
        for name, qs in cands.items():
            R.install_q(pipe.transformer, qs)
            X = torch.cat([R.run_loop(pipe, pipe.transformer, R.init_latents(pipe, [j], dev), *R.emb(e, [j], dev), dev).float().cpu() for j in FRESH])
            res[name] = D.endpoint_detail(X, T)
            print(f"  {name}: mean {res[name]['rel_mse_mean']:.4f} median {res[name]['rel_mse_median']:.4f} cos {res[name]['cos_median']:.4f}", flush=True)
        p0 = np.asarray(res["P0"]["rel_mse"])
        idx = np.random.default_rng(0).integers(0, len(p0), (2000, len(p0)))
        paired = {}
        for name in res:
            if name == "P0":
                continue
            x = np.asarray(res[name]["rel_mse"]) - p0
            bs = x[idx].mean(1)
            paired[name] = {"mean_diff": float(x.mean()), "rel_change": float(x.mean() / p0.mean()), "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                            "median_diff": float(np.median(x)), "wins": int((x < 0).sum()), "n": int(len(x))}
            print(f"  {name} vs P0: {100 * paired[name]['rel_change']:+.1f}% diff {paired[name]['mean_diff']:+.4f} [{paired[name]['ci'][0]:+.4f}, {paired[name]['ci'][1]:+.4f}] "
                  f"median {paired[name]['median_diff']:+.4f} wins {paired[name]['wins']}/32", flush=True)
        out = {"prompts": [FRESH[0], FRESH[-1] + 1], "endpoints": res, "paired_vs_P0": paired, **run.record()}
    atomic_write_json(OUTF, out)


if __name__ == "__main__":
    main()
