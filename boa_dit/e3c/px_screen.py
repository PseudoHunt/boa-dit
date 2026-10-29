"""Update-matched PixArt screen (results/e3c/PROTOCOL.md, 2026-10-29 12:20 amendment). Design prompts only; no promotion decision.
Arms from the base branch, seed 0 (identical prompt/start-step draws), lr 1e-4, K = 5: P1 (local), P2 (local + 0.5 rollout), P1-drift (local + μ·D),
checkpoints at 10/25/50/100 updates. μ in {1, 10, 100}: the one whose functional drift at 100 updates is closest to P2's is evaluated in full.
Run: python -m boa_dit.e3c.px_screen
"""
import json
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.e3c import pixart_refit as R
from boa_dit.e3c import px_diag as D
from boa_dit.f5 import STAGE
from boa_dit.w4a4 import common as WC

OUTF = os.path.join(R.OUT, "screen.json")
LR, LAM, CKS, MUS = 1e-4, 0.5, (10, 25, 50, 100), (1.0, 10.0, 100.0)


@torch.no_grad()
def drift(dev, pipe, qpath, C, probe=((0, 5, 10, 15, 19))):
    """functional drift (ratio of sums) of a checkpoint vs the base, on all design prompts x probe steps, through the deployed path."""
    base = torch.load(R.QBASE, map_location="cpu")
    q = torch.load(qpath, map_location="cpu")
    keys = sorted(base)
    R.install_q(pipe.transformer, base)
    e = WC.load_embeds("design")
    ts = R.new_sched(pipe, dev).timesteps
    num, den = {k: 0.0 for k in keys}, {k: 0.0 for k in keys}
    W = {k: (q[k]["L1"].to(dev).to(torch.bfloat16), q[k]["L2"].to(dev).to(torch.bfloat16), base[k]["L1"].to(dev).to(torch.bfloat16), base[k]["L2"].to(dev).to(torch.bfloat16)) for k in keys}
    hs = []
    for k in keys:
        l, ln = k.split(".", 1)
        mod = WC.block_linear(pipe.transformer.transformer_blocks[int(l)], ln)

        def hook(m, inp, out, k=k):
            x = inp[0].float()
            xs = (x if m.d is None else x / m.d).to(torch.bfloat16)
            den[k] += float(out.float().pow(2).sum())
            num[k] += float(((xs @ W[k][0]) @ W[k][1] - (xs @ W[k][2]) @ W[k][3]).float().pow(2).sum())
        hs.append(mod.register_forward_hook(hook))
    for n, j in enumerate(R.DESIGN):
        ehs, mask = R.emb(e, [j], dev)
        for i in probe:
            R.pred(pipe.transformer, C["design"]["x"][n, i:i + 1].to(dev), ts[i], ehs, mask)
    for h in hs:
        h.remove()
    types = sorted({k.split(".", 1)[1] for k in keys})
    return {"total": float(np.sqrt(sum(num.values()) / sum(den.values()))),
            "by_type": {t: float(np.sqrt(sum(num[k] for k in keys if k.endswith(t)) / sum(den[k] for k in keys if k.endswith(t)))) for t in types}}


@torch.no_grad()
def heldout(dev, pipe, tch, qpath, C):
    """held-out local loss at all 20 steps and short-rollout loss at start steps 0/5/10/15 (cached base states + DPM history)."""
    R.install_q(pipe.transformer, torch.load(qpath, map_location="cpu"))
    e = WC.load_embeds("design")
    ts = R.new_sched(pipe, dev).timesteps
    nl, nr = C["norm"]["loc"], C["norm"]["roll"]
    loc, roll = np.zeros((8, 20)), np.zeros((8, 4))
    for n, j in enumerate(R.DESIGN):
        ehs, mask = R.emb(e, [j], dev)
        for i in range(20):
            x = C["design"]["x"][n, i:i + 1].to(dev)
            loc[n, i] = float((R.pred(pipe.transformer, x, ts[i], ehs, mask).float() - R.pred(tch, x, ts[i], ehs, mask).float()).pow(2).sum()) / float(nl[i])
        for c, i in enumerate((0, 5, 10, 15)):
            x = C["design"]["x"][n, i:i + 1].to(dev)
            h = None if i == 0 else C["design"]["hist"][n, i:i + 1].to(dev)
            _, xs = R.rollout(pipe, pipe.transformer, x, h, i, R.K, ehs, mask, dev)
            _, xt = R.rollout(pipe, tch, x, h, i, R.K, ehs, mask, dev)
            roll[n, c] = float((xs.float() - xt.float()).pow(2).sum()) / float(nr[i])
    return {"local_mean": float(loc.mean()), "local_per_step": loc.mean(0).tolist(), "roll_mean": float(roll.mean()), "roll_per_start": roll.mean(0).tolist()}


def evaluate(dev, pipe, tch, C, qpath, tag):
    r = {"endpoint": D.endpoint(dev, pipe, torch.load(qpath, map_location="cpu"), C, detail=True), "drift": drift(dev, pipe, qpath, C),
         "heldout": heldout(dev, pipe, tch, qpath, C)}
    print(f"  {tag}: endpoint mean {r['endpoint']['rel_mse_mean']:.4f} med {r['endpoint']['rel_mse_median']:.4f} cos {r['endpoint']['cos_median']:.4f} | "
          f"drift {r['drift']['total']:.4f} | held-out local {r['heldout']['local_mean']:.3f} roll {r['heldout']['roll_mean']:.3f}", flush=True)
    return r


def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    C = torch.load(os.path.join(R.CK, "cache.pt"))
    res = json.load(open(OUTF)) if os.path.exists(OUTF) else {"lr": LR, "lam": LAM, "checkpoints": list(CKS), "arms": {}}
    with Run(STAGE, "e3c_px_screen", dev, {"lr": LR, "lam": LAM, "cks": CKS, "mus": MUS}) as run:
        pipe, _ = R.setup(dev, trainable=False)
        tch = R.teacher(dev)
        if "P0" not in res["arms"]:
            res["arms"]["P0"] = {"u0": evaluate(dev, pipe, tch, C, R.QBASE, "P0")}
            atomic_write_json(OUTF, res)
        runs = {"P1": ("P1", 0.0, 0.0), "P2": ("P2", LAM, 0.0)}
        for name, (arm, lam, mu) in runs.items():
            if name in res["arms"]:
                continue
            _, lg = R.train(dev, arm, lam, LR, 10.0, f"screen_{name}", pipe=pipe, tch=tch, ckpt_at=CKS, max_updates=max(CKS), mu=mu)
            res["arms"][name] = {"train": {k: v for k, v in lg.items() if k != "log"}}
            for u in CKS:
                res["arms"][name][f"u{u}"] = evaluate(dev, pipe, tch, C, lg["checkpoints"][u]["q"], f"{name} u{u}")
                res["arms"][name][f"u{u}"]["seconds"] = lg["checkpoints"][u]["seconds"]
            atomic_write_json(OUTF, res)
        # μ selection for P1-drift by functional drift at 100 updates (closest to P2's), then full evaluation of the chosen arm
        target = res["arms"]["P2"]["u100"]["drift"]["total"]
        if "mu_select" not in res:
            sel = {}
            for mu in MUS:
                _, lg = R.train(dev, "P1", 0.0, LR, 10.0, f"screen_P1drift_mu{mu:g}", pipe=pipe, tch=tch, ckpt_at=CKS, max_updates=max(CKS), mu=mu)
                d100 = drift(dev, pipe, lg["checkpoints"][100]["q"], C)["total"]
                sel[str(mu)] = {"drift_u100": d100, "train": {k: v for k, v in lg.items() if k != "log"}}
                print(f"  P1-drift μ {mu:g}: drift at 100 updates {d100:.4f} (P2 {target:.4f})", flush=True)
            res["mu_select"] = sel
            res["mu_star"] = float(min(MUS, key=lambda m: abs(sel[str(m)]["drift_u100"] - target)))
            atomic_write_json(OUTF, res)
        if "P1drift" not in res["arms"]:
            ck = res["mu_select"][str(res["mu_star"]) if str(res["mu_star"]) in res["mu_select"] else f"{res['mu_star']:g}"]["train"]["checkpoints"]
            res["arms"]["P1drift"] = {"mu": res["mu_star"]}
            for u in CKS:
                q = ck[str(u)]["q"] if str(u) in ck else ck[u]["q"]
                res["arms"]["P1drift"][f"u{u}"] = evaluate(dev, pipe, tch, C, q, f"P1drift(μ {res['mu_star']:g}) u{u}")
            atomic_write_json(OUTF, res)
        res["run"] = run.record()
    atomic_write_json(OUTF, res)


if __name__ == "__main__":
    main()
