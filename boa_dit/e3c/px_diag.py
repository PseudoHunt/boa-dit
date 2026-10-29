"""PixArt refit repair diagnostics (results/e3c/PROTOCOL.md, 2026-10-29 amendment after the user's review). Design prompts design[24:32]
only; confirmation prompts untouched; no new training. Checkpoints: the existing short design runs.
  steps   : per-step (all 20) teacher/student prediction error on (A) cached base states and (B) each model's own states; (C) free-running
            state error vs the teacher trajectory at every step; (D) held-out short-rollout (K=5) loss from cached base states
  branch  : ||L1L2 - L10L20||_F / ||L10L20||_F and the branch-output change ||X(L1L2 - L10L20)|| / ||layer output|| on probe activations
  interv  : restore only the cross-attention output branches (attn2.to_out.0) / keep only those trained -> design endpoint latent
  alpha   : L_j(α) = L_j0 + α(L_j - L_j0), α in {0.03, 0.1, 0.3} (0 = base, 1 = trained) -> design endpoint latent (bf16 deployment path)
  norm    : loss denominators per step and per-step gradient norms of the normalized losses at the base branch (2 training prompts)
Run: python -m boa_dit.e3c.px_diag steps|branch|interv|alpha|norm|all
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.e3c import pixart_refit as R
from boa_dit.f5 import STAGE
from boa_dit.w4a4 import common as WC

PQ = R.CK
OUTF = os.path.join(R.OUT, "diag.json")
CKPTS = {"P0": R.QBASE,
         "P1_lr1e-5": f"{PQ}/q_design_P1_lr1e-05.pt", "P1_lr3e-5": f"{PQ}/q_design_P1_lr3e-05.pt", "P1_lr1e-4": f"{PQ}/q_design_P1_lr0.0001.pt",
         "P2_l0.5_lr1e-4": f"{PQ}/q_design_P2_lam0.5.pt", "P2_l2_lr1e-4": f"{PQ}/q_design_P2_lam2.pt",
         "P2_l0.5_lr1e-5": f"{PQ}/q_design_P2_lam0.5_lr1e-05.pt", "P2_l2_lr1e-5": f"{PQ}/q_design_P2_lam2_lr1e-05.pt"}
CKPTS = {k: v for k, v in CKPTS.items() if os.path.exists(v)}
XATTN = "attn2.to_out.0"


EPD = {}


def endpoint_detail(x, t):
    """per prompt: latent rel. MSE, RMS ratio to the teacher latent, cosine similarity (distinguishes changed direction from magnitude blow-up)."""
    if x is None:
        return None
    x, t = x.flatten(1).double(), t.flatten(1).double()
    rel = ((x - t).pow(2).sum(1) / t.pow(2).sum(1))
    rms = x.pow(2).mean(1).sqrt() / t.pow(2).mean(1).sqrt()
    cos = (x * t).sum(1) / (x.norm(dim=1) * t.norm(dim=1))
    return {"rel_mse": rel.tolist(), "rel_mse_mean": float(rel.mean()), "rel_mse_median": float(rel.median()), "rms_ratio": rms.tolist(),
            "cos": cos.tolist(), "cos_median": float(cos.median()), "rms_ratio_median": float(rms.median())}


def load_out():
    return json.load(open(OUTF)) if os.path.exists(OUTF) else {}


def save(key, val):
    d = load_out()
    d[key] = val
    atomic_write_json(OUTF, d)


@torch.no_grad()
def steps(dev, pipe, tch, C):
    e = WC.load_embeds("design")
    nl, nr = C["norm"]["loc"], C["norm"]["roll"]
    ts = R.new_sched(pipe, dev).timesteps
    teach = []                                                                # teacher trajectories (states incl. final)
    for n, j in enumerate(R.DESIGN):
        ehs, mask = R.emb(e, [j], dev)
        lat, st, _ = R.run_loop(pipe, tch, R.init_latents(pipe, [j], dev), ehs, mask, dev, record=True)
        teach.append([s.float() for s in st] + [lat.float().cpu()])
    res = {}
    for name, q in CKPTS.items():
        R.install_q(pipe.transformer, torch.load(q, map_location="cpu"))
        A, B, Cst, D = np.zeros((8, 20)), np.zeros((8, 20)), np.zeros((8, 21)), np.zeros((8, 20))
        ends, tends = [], []
        for n, j in enumerate(R.DESIGN):
            ehs, mask = R.emb(e, [j], dev)
            for i in range(20):                                               # (A) cached base states
                x = C["design"]["x"][n, i:i + 1].to(dev)
                A[n, i] = float((R.pred(pipe.transformer, x, ts[i], ehs, mask).float() - R.pred(tch, x, ts[i], ehs, mask).float()).pow(2).sum()) / float(nl[i])
                k = min(R.K, 20 - i)                                          # (D) held-out short rollout from cached base state + history
                h = None if i == 0 else C["design"]["hist"][n, i:i + 1].to(dev)
                _, xs = R.rollout(pipe, pipe.transformer, x, h, i, k, ehs, mask, dev)
                _, xt = R.rollout(pipe, tch, x, h, i, k, ehs, mask, dev)
                D[n, i] = float((xs.float() - xt.float()).pow(2).sum()) / float(nr[i])
            lat, st, _ = R.run_loop(pipe, pipe.transformer, R.init_latents(pipe, [j], dev), ehs, mask, dev, record=True)
            own = st + [lat.cpu()]
            for i in range(20):                                               # (B) the model's own states
                x = own[i].to(dev)
                B[n, i] = float((R.pred(pipe.transformer, x, ts[i], ehs, mask).float() - R.pred(tch, x, ts[i], ehs, mask).float()).pow(2).sum()) / float(nl[i])
            for i in range(21):                                               # (C) free-running state error vs the teacher trajectory
                a, b = own[i].float(), teach[n][i]
                Cst[n, i] = float((a - b).pow(2).sum() / b.pow(2).sum())
            ends.append(own[20].float()); tends.append(teach[n][20])
        EPD["last"] = endpoint_detail(torch.cat(ends), torch.cat(tends))
        atomic_torch_save({"own_final": torch.cat(ends), "teacher_final": torch.cat(tends)}, os.path.join(PQ, f"diag_endpoints_{name}.pt"))
        res[name] = {"local_cached": A.mean(0).tolist(), "local_own": B.mean(0).tolist(), "state_err": Cst.mean(0).tolist(), "rollout_heldout": D.mean(0).tolist(),
                     "endpoint": float(Cst[:, 20].mean()), "endpoint_median": float(np.median(Cst[:, 20])),
                     "per_prompt": {"local_cached": A.tolist(), "local_own": B.tolist(), "state_err": Cst.tolist(), "rollout_heldout": D.tolist()},
                     "endpoint_detail": EPD.pop("last")}
        print(f"  {name}: endpoint {res[name]['endpoint']:.4f} | local cached mean {A.mean():.3f} own {B.mean():.3f} | rollout held-out {D.mean():.3f} | "
              f"state err at steps 5/10/15/20 {[round(Cst[:, i].mean(), 4) for i in (5, 10, 15, 20)]}", flush=True)
    save("steps", res)


@torch.no_grad()
def branch(dev, pipe, C):
    base = torch.load(R.QBASE, map_location="cpu")
    ck = {k: torch.load(v, map_location="cpu") for k, v in CKPTS.items() if k != "P0"}
    keys = sorted(base)
    prod = {}
    for name, q in ck.items():
        r = {}
        for k in keys:
            B0 = base[k]["L1"].to(torch.bfloat16).float() @ base[k]["L2"].to(torch.bfloat16).float()
            B1 = q[k]["L1"].float() @ q[k]["L2"].float()
            r[k] = float((B1 - B0).norm() / B0.norm().clamp_min(1e-12))
        prod[name] = r
    # branch-output change on probe activations (base trajectory states of design prompt 0, steps 0, 10, 19), relative to the layer output energy
    R.install_q(pipe.transformer, base)
    e = WC.load_embeds("design")
    ehs, mask = R.emb(e, [R.DESIGN[0]], dev)
    ts = R.new_sched(pipe, dev).timesteps
    acc = {name: {k: [0.0, 0.0] for k in keys} for name in ck}
    dev_d = {name: {k: (q[k]["L1"].to(dev).to(torch.bfloat16), q[k]["L2"].to(dev).to(torch.bfloat16),
                        base[k]["L1"].to(dev).to(torch.bfloat16), base[k]["L2"].to(dev).to(torch.bfloat16)) for k in keys} for name, q in ck.items()}
    hs = []
    for k in keys:
        l, ln = k.split(".", 1)
        mod = R.WC.block_linear(pipe.transformer.transformer_blocks[int(l)], ln)

        def hook(m, inp, out, k=k):
            x = inp[0].float()
            xs = (x if m.d is None else x / m.d).to(torch.bfloat16)
            yy = float(out.float().pow(2).sum())
            for name in ck:
                L1, L2, L10, L20 = dev_d[name][k]
                dlt = (xs @ L1) @ L2 - (xs @ L10) @ L20
                acc[name][k][0] += float(dlt.float().pow(2).sum())
                acc[name][k][1] += yy
        hs.append(mod.register_forward_hook(hook))
    for i in (0, 10, 19):
        R.pred(pipe.transformer, C["design"]["x"][0, i:i + 1].to(dev), ts[i], ehs, mask)
    for h in hs:
        h.remove()
    out = {}
    for name in ck:
        rel = {k: float(np.sqrt(v[0] / max(v[1], 1e-12))) for k, v in acc[name].items()}
        by = {}
        for k, v in rel.items():
            by.setdefault(k.split(".", 1)[1], []).append(v)
        top = sorted(rel.items(), key=lambda kv: -kv[1])[:8]
        out[name] = {"output_change_rel_by_type_mean": {t: float(np.mean(v)) for t, v in by.items()}, "output_change_top": top,
                     "product_change_rel_by_type_mean": {t: float(np.mean([prod[name][k] for k in keys if k.endswith(t)])) for t in by},
                     "product_change_top": sorted(prod[name].items(), key=lambda kv: -kv[1])[:8]}
        print(f"  {name}: output change by type {json.dumps({t: round(v, 4) for t, v in out[name]['output_change_rel_by_type_mean'].items()})}", flush=True)
    save("branch", out)


def mix(base, q, alpha=1.0, only=None, restore=None):
    o = {}
    for k in base:
        use = (only is None or k.endswith(only)) and not (restore is not None and k.endswith(restore))
        if use:
            L1 = base[k]["L1"].to(torch.bfloat16).float() + alpha * (q[k]["L1"].float() - base[k]["L1"].to(torch.bfloat16).float())
            L2 = base[k]["L2"].to(torch.bfloat16).float() + alpha * (q[k]["L2"].float() - base[k]["L2"].to(torch.bfloat16).float())
            o[k] = dict(base[k], L1=L1.to(torch.bfloat16), L2=L2.to(torch.bfloat16))
        else:
            o[k] = base[k]
    return o


@torch.no_grad()
def endpoint(dev, pipe, qs, C, detail=False, grid=None):
    """design endpoint through the deployment path; detail -> per-prompt rel. MSE, RMS ratio, cosine; grid -> decoded uint8 images saved."""
    R.install_q(pipe.transformer, qs)
    e = WC.load_embeds("design")
    X = []
    for n, j in enumerate(R.DESIGN):
        ehs, mask = R.emb(e, [j], dev)
        X.append(R.run_loop(pipe, pipe.transformer, R.init_latents(pipe, [j], dev), ehs, mask, dev).float().cpu())
    X = torch.cat(X)
    d = endpoint_detail(X, C["design"]["x0fp"].float())
    if grid:
        from boa_dit.temporal import pixart as PX
        imgs = torch.cat([PX.decode(pipe, X[k:k + 1].to(dev)) for k in range(len(X))])
        atomic_torch_save(imgs, os.path.join(PQ, f"grid_{grid}.pt"))
    return d if detail else d["rel_mse_mean"]


def interv(dev, pipe, C):
    base = torch.load(R.QBASE, map_location="cpu")
    out = {}
    for name in [k for k in CKPTS if k != "P0"]:
        q = torch.load(CKPTS[name], map_location="cpu")
        out[name] = {"restore_xattn_out": endpoint(dev, pipe, mix(base, q, restore=XATTN), C),
                     "only_xattn_out": endpoint(dev, pipe, mix(base, q, only=XATTN), C)}
        print(f"  {name}: restore {XATTN} -> {out[name]['restore_xattn_out']:.4f}; only {XATTN} trained -> {out[name]['only_xattn_out']:.4f}", flush=True)
    save("interv", out)


QO = ("attn2.to_q", "attn2.to_out.0")


def mix2(base, q, keep):
    """keep: predicate on the layer key -> use the trained factors; else base."""
    return {k: (dict(base[k], L1=q[k]["L1"], L2=q[k]["L2"]) if keep(k) else base[k]) for k in base}


def interv2(dev, pipe, C):
    """attn2.to_q alone and attn2.to_q + attn2.to_out.0 together (restore / keep-only), with per-prompt endpoint detail; checkpoint transplants,
    not frozen-subset training. Also re-reports the attn2.to_out-only cases with detail, the trained checkpoints, and image grids."""
    base = torch.load(R.QBASE, map_location="cpu")
    out = {"P0": endpoint(dev, pipe, base, C, detail=True, grid="P0")}
    for name in [k for k in CKPTS if k != "P0"]:
        q = torch.load(CKPTS[name], map_location="cpu")
        r = {"trained": endpoint(dev, pipe, q, C, detail=True, grid=name)}
        for lab, keep in (("restore_q", lambda k: not k.endswith("attn2.to_q")), ("only_q", lambda k: k.endswith("attn2.to_q")),
                          ("restore_qo", lambda k: not k.endswith(QO)), ("only_qo", lambda k: k.endswith(QO)),
                          ("restore_o", lambda k: not k.endswith("attn2.to_out.0")), ("only_o", lambda k: k.endswith("attn2.to_out.0"))):
            r[lab] = endpoint(dev, pipe, mix2(base, q, keep), C, detail=True, grid=f"{name}_{lab}" if lab in ("restore_o", "restore_qo") else None)
        out[name] = r
        print(f"  {name}: " + "; ".join(f"{k} mean {v['rel_mse_mean']:.3f} med {v['rel_mse_median']:.3f} cos {v['cos_median']:.3f} rms {v['rms_ratio_median']:.3f}"
                                        for k, v in r.items()), flush=True)
    save("interv2", out)


@torch.no_grad()
def branch2(dev, pipe, C):
    """functional drift on all 8 design prompts x steps {0, 5, 10, 15, 19}; per layer and per type as a ratio of sums (not a mean of ratios)."""
    base = torch.load(R.QBASE, map_location="cpu")
    ck = {k: torch.load(v, map_location="cpu") for k, v in CKPTS.items() if k != "P0"}
    keys = sorted(base)
    R.install_q(pipe.transformer, base)
    e = WC.load_embeds("design")
    ts = R.new_sched(pipe, dev).timesteps
    num = {n: {k: 0.0 for k in keys} for n in ck}
    den = {k: 0.0 for k in keys}
    W = {n: {k: (q[k]["L1"].to(dev).to(torch.bfloat16), q[k]["L2"].to(dev).to(torch.bfloat16)) for k in keys} for n, q in ck.items()}
    B0 = {k: (base[k]["L1"].to(dev).to(torch.bfloat16), base[k]["L2"].to(dev).to(torch.bfloat16)) for k in keys}
    hs = []
    for k in keys:
        l, ln = k.split(".", 1)
        mod = R.WC.block_linear(pipe.transformer.transformer_blocks[int(l)], ln)

        def hook(m, inp, out, k=k):
            x = inp[0].float()
            xs = (x if m.d is None else x / m.d).to(torch.bfloat16)
            den[k] += float(out.float().pow(2).sum())
            b0 = (xs @ B0[k][0]) @ B0[k][1]
            for n in ck:
                num[n][k] += float(((xs @ W[n][k][0]) @ W[n][k][1] - b0).float().pow(2).sum())
        hs.append(mod.register_forward_hook(hook))
    for n_, j in enumerate(R.DESIGN):
        ehs, mask = R.emb(e, [j], dev)
        for i in (0, 5, 10, 15, 19):
            R.pred(pipe.transformer, C["design"]["x"][n_, i:i + 1].to(dev), ts[i], ehs, mask)
    for h in hs:
        h.remove()
    out = {}
    for n in ck:
        types = sorted({k.split(".", 1)[1] for k in keys})
        out[n] = {"by_type": {t: float(np.sqrt(sum(num[n][k] for k in keys if k.endswith(t)) / sum(den[k] for k in keys if k.endswith(t)))) for t in types},
                  "total": float(np.sqrt(sum(num[n].values()) / sum(den.values()))),
                  "top_layers": sorted(((k, float(np.sqrt(num[n][k] / max(den[k], 1e-12)))) for k in keys), key=lambda kv: -kv[1])[:10]}
        print(f"  {n}: total drift {out[n]['total']:.4f}; by type {json.dumps({t: round(v, 4) for t, v in out[n]['by_type'].items()})}", flush=True)
    save("branch2", out)


def alpha(dev, pipe, C):
    base = torch.load(R.QBASE, map_location="cpu")
    out = {}
    for name in [k for k in CKPTS if k != "P0"]:
        q = torch.load(CKPTS[name], map_location="cpu")
        out[name] = {str(a): endpoint(dev, pipe, mix(base, q, alpha=a), C) for a in (0.03, 0.1, 0.3)}
        print(f"  {name}: endpoint at α 0.03/0.1/0.3 = {[round(v, 4) for v in out[name].values()]}", flush=True)
    save("alpha", out)


def norm(dev, pipe, tch, C):
    """denominators and per-step gradient norms of the normalized losses at the base branch (training prompts 0, 1)."""
    mods = R.install_train(pipe.transformer, dev)
    pipe.transformer.enable_gradient_checkpointing()
    params = [p for m in mods.values() for p in (m.L1p, m.L2p)]
    e = WC.load_embeds(C["train"]["set"])
    ts = R.new_sched(pipe, dev).timesteps
    nl, nr = C["norm"]["loc"], C["norm"]["roll"]
    g_loc, g_roll = np.zeros((2, 20)), np.zeros((2, 20))
    for n in range(2):
        ehs, mask = R.emb(e, [C["train"]["idx"][n]], dev)
        for i in range(20):
            x = C["train"]["x"][n, i:i + 1].to(dev)
            h = None if i == 0 else C["train"]["hist"][n, i:i + 1].to(dev)
            k = min(R.K, 20 - i)
            with torch.no_grad():
                et, xt = R.rollout(pipe, tch, x, h, i, k, ehs, mask, dev)
            es = R.pred(pipe.transformer, x, ts[i], ehs, mask)
            gl = torch.autograd.grad((es.float() - et.float()).pow(2).sum() / float(nl[i]), params)
            g_loc[n, i] = float(torch.sqrt(sum(g.pow(2).sum() for g in gl)))
            _, xs = R.rollout(pipe, pipe.transformer, x, h, i, k, ehs, mask, dev)
            gr = torch.autograd.grad((xs.float() - xt.float()).pow(2).sum() / float(nr[i]), params)
            g_roll[n, i] = float(torch.sqrt(sum(g.pow(2).sum() for g in gr)))
    pipe.transformer.disable_gradient_checkpointing()
    R.install_q(pipe.transformer, torch.load(R.QBASE, map_location="cpu"))
    out = {"n_loc": nl.tolist(), "n_roll": nr.tolist(), "grad_norm_local_per_step": g_loc.mean(0).tolist(), "grad_norm_roll_per_step": g_roll.mean(0).tolist(),
           "note": "denominators are base errors on 4 training prompts; gradients at the untrained branch, training prompts 0-1"}
    print("  n_loc", np.round(nl.numpy(), 4).tolist(), "\n  n_roll", np.round(nr.numpy(), 4).tolist(),
          "\n  grad local", np.round(g_loc.mean(0), 3).tolist(), "\n  grad roll", np.round(g_roll.mean(0), 3).tolist(), flush=True)
    save("norm", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["steps", "branch", "branch2", "interv", "interv2", "alpha", "norm", "all", "v2"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    C = torch.load(os.path.join(R.CK, "cache.pt"))
    pipe, _ = R.setup(dev, trainable=False)
    tch = R.teacher(dev)
    todo = {"all": ["norm", "branch", "interv", "alpha", "steps"], "v2": ["steps", "branch2", "interv2"]}.get(a.cmd, [a.cmd])
    with Run(STAGE, f"e3c_px_diag_{a.cmd}", dev, {"ckpts": list(CKPTS)}) as run:
        for c in todo:
            print(f"== {c}", flush=True)
            {"steps": lambda: steps(dev, pipe, tch, C), "branch": lambda: branch(dev, pipe, C), "interv": lambda: interv(dev, pipe, C),
             "alpha": lambda: alpha(dev, pipe, C), "norm": lambda: norm(dev, pipe, tch, C), "branch2": lambda: branch2(dev, pipe, C),
             "interv2": lambda: interv2(dev, pipe, C)}[c]()
        run.record()


if __name__ == "__main__":
    main()
