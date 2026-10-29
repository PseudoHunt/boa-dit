"""Consolidated §2 (results/temporal/observability/rule_observability.md): observability of heig-8's persistent weight error.
  traces  : per seed (cal 7000-7063, eval 7100-7163): FP states, FP eps and heig-8 eps-hat on FP states, free-running heig-8 states
  audit   : (a1) shared mean vs per-seed residual (split-half unbiased), (a2) distributional-interpretation gate (cross term)
  predict : (b) ridge predictors of the DCT-16 low-band residual: oracle history, deployable history, current-only
  bt      : (a1) per-step bias-table arm, latent on eval seeds (only if the shared mean is >= 10%)
Run: python -m boa_dit.observability.run traces|audit|predict|bt
"""
import argparse
import os
import sys

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks
from boa_dit.sample import ddim_sample
from boa_dit.sampler_corr.run import band_split, heig
from boa_dit.weights_joint.common import step_outputs

import importlib.util as _ilu                                                       # shared c3 budget (temporal-oracles branch)
_spec = _ilu.spec_from_file_location("c3", __import__("os").path.join(__import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))), "temporal", "c3.py"))
_c3 = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_c3)
STAGE, guard = _c3.STAGE, _c3.guard

OUT = os.path.join(RESULTS, "temporal", "observability")
CK = os.path.join(CKPT_DIR, "observability")
CAL, EVAL = list(range(7000, 7064)), list(range(7100, 7164))
T, NB = 50, 10


def tpath(name):
    return os.path.join(CK, f"traces_{name}.pt")


@torch.no_grad()
def traces(dev):
    if not guard(0.15, "observability traces"):
        sys.exit(5)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "obs_traces", dev, {"cal": [CAL[0], CAL[-1]], "eval": [EVAL[0], EVAL[-1]]}) as run:
        d = make_diffusion(50)
        fp = load_dit(dev)
        q, _ = heig(dev)
        for name, seeds in (("cal", CAL), ("eval", EVAL)):
            if os.path.exists(tpath(name)):
                continue
            R = {k: torch.zeros(len(seeds), T, 4, 32, 32) for k in ("xfp", "xq", "eps", "epsq")}
            R["x0fp"], R["x0q"] = torch.zeros(len(seeds), 4, 32, 32), torch.zeros(len(seeds), 4, 32, 32)
            for b in range(0, len(seeds), 32):
                s = seeds[b:b + 32]
                lab, nz = labels_for_seeds(s), noise_for_seeds(s, dev)
                x0f, st = ddim_sample(fp, d, nz, lab, 1.5, record_states=True)
                x0q, sq = ddim_sample(q, d, nz, lab, 1.5, record_states=True)
                R["x0fp"][b:b + 32], R["x0q"][b:b + 32] = x0f.cpu(), x0q.cpu()
                for i in range(T):
                    R["xfp"][b:b + 32, i], R["xq"][b:b + 32, i] = st[i][:, :4].cpu(), sq[i][:, :4].cpu()
                    R["eps"][b:b + 32, i] = step_outputs(fp, d, st[i], lab, i)[:, :4].cpu()
                    R["epsq"][b:b + 32, i] = step_outputs(q, d, st[i], lab, i)[:, :4].cpu()
                print(f"  traces {name}: seeds {s[0]}-{s[-1]} done", flush=True)
            R["seeds"] = seeds
            atomic_torch_save(R, tpath(name))
        run.record()


def bucket(i):
    return i // (T // NB)


def lag1(X):
    """X [S, T, ...]: pooled lag-1 correlation over steps (mean over t of per-t pooled corr)."""
    a, b = X[:, :-1].flatten(2), X[:, 1:].flatten(2)
    return float(((a * b).sum((0, 2)) / torch.sqrt(a.pow(2).sum((0, 2)) * b.pow(2).sum((0, 2))).clamp_min(1e-30)).mean())


@torch.no_grad()
def audit():
    C, E = torch.load(tpath("cal")), torch.load(tpath("eval"))
    masks = band_masks()
    res = {"bands": "4 equal-area radial bands (d2.py), band 1 lowest", "buckets": "10 x 5 steps"}
    # (a1) shared mean vs per-seed residual on calibration seeds, split-half unbiased shared energy
    D = (C["epsq"] - C["eps"]).double()                                          # [64, T, 4, 32, 32]
    S = D.shape[0]
    comps = band_split(D, masks)
    a1 = {"per_band": {}, "per_bucket": {}}
    tot_shared, tot = 0.0, 0.0
    for bi, cp in enumerate(comps):
        mA, mB, m = cp[: S // 2].mean(0), cp[S // 2:].mean(0), cp.mean(0)
        sh_t = (mA * mB).flatten(1).sum(1)                                          # [T] unbiased shared energy (per sample)
        en_t = cp.pow(2).flatten(2).sum(2).mean(0)                                 # [T] mean per-seed energy
        resid = cp - m
        a1["per_band"][f"band{bi + 1}"] = {
            "shared_energy": float(sh_t.sum()), "total_energy": float(en_t.sum()), "shared_fraction": float(sh_t.sum() / en_t.sum()),
            "shared_fraction_naive": float(m.pow(2).flatten(1).sum(1).sum() / en_t.sum()),
            "lag1_mean_part": lag1(m[None]), "lag1_residual_part": lag1(resid),
            "per_bucket_shared_fraction": [float(sh_t[k * 5:(k + 1) * 5].sum() / en_t[k * 5:(k + 1) * 5].sum()) for k in range(NB)]}
        tot_shared += float(sh_t.sum()); tot += float(en_t.sum())
    a1["shared_fraction_total"] = tot_shared / tot
    a1["bt_arm_triggered"] = a1["shared_fraction_total"] >= 0.10
    res["a1"] = a1
    # (a2) distributional-interpretation gate on eval seeds: x^q vs x^FP, d = x^q - x^FP, across-seed covariance traces
    XF, XQ = E["xfp"].double(), E["xq"].double()
    Dd = XQ - XF
    cf, cd = band_split(XF, masks), band_split(Dd, masks)
    a2 = {}
    for bi in range(4):
        f, dd = cf[bi], cd[bi]
        f, dd = f - f.mean(0), dd - dd.mean(0)
        n = f.shape[0] - 1
        vF = f.pow(2).sum(0).flatten(1).sum(1) / n                                  # [T]
        vD = dd.pow(2).sum(0).flatten(1).sum(1) / n
        X = (f * dd).sum(0).flatten(1).sum(1) / n
        q = f + dd
        vQ = q.pow(2).sum(0).flatten(1).sum(1) / n
        rows = []
        for k in range(NB):
            sl = slice(k * 5, (k + 1) * 5)
            rows.append({"trCovFP": float(vF[sl].sum()), "trCovD": float(vD[sl].sum()), "dV": float(vQ[sl].sum() - vF[sl].sum()),
                         "cross": float(X[sl].sum()), "rho": float(2 * X[sl].sum() / vD[sl].sum().clamp_min(1e-30)),
                         "corr_norm": float(X[sl].sum() / torch.sqrt(vF[sl].sum() * vD[sl].sum()).clamp_min(1e-30))})
        w = np.array([r["trCovD"] for r in rows])
        a2[f"band{bi + 1}"] = {"buckets": rows, "rho_weighted": float(np.sum(w * np.array([r["rho"] for r in rows])) / w.sum()),
                               "trCovD_total": float(w.sum())}
    top = max(range(4), key=lambda b: a2[f"band{b + 1}"]["trCovD_total"])
    a2["band_most_trCovD"] = f"band{top + 1}"
    a2["gate_pass"] = bool(abs(a2["band1"]["rho_weighted"]) < 0.5 and abs(a2[f"band{top + 1}"]["rho_weighted"]) < 0.5)
    a2["interpretation"] = "adds marginal variance" if a2["gate_pass"] else "mostly rotates/contracts the signal (cross term comparable to Cov(d))"
    res["a2"] = a2
    atomic_write_json(os.path.join(OUT, "audit.json"), res)
    print(f"(a1) shared-mean fraction {a1['shared_fraction_total']:.4f} (naive {sum(v['shared_fraction_naive'] * v['total_energy'] for v in a1['per_band'].values()) / tot:.4f}); "
          f"B_t arm triggered: {a1['bt_arm_triggered']}")
    for b, v in a1["per_band"].items():
        print(f"   {b}: shared {v['shared_fraction']:.4f}, lag-1 mean part {v['lag1_mean_part']:.3f}, residual part {v['lag1_residual_part']:.3f}, energy {v['total_energy']:.4g}")
    for b in ("band1", "band2", "band3", "band4"):
        v = a2[b]
        print(f"(a2) {b}: rho (trCov(d)-weighted) {v['rho_weighted']:+.3f}; per bucket {[round(r['rho'], 2) for r in v['buckets']]}; trCov(d) {v['trCovD_total']:.4g}")
    print(f"(a2) gate pass: {a2['gate_pass']} -> {a2['interpretation']}", flush=True)


def dct_basis(n=32, k=4):
    i = torch.arange(n, dtype=torch.float64)
    B = torch.stack([torch.cos(np.pi * (i + 0.5) * f / n) for f in range(k)])
    B[0] *= 1 / np.sqrt(2)
    return B * np.sqrt(2 / n)                                                    # [k, n] orthonormal rows


def coef(X, B):
    """X [..., 4, 32, 32] -> [..., 64] DCT-16 coefficients per channel."""
    c = torch.einsum("ai,...ij,bj->...ab", B, X.double(), B)
    return c.flatten(-3)


def ridge_fit(F, Y, lam):
    mu_f, mu_y, sd = F.mean(0), Y.mean(0), F.std(0).clamp_min(1e-12)
    Fs = (F - mu_f) / sd
    A = Fs.T @ Fs + lam * len(F) * torch.eye(F.shape[1], dtype=F.dtype)
    W = torch.linalg.solve(A, Fs.T @ (Y - mu_y))
    return lambda G: ((G - mu_f) / sd) @ W + mu_y


@torch.no_grad()
def predict():
    C, E = torch.load(tpath("cal")), torch.load(tpath("eval"))
    B = dct_basis()
    mu = (C["epsq"] - C["eps"]).double().mean(0)                                  # [T, 4, 32, 32] cal shared mean
    def prep(R):
        r = coef((R["epsq"] - R["eps"]).double() - mu, B)                         # [S, T, 64] centred residual, DCT-16
        e = coef(R["epsq"], B)                                                     # eps-hat coefficients
        return r, e
    rC, eC = prep(C); rE, eE = prep(E)
    def feats(kind, r, e, t):
        if kind == "oracle":
            return r[:, t - 1]
        if kind == "current":
            return e[:, t]
        return torch.cat([e[:, t], e[:, t - 1], e[:, t] - e[:, t - 1], torch.full((len(e), 1), float(t), dtype=e.dtype)], 1)
    LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    res = {"target": "DCT-16 (4x4 lowest 2-D DCT-II modes per channel, 64 coefficients) of the centred residual", "steps": "1..49"}
    preds = {}
    for kind in ("oracle", "deployable", "current"):
        P = torch.zeros_like(rE)
        chosen = []
        for k in range(NB):
            ts = [t for t in range(k * 5, (k + 1) * 5) if t >= 1]
            Fc = torch.cat([feats(kind, rC, eC, t) for t in ts]); Yc = torch.cat([rC[:, t] for t in ts])
            sid = torch.cat([torch.arange(len(rC)) for _ in ts])
            best, bl = None, None
            for lam in LAMS:                                                     # 4-fold CV over calibration seeds
                err = 0.0
                for f in range(4):
                    tr, te = (sid % 4) != f, (sid % 4) == f
                    m = ridge_fit(Fc[tr], Yc[tr], lam)
                    err += float((m(Fc[te]) - Yc[te]).pow(2).sum())
                if best is None or err < best:
                    best, bl = err, lam
            chosen.append(bl)
            m = ridge_fit(Fc, Yc, bl)
            for t in ts:
                P[:, t] = m(feats(kind, rE, eE, t))
        preds[kind] = P
        res[kind] = {"lambda_per_bucket": chosen}
    r = rE[:, 1:]
    g = torch.Generator().manual_seed(0)
    idx = [torch.randint(0, len(r), (len(r),), generator=g) for _ in range(2000)]
    def metrics(P, ix=None):
        rr, pp = (r, P[:, 1:]) if ix is None else (r[ix], P[ix][:, 1:])
        res_ = rr - pp
        Tn = rr.shape[1]
        return {"mse_frac": float(res_.pow(2).sum() / rr.pow(2).sum()),
                "C_before": float(Tn * rr.mean(1).pow(2).sum()), "C_after": float(Tn * res_.mean(1).pow(2).sum()),
                "temporal_sum_left": float(res_.sum(1).pow(2).sum() / rr.sum(1).pow(2).sum())}
    for kind, P in preds.items():
        m = metrics(P)
        bs = [metrics(P, ix) for ix in idx]
        for key in ("mse_frac", "temporal_sum_left"):
            v = np.array([b[key] for b in bs])
            m[key + "_ci"] = [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
        m["C_removed_frac"] = 1 - m["C_after"] / m["C_before"]
        res[kind].update(m)
    rat = lambda ix=None: metrics(preds["deployable"], ix)["mse_frac"] / metrics(preds["current"], ix)["mse_frac"]
    v = np.array([rat(ix) for ix in idx])
    res["deployable_over_current"] = {"ratio": rat(), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}
    res["gate_c"] = bool(res["deployable_over_current"]["ratio"] <= 0.8 and res["deployable_over_current"]["ci"][1] < 1)
    atomic_write_json(os.path.join(OUT, "predict.json"), res)
    for kind in ("oracle", "deployable", "current"):
        m = res[kind]
        print(f"(b) {kind:10s}: held-out MSE {m['mse_frac']:.3f} {[round(x, 3) for x in m['mse_frac_ci']]} of zero-prediction; coherent C removed "
              f"{100 * m['C_removed_frac']:.1f}% ({m['C_before']:.4g} -> {m['C_after']:.4g}); temporal sum left {m['temporal_sum_left']:.3f}; lambdas {m['lambda_per_bucket']}")
    print(f"(b) deployable / current MSE {res['deployable_over_current']['ratio']:.3f} {[round(x, 3) for x in res['deployable_over_current']['ci']]}; gate for (c): {res['gate_c']}", flush=True)


@torch.no_grad()
def bt(dev):
    """per-step bias table B_t (cal mean of eps-hat - eps), subtracted from heig-8's guided eps; latent rel. MSE on eval seeds."""
    if not guard(0.08, "observability B_t latent"):
        sys.exit(5)
    C, E = torch.load(tpath("cal")), torch.load(tpath("eval"))
    Bt = (C["epsq"] - C["eps"]).mean(0).to(dev)
    with Run(STAGE, "obs_bt_latent", dev, {}) as run:
        d = make_diffusion(50)
        q, _ = heig(dev)
        def hook(i, xx, out):
            out = out.clone(); out[:, :4] -= Bt[i]; return out
        xs = []
        for b in range(0, len(EVAL), 32):
            s = EVAL[b:b + 32]
            xs.append(ddim_sample(q, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5, eps_hook=hook).cpu())
        x = torch.cat(xs)
        ref, base = E["x0fp"], E["x0q"]
        rel = lambda a: ((a - ref).flatten(1).pow(2).sum(1) / ref.flatten(1).pow(2).sum(1))
        rb, rt = rel(base), rel(x)
        g = torch.Generator().manual_seed(0)
        bs = []
        for _ in range(2000):
            ix = torch.randint(0, len(rb), (len(rb),), generator=g)
            bs.append(float(1 - rt[ix].mean() / rb[ix].mean()))
        res = {"base_rel_mse": float(rb.mean()), "bt_rel_mse": float(rt.mean()), "reduction": float(1 - rt.mean() / rb.mean()),
               "ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))], "seeds": [EVAL[0], EVAL[-1]]}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "bt_latent.json"), res)
    print(res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["traces", "audit", "predict", "bt"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"traces": lambda: traces(dev), "audit": audit, "predict": predict, "bt": lambda: bt(dev)}[a.cmd]()


if __name__ == "__main__":
    main()
