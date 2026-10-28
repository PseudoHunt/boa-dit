"""Follow-up Experiment 1 (results/followup/PROTOCOL.md §1): sampler-side state correction on DiT heig-8.
  screen : 1A on the cached §2 traces (fit 7000-7063, exploratory held-out 7100-7163): arms I, G, P(/P-mix), M at τ = 30, 40, final
  live   : 1B once-per-trajectory intervention: placement on design seeds 8000-8031, confirmation on 8100-8163
Run: python -m boa_dit.followup.state_corr screen|live_design|live_confirm
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.f4 import STAGE, guard
from boa_dit.objective.d2 import band_masks

OUT = os.path.join(RESULTS, "followup", "e1")
CK = os.path.join(CKPT_DIR, "followup", "e1")
TRACES = os.path.join(CKPT_DIR, "observability", "traces_{}.pt")
PLACES = ["t30", "t40", "final"]
LOW = (0, 1)                                                                  # bands 1, 2 corrected; 3, 4 pass through
LAMS = [0.0, 1e-3, 1e-2, 1e-1, 1.0]
MASKS = [m.double() for m in band_masks(32)]


def split(x):
    """x [S, 4, 32, 32] -> list of 4 band components (float64); they sum to x exactly."""
    F = torch.fft.fft2(x.double())
    return [torch.fft.ifft2(F * m).real for m in MASKS]


def dc(x):
    return x.mean((-1, -2))                                                   # [S, 4]


def pooled_var(x):
    """across-seed variance per element, pooled over channels and positions."""
    return float(((x - x.mean(0, keepdim=True)) ** 2).mean(0).mean())


def state(R, place):
    return {"t30": R["xq"][:, 30], "t40": R["xq"][:, 40], "final": R["x0q"]}[place], \
           {"t30": R["xfp"][:, 30], "t40": R["xfp"][:, 40], "final": R["x0fp"]}[place]


# ---- fitting (calibration seeds only) ----
def fit_G(q, f):
    mq, mf = dc(q).mean(0), dc(f).mean(0)
    return {"mq": mq, "mf": mf, "g": (pooled_var(f) / pooled_var(q)) ** 0.5}


def fit_M(q, f):
    Q, Fb = split(q), split(f)
    out = {}
    for b in LOW:
        out[b] = {"g": (pooled_var(Fb[b]) / pooled_var(Q[b])) ** 0.5,
                  "mq": dc(Q[b]).mean(0) if b == 0 else torch.zeros(4, dtype=torch.float64),
                  "mf": dc(Fb[b]).mean(0) if b == 0 else torch.zeros(4, dtype=torch.float64)}
    return out


def _centre(Qb, Fb, b):
    if b != 0:
        return Qb, Fb, torch.zeros(4, dtype=torch.float64), torch.zeros(4, dtype=torch.float64)
    mq, mf = Qb.mean((0, 2, 3)), Fb.mean((0, 2, 3))                          # per-channel means (the DC bias)
    return Qb - mq.view(1, 4, 1, 1), Fb - mf.view(1, 4, 1, 1), mq, mf


def fit_P_band(Qb, Fb, b, lam, mix):
    """ridge toward identity, scale-free: scalar a = (cov/v + λ)/(1 + λ); mix M = (Cfq + λvI)(Cqq + λvI)^-1."""
    qc, fc, mq, mf = _centre(Qb, Fb, b)
    if not mix:
        v, cv = float((qc * qc).mean()), float((qc * fc).mean())
        A = torch.eye(4, dtype=torch.float64) * ((cv / v + lam) / (1 + lam))
    else:
        qv, fv = qc.transpose(0, 1).reshape(4, -1), fc.transpose(0, 1).reshape(4, -1)
        n = qv.shape[1]
        Cqq, Cfq = qv @ qv.T / n, fv @ qv.T / n
        v = float(torch.diagonal(Cqq).mean())
        I = torch.eye(4, dtype=torch.float64)
        A = (Cfq + lam * v * I) @ torch.linalg.inv(Cqq + lam * v * I)
    return {"A": A, "mq": mq, "mf": mf}


def apply_band(Xb, p):
    """x' = A (x − mq) + mf, channel-mixing A [4,4]."""
    return torch.einsum("ij,sjhw->sihw", p["A"], Xb - p["mq"].view(1, 4, 1, 1)) + p["mf"].view(1, 4, 1, 1)


def folds(n, k=4):
    idx = np.arange(n)
    return [(np.setdiff1d(idx, idx[i::k]), idx[i::k]) for i in range(k)]


def cv_P(q, f, mix):
    """per band: λ by 4-fold CV over seeds; returns (λ, per-fold held-out MSE at that λ)."""
    Q, Fb = split(q), split(f)
    res = {}
    for b in LOW:
        scores = {}
        for lam in LAMS:
            m = []
            for tr, te in folds(len(q)):
                p = fit_P_band(Q[b][tr], Fb[b][tr], b, lam, mix)
                m.append(float(((apply_band(Q[b][te], p) - Fb[b][te]) ** 2).mean()))
            scores[lam] = m
        lam = min(LAMS, key=lambda l: np.mean(scores[l]))
        res[b] = {"lam": lam, "fold_mse": scores[lam], "all": {str(k): float(np.mean(v)) for k, v in scores.items()}}
    return res


def fit_P(q, f, cv, mix):
    Q, Fb = split(q), split(f)
    return {b: fit_P_band(Q[b], Fb[b], b, cv[b]["lam"], mix) for b in LOW}


# ---- application ----
def apply(arm, params, x):
    x = x.double()
    if arm == "I":
        return x
    if arm == "G":
        return params["mf"].view(1, 4, 1, 1) + params["g"] * (x - params["mq"].view(1, 4, 1, 1))
    B = split(x)
    for b in LOW:
        p = params[b]
        if arm == "M":
            B[b] = p["g"] * (B[b] - p["mq"].view(1, 4, 1, 1)) + p["mf"].view(1, 4, 1, 1)
        else:
            B[b] = apply_band(B[b], p)
    return sum(B)


def boot_ci(v, n=2000):
    v = np.asarray(v)
    g = np.random.default_rng(0)
    bs = [v[g.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def metrics(xc, f, x_id=None):
    """held-out: paired MSE (total, per band), band DC-mean mismatch, variance ratio, tr Cov(x', d') per band."""
    d = xc - f
    per_seed = (d ** 2).flatten(1).sum(1) / (f.double() ** 2).flatten(1).sum(1)
    C, Fb, D = split(xc), split(f), split(d)
    out = {"rel_mse": float(per_seed.mean()), "rel_mse_per_seed": per_seed.tolist(), "bands": []}
    for b in range(4):
        cc, ff, dd = C[b], Fb[b], D[b]
        cov = float(((cc - cc.mean(0)) * (dd - dd.mean(0))).mean(0).sum())
        out["bands"].append({"mse_rel": float((dd ** 2).sum() / (f.double() ** 2).sum()),
                             "var_ratio": pooled_var(cc) / pooled_var(ff),
                             "var_ratio_ci": boot_var_ratio(cc, ff),
                             "dc_mean_mismatch": float((dc(cc).mean(0) - dc(ff).mean(0)).abs().mean()) if b == 0 else None,
                             "trcov_x_d_over_trcov_fp": cov / float(((ff - ff.mean(0)) ** 2).mean(0).sum())})
    if x_id is not None:
        base = ((x_id - f) ** 2).flatten(1).sum(1) / (f.double() ** 2).flatten(1).sum(1)
        diff = (per_seed - base).numpy()
        out["vs_identity"] = {"rel_change": float(diff.mean() / base.mean()), "ci": [c / float(base.mean()) for c in boot_ci(diff)]}
    return out


def boot_var_ratio(c, f, n=500):
    g = np.random.default_rng(0)
    r = []
    for _ in range(n):
        i = torch.as_tensor(g.integers(0, len(c), len(c)))
        r.append(pooled_var(c[i]) / pooled_var(f[i]))
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]


def coef_summary(arm, p):
    if arm == "G":
        return {"g": p["g"], "dc_shift": (p["mf"] - p["mq"]).tolist()}
    if arm == "M":
        return {f"band{b + 1}": {"g": p[b]["g"], "dc_shift": (p[b]["mf"] - p[b]["mq"]).tolist()} for b in LOW}
    return {f"band{b + 1}": {"A": p[b]["A"].tolist(), "dc_shift": (p[b]["mf"] - p[b]["A"] @ p[b]["mq"]).tolist()} for b in LOW}


def fit_all(q, f):
    cvs, cvm = cv_P(q, f, False), cv_P(q, f, True)
    keep_mix = all(sum(m < s for m, s in zip(cvm[b]["fold_mse"], cvs[b]["fold_mse"])) >= 3 and
                   np.mean(cvm[b]["fold_mse"]) < np.mean(cvs[b]["fold_mse"]) for b in LOW)
    return {"G": fit_G(q, f), "P": fit_P(q, f, cvs, False), "Pmix": fit_P(q, f, cvm, True), "M": fit_M(q, f)}, \
           {"cv_scalar": cvs, "cv_mix": cvm, "mix_retained": bool(keep_mix)}


@torch.no_grad()
def screen():
    os.makedirs(OUT, exist_ok=True)
    with Run(STAGE, "e1_trace_screen", "cpu", {"fit": "traces_cal 7000-7063", "heldout": "traces_eval 7100-7163 (inspected; exploratory)"}) as run:
        cal, ev = torch.load(TRACES.format("cal")), torch.load(TRACES.format("eval"))
        res = {"placements": {}}
        for place in PLACES:
            q, f = (t.double() for t in state(cal, place))
            qe, fe = (t.double() for t in state(ev, place))
            params, cvinfo = fit_all(q, f)
            P_arm = "Pmix" if cvinfo["mix_retained"] else "P"
            r = {"cv": cvinfo, "P_arm_retained": P_arm, "arms": {}}
            stab = {a: [] for a in ("G", "P", "Pmix", "M")}
            for tr, _ in folds(len(q)):                                       # coefficient stability over calibration splits
                pk, _ = fit_all(q[tr], f[tr])
                for a in stab:
                    stab[a].append(coef_summary(a, pk[a]))
            for arm in ("I", "G", "P", "Pmix", "M"):
                xc = apply(arm, params.get(arm), qe)
                m = metrics(xc, fe, None if arm == "I" else qe)
                m["coef"] = None if arm == "I" else coef_summary(arm, params[arm])
                m["coef_folds"] = stab.get(arm)
                r["arms"][arm] = m
            res["placements"][place] = r
            line = " | ".join(f"{a} {r['arms'][a]['rel_mse']:.5f}" + (f" ({100 * r['arms'][a]['vs_identity']['rel_change']:+.1f}%)" if a != "I" else "") +
                              f" var1 {r['arms'][a]['bands'][0]['var_ratio']:.3f}" for a in r["arms"])
            print(f"  {place}: {line}; P-mix retained: {cvinfo['mix_retained']}", flush=True)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "screen.json"), res)
    torch.save({p: fit_all(*(t.double() for t in state(torch.load(TRACES.format("cal")), p)))[0] for p in PLACES},
               os.path.join(OUT, "params_cal.pt"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["screen", "live_design", "live_confirm"])
    a = ap.parse_args()
    if a.cmd == "screen":
        screen()
    else:
        from boa_dit.followup import state_live
        state_live.run(a.cmd)


if __name__ == "__main__":
    main()
