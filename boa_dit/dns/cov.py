"""C-DNS (results/dns/PROTOCOL.md §4 amendment): band-wise quantities, the propagation model with the denoiser response, the joint
covariance recursion (full) and the matched temporal-independence ablation, the §5 held-out gate, rho_i, and limits L1-L3. CPU only.
Run: python -m boa_dit.dns.cov
"""
import json
import math
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, atomic_write_json
from boa_dit.dns import port as P
from boa_dit.model import make_diffusion
from boa_dit.objective.d2 import band_masks

OUT = os.path.join(RESULTS, "dns")
CK = os.path.join(CKPT_DIR, "dns")
MASKS = [m.double() for m in band_masks(32)]
NB = len(MASKS)
DB = [4 * float(m.sum()) for m in MASKS]                                     # elements per band (4 channels x frequencies)
D = 4 * 32 * 32
W = [db / D for db in DB]


def coefs(tm):
    a, c, g = [], [], []
    for i, t in enumerate(tm):
        p = P.prev_of(tm, i)
        ab_t, ab_p = P.AB[t], (P.AB[p] if p >= 0 else 1.0)
        a.append(math.sqrt(ab_p / ab_t))
        c.append(math.sqrt(1 - ab_p) - math.sqrt(ab_p * (1 - ab_t) / ab_t))
        g.append(math.sqrt((1 - ab_p) / (1 - ab_t)))
    return np.array(a), np.array(c), np.array(g)


def bands(x):
    """x [S, T, 4, 32, 32] -> band components [B, S, T, 4, 32, 32] (float64, sum to x)."""
    F = torch.fft.fft2(x.double())
    return torch.stack([torch.fft.ifft2(F * m).real for m in MASKS])


def centre(x):
    return x - x.mean(1, keepdim=True)                                       # across seeds (dim 1 of [B, S, ...])


def inner(u, v):
    """per band, per step: sum over seeds and elements of u*v / (S * D_b)  -> [B, T]"""
    S = u.shape[1]
    return torch.stack([(u[b] * v[b]).flatten(2).sum(2).sum(0) / (S * DB[b]) for b in range(NB)])


def quantities(tr):
    """band-wise centred quantities from one trace set."""
    xf, xq, eq, efq, ef = tr["xf"], tr["xq"], tr["eq"], tr["efq"], tr["ef"]
    d = bands(xq - xf)                                                       # [B, S, 51, ...]
    xi = bands(eq - efq)                                                     # [B, S, 50, ...]
    De = bands(efq - ef)                                                     # denoiser response to the state error
    xF = bands(xf)
    dc, xic, Dc, xFc = centre(d), centre(xi), centre(De), centre(xF)
    q = {"Vd": inner(dc, dc), "Vxi": inner(xic, xic), "C": inner(dc[:, :, :50], xic), "X": inner(xFc, dc),
         "J": inner(Dc, dc[:, :, :50]) / inner(dc[:, :, :50], dc[:, :, :50]).clamp_min(1e-30),
         "F": inner(xic[:, :, 1:], xic[:, :, :-1]) / inner(xic[:, :, :-1], xic[:, :, :-1]).clamp_min(1e-30),
         "lag1_corr": inner(xic[:, :, 1:], xic[:, :, :-1]) / (inner(xic[:, :, 1:], xic[:, :, 1:]) * inner(xic[:, :, :-1], xic[:, :, :-1])).sqrt(),
         "md2": torch.stack([d[b].mean(0).pow(2).flatten(1).sum(1) / DB[b] for b in range(NB)]),      # squared mean (uncentred minus centred)
         "mxi2": torch.stack([xi[b].mean(0).pow(2).flatten(1).sum(1) / DB[b] for b in range(NB)]),
         "Vd_unc": inner(d, d), "Vxi_unc": inner(xi, xi)}
    q["J"][:, 0] = 0.0                                                       # d_0 = 0 (same noise): undefined, irrelevant (V_d,0 = 0)
    return q, (dc, xic, Dc)


def recursion(A, F, Vxi, c, full=True):
    """[B, T] coefficients -> predicted V_d [B, 51] and C [B, 50]."""
    Bn, T = Vxi.shape
    Vd, C = torch.zeros(Bn, T + 1, dtype=torch.float64), torch.zeros(Bn, T, dtype=torch.float64)
    for i in range(T):
        Vd[:, i + 1] = A[:, i] ** 2 * Vd[:, i] + 2 * A[:, i] * c[i] * C[:, i] + c[i] ** 2 * Vxi[:, i]
        if full and i + 1 < T:
            C[:, i + 1] = A[:, i] * F[:, i] * C[:, i] + c[i] * F[:, i] * Vxi[:, i]
    return Vd, C


def dV(A, Vd, C, Vxi, c, g):
    """per band one-step excess increment [B, T]."""
    cc, gg = torch.tensor(c), torch.tensor(g)
    return (A ** 2 - gg ** 2) * Vd[:, :-1] + 2 * A * cc * C + cc ** 2 * Vxi


def rho(A, Vd, C, Vxi, c, g):
    w = torch.tensor(W, dtype=torch.float64)[:, None]
    num = (w * dV(A, Vd, C, Vxi, c, g)).sum(0)
    den = torch.tensor(c) ** 2 * (w * Vxi).sum(0)
    return torch.clamp(num / den.clamp_min(1e-300), min=0.0), num / den.clamp_min(1e-300)


def score_a(Vpred, Vemp, floor=1e-8):
    m = Vemp > floor
    return float(torch.log(Vpred[m].clamp_min(1e-300) / Vemp[m]).abs().mean())


def main():
    tr = torch.load(os.path.join(CK, "cdns_traces.pt"))
    tm = P.tmap(make_diffusion(P.T))
    a, c, g = coefs(tm)
    qf, _ = quantities(tr["fit"])
    qh, (dch, xich, Dch) = quantities(tr["held"])
    A = torch.tensor(a)[None] + torch.tensor(c)[None] * qf["J"]
    Ff, Vxi = qf["F"], qf["Vxi"]
    Vd_full, C_full = recursion(A, Ff, Vxi, c, True)
    Vd_ind, C_ind = recursion(A, Ff, Vxi, c, False)
    # §5 gate (held-out): (a) V_d trajectory, (b) one-step increment
    sa_full, sa_ind = score_a(Vd_full[:, 1:], qh["Vd"][:, 1:]), score_a(Vd_ind[:, 1:], qh["Vd"][:, 1:])
    emp_dV = qh["Vd"][:, 1:] - torch.tensor(g) ** 2 * qh["Vd"][:, :-1]
    nae = lambda p: float((p - emp_dV).abs().sum() / emp_dV.abs().sum())
    sb_full, sb_ind = nae(dV(A, qh["Vd"], C_full, Vxi, c, g)), nae(dV(A, qh["Vd"], C_ind, Vxi, c, g))
    # seed bootstrap of the score difference (recompute empirical held-out V_d per resample)
    S = dch.shape[1]
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        idx = torch.as_tensor(rng.integers(0, S, S))
        dd = dch[:, idx]
        dd = dd - dd.mean(1, keepdim=True)
        Vemp = inner(dd, dd)[:, 1:]
        diffs.append(score_a(Vd_ind[:, 1:], Vemp) - score_a(Vd_full[:, 1:], Vemp))
    ci = [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))]
    passed = (sa_full <= 0.9 * sa_ind) and ci[0] > 0
    # innovation / closure assumptions on held-out (pooled correlations per band and step)
    nu = xich[:, :, 1:] - torch.stack([qf["F"][b][None, :, None, None, None] * xich[b, :, :-1] for b in range(NB)])   # nu_i, i = 0..48
    r = dch[:, :, 1:] - torch.stack([A[b][None, :, None, None, None] * dch[b, :, :-1] + torch.tensor(c)[None, :, None, None, None] * xich[b] for b in range(NB)])
    corr = lambda u, v: inner(u, v) / (inner(u, u) * inner(v, v)).sqrt().clamp_min(1e-300)
    tests = {"nu_vs_d": corr(nu, dch[:, :, :49]), "nu_vs_xi": corr(nu, xich[:, :, :-1]), "nu_vs_r": corr(nu, r[:, :, :49]),
             "r_vs_d": corr(r, dch[:, :, :50]), "r_vs_xi": corr(r, xich)}
    tests_s = {k: {"max_abs": float(v[:, 2:].abs().max()), "frac_steps_flagged": float((v[:, 2:].abs() > 0.1).float().mean()),
                   "per_band_mean_abs": v[:, 2:].abs().mean(1).tolist()} for k, v in tests.items()}
    Vr = inner(centre(r), centre(r))
    # rho (calibration fit set): full and ablation
    rf, rf_raw = rho(A, Vd_full, C_full, Vxi, c, g)
    ri, ri_raw = rho(A, Vd_ind, C_ind, Vxi, c, g)
    # limits
    L1 = rho(torch.tensor(g)[None].expand(NB, -1), *recursion(torch.tensor(g)[None].expand(NB, -1), torch.zeros_like(Ff), Vxi, c, False), Vxi, c, g)[0]
    st = json.load(open(os.path.join(OUT, "stats.json")))["steps"] if os.path.exists(os.path.join(OUT, "stats.json")) else None
    lim = {"L1_max_abs_rho_minus_1": float((L1 - 1).abs().max())}
    if st is not None:
        sd = P.schedule(st, tm)
        s1 = P.schedule([dict(s, residual_std=float(math.sqrt(r_) * s["residual_std"])) for s, r_ in zip(st, L1.tolist())], tm)
        lim["L1_max_abs_abar_q_diff_vs_dns"] = max(abs(x["abar_q"] - y["abar_q"]) for x, y in zip(sd, s1))
        lim["L1_tau_identical"] = all(x["tau"] == y["tau"] for x, y in zip(sd, s1))
        zero = [dict(s, residual_std=0.0, slope_k=[0.0] * 4, intercept=[0.0] * 4, uniform_scale=0.0) for s in st]
        s2 = P.schedule([dict(z, residual_std=float(math.sqrt(r_) * z["residual_std"])) for z, r_ in zip(zero, rf.tolist())], tm)
        lim["L2_identity"] = all(x["abar_q"] == x["abar_prev"] and x["C3sq"] == 1.0 for x in s2)
        s3 = P.schedule([dict(s, residual_std=0.0) for s in st], tm)
        lim["L3_rho0_no_deformation"] = all(x["abar_q"] == x["abar_prev"] for x in s3)
    res = {"bands_elements": DB, "w": W,
           "coefs": {"a": a.tolist(), "c": c.tolist(), "g": g.tolist()},
           "fit": {k: v.tolist() for k, v in qf.items()}, "held": {k: v.tolist() for k, v in qh.items()},
           "A": A.tolist(), "A_over_g": (A / torch.tensor(g).clamp_min(1e-300)).tolist(),
           "pred": {"Vd_full": Vd_full.tolist(), "Vd_ind": Vd_ind.tolist(), "C_full": C_full.tolist()},
           "Var_r_held": Vr.tolist(),
           "gate": {"score_a_full": sa_full, "score_a_ind": sa_ind, "rel_improvement": 1 - sa_full / sa_ind, "diff_ci": ci,
                    "score_b_full": sb_full, "score_b_ind": sb_ind, "pass": bool(passed)},
           "assumption_tests_held": tests_s,
           "rho_full": rf.tolist(), "rho_ind": ri.tolist(), "rho_full_unclamped": rf_raw.tolist(), "rho_ind_unclamped": ri_raw.tolist(),
           "rho_clamp_events": {"full": int((rf_raw < 0).sum()), "ind": int((ri_raw < 0).sum())},
           "limits": lim}
    atomic_write_json(os.path.join(OUT, "cov_fit.json"), res)
    print(json.dumps({"gate": res["gate"], "assumption_tests_held": {k: {kk: vv for kk, vv in v.items() if kk != "per_band_mean_abs"} for k, v in tests_s.items()},
                      "limits": lim, "rho_full(0,10,25,40,48)": [round(rf[i].item(), 4) for i in (0, 10, 25, 40, 48)],
                      "rho_ind(0,10,25,40,48)": [round(ri[i].item(), 4) for i in (0, 10, 25, 40, 48)],
                      "lag1_corr_band_mean": qf["lag1_corr"].mean(1).tolist()}, indent=1))


if __name__ == "__main__":
    main()
