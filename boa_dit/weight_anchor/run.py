"""Measured weight-error anchoring pilot (results/weight_anchor/PROTOCOL.md). Stage tag wa1.
Bed: DiT-XL/2 256, heig-8 W4A16 (sampler_corr.run.heig), DDIM-50, cfg 1.5, fp16 autocast, batch 32. Errors and corrections act on the
sampler-consumed eps channels 0-3 only.
CPU stages : provenance lag sim                (cached observability traces, FP trajectory; Stage-1 W4-trajectory diagnostic)
GPU stages : checks live_dev live_conf          (live stages only if the sim gate passes; live_conf only if s = 0 does not win on dev)
Run: python -m boa_dit.weight_anchor.run <stage>
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import NULL_CLASS, decode, labels_for_seeds, load_dit, load_vae, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks
from boa_dit.sample import CTX
from boa_dit.sampler_corr.run import CARRY, HEIG, heig

STAGE = "wa1"
OUT = os.path.join(RESULTS, "weight_anchor")
CK = os.path.join(CKPT_DIR, "weight_anchor")
CAP_GPU_H = 1.0
T, NB, CFG, BS = 50, 5, 1.5, 32
LAGS = (1, 2, 4, 8)
KS = (4, 8)
NPERM = 20
SPLITS = {"dev": list(range(92000, 92032)), "conf": list(range(92100, 92164)), "check": list(range(7100, 7132))}
TRACES = {"cal": os.path.join(CKPT_DIR, "observability", "traces_cal.pt"), "eval": os.path.join(CKPT_DIR, "observability", "traces_eval.pt")}
STAGE1 = os.path.join(CKPT_DIR, "temporal", "stage1_delta_heig8.pt")
HASHED = {"dit": os.path.join(CKPT_DIR, "DiT-XL-2-256x256.pt"), "heig8_model": os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"), "heig8_carry": CARRY,
          "traces_cal": TRACES["cal"], "traces_eval": TRACES["eval"], "stage1_delta": STAGE1}


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


def guard(label, est_h):
    s = spent()
    ok = s + est_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "log": []}
    b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h, "ok": ok})
    jdump("budget.json", b)
    print(f"[wa1 budget] {label}: spent {s:.3f} GPU-h, estimate {est_h:.2f} h -> {'OK' if ok else 'STOP (cap)'}", flush=True)
    if not ok:
        sys.exit(5)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def provenance():
    out = {k: {"path": os.path.relpath(p, os.path.dirname(CKPT_DIR)), "sha256": sha256(p)} for k, p in HASHED.items()}
    out["settings"] = {"sampler": "DDIM-50 eta 0", "cfg": CFG, "cfg_channels": "eps 0-2 guided, eps 3 from cond half", "consumed": "channels 0-3",
                       "precision": "fp16 autocast forward, fp32 output/sampler", "batch": BS}
    jdump("provenance.json", out)
    print(json.dumps(out, indent=1))


# ------------------------------------------------------------------------------------------------------------- cached analysis
def perms(n):
    out = []
    for j in range(NPERM):
        g = np.random.default_rng(1000 + j)
        while True:
            p = g.permutation(n)
            if not np.any(p == np.arange(n)):
                break
        out.append(p)
    return out


def errors(name):
    if name == "w4traj":
        return torch.load(STAGE1, map_location="cpu")["delta"].double()
    R = torch.load(TRACES[name], map_location="cpu")
    return (R["epsq"] - R["eps"]).double()                                          # [S, T, 4, 32, 32]


def bands(E):
    """whole + 4 radial bands: list of [S, T, 4, 32, 32] (orthogonal components)."""
    F = torch.fft.fft2(E)
    return [E] + [torch.fft.ifft2(F * m.double()).real for m in band_masks()]


def grams(E, P=None, kmax=8):
    """xy[s, a, k] = <e_a, e_{a+k}>, n[s, i] = ||e_i||^2; with permutations also the shuffled xy and xx."""
    S = E.shape[0]
    Ef = E.flatten(2)
    n = Ef.pow(2).sum(2)
    xy = torch.full((S, T, kmax + 1), float("nan"), dtype=torch.float64)
    for k in range(1, kmax + 1):
        xy[:, :T - k, k] = (Ef[:, :T - k] * Ef[:, k:]).sum(2)
    out = {"n": n, "xy": xy}
    if P is not None:
        sx = torch.full((len(P), S, T, kmax + 1), float("nan"), dtype=torch.float64)
        for j, p in enumerate(P):
            Ep = Ef[torch.as_tensor(p)]
            for k in range(1, kmax + 1):
                sx[j, :, :T - k, k] = (Ep[:, :T - k] * Ef[:, k:]).sum(2)
        out["sxy"], out["sxx"] = sx, torch.stack([n[torch.as_tensor(p)] for p in P])
    return out


def fit_beta(G, cells):
    """cells: list of (a, k). beta = sum xy / sum xx over images and cells."""
    num = sum(float(G["xy"][:, a, k].sum()) for a, k in cells)
    den = sum(float(G["n"][:, a].sum()) for a, k in cells)
    return num / den


def lag_cells(b, k):
    return [(a, k) for a in range(b * 10, b * 10 + 10) if a + k <= T - 1]


def per_image(G, cells, beta, arm, j=None):
    """per-image (numerator, denominator) of the energy-remaining ratio over the given (a, k) cells."""
    num = torch.zeros(G["n"].shape[0], dtype=torch.float64)
    den = torch.zeros_like(num)
    for a, k in cells:
        yy = G["n"][:, a + k]
        if arm == "zero":
            r = yy
        elif arm in ("unit", "cal"):
            bb = 1.0 if arm == "unit" else beta[(a, k)]
            r = yy - 2 * bb * G["xy"][:, a, k] + bb * bb * G["n"][:, a]
        else:                                                                      # shuffled anchor, permutation j
            bb = beta[(a, k)]
            r = yy - 2 * bb * G["sxy"][j, :, a, k] + bb * bb * G["sxx"][j, :, a]
        num += r
        den += yy
    return num.numpy(), den.numpy()


BOOT = None


def boot_idx(n):
    global BOOT
    if BOOT is None or BOOT[0] != n:
        g = np.random.default_rng(0)
        BOOT = (n, [g.integers(0, n, n) for _ in range(2000)])
    return BOOT[1]


def ratio_ci(num, den):
    ix = boot_idx(len(num))
    v = [num[i].sum() / den[i].sum() for i in ix]
    return {"v": float(num.sum() / den.sum()), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


def diff_ci(num_a, num_b, den):
    """(sum num_a - sum num_b) / sum den, paired image bootstrap."""
    ix = boot_idx(len(den))
    v = [(num_a[i].sum() - num_b[i].sum()) / den[i].sum() for i in ix]
    return {"v": float((num_a.sum() - num_b.sum()) / den.sum()), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]}


def arm_block(G, cells, beta, shuffle=True):
    out = {}
    nums = {}
    for arm in ("zero", "unit", "cal"):
        num, den = per_image(G, cells, beta, arm)
        nums[arm] = num
        out[arm] = ratio_ci(num, den)
    out["cal_wins_vs_zero"] = int(np.sum(nums["cal"] < nums["zero"]))
    out["cal_wins_vs_unit"] = int(np.sum(nums["cal"] < nums["unit"]))
    out["cal_minus_unit"] = diff_ci(nums["cal"], nums["unit"], den)
    if shuffle and "sxy" in G:
        sh = [float(per_image(G, cells, beta, "shuf", j)[0].sum() / den.sum()) for j in range(NPERM)]
        out["shuffle"] = {"mean": float(np.mean(sh)), "min": float(np.min(sh)), "max": float(np.max(sh))}
        out["cal_wins_vs_shuffle_mean_per_image"] = int(np.sum(nums["cal"] < np.mean([per_image(G, cells, beta, "shuf", j)[0] for j in range(NPERM)], 0)))
    out["n_images"] = len(den)
    return out


def cosnorm(G, cells):
    c, r = [], []
    for a, k in cells:
        xx, yy, xy = G["n"][:, a], G["n"][:, a + k], G["xy"][:, a, k]
        c.append(xy / (xx * yy).sqrt())
        r.append((yy / xx).sqrt())
    c, r = torch.stack(c, 1).mean(1), torch.stack(r, 1).mean(1)                     # per image: mean over cells
    return {"cos_mean": float(c.mean()), "cos_image_min": float(c.min()), "norm_ratio_mean": float(r.mean())}


@torch.no_grad()
def lag():
    Ecal, Eev, Ew = errors("cal"), errors("eval"), errors("w4traj")
    Bc, Be, Bw = bands(Ecal), bands(Eev), bands(Ew)
    names = ["whole", "band1", "band2", "band3", "band4"]
    Gc = [grams(x) for x in Bc]
    Ge = [grams(x, perms(Eev.shape[0])) for x in Be]
    Gw = [grams(x, perms(Ew.shape[0])) for x in Bw]
    beta_u = {(b, k): fit_beta(Gc[0], lag_cells(b, k)) for b in range(NB) for k in LAGS}
    beta_d = {key: float(min(max(v, 0.0), 1.5)) for key, v in beta_u.items()}
    cellbeta = lambda: {(a, k): beta_d[(a // 10, k)] for a in range(T) for k in LAGS}
    cb = cellbeta()
    res = {"beta_unconstrained": {f"b{b}_k{k}": beta_u[(b, k)] for b in range(NB) for k in LAGS},
           "beta_deployed": {f"b{b}_k{k}": beta_d[(b, k)] for b in range(NB) for k in LAGS},
           "beta_band_unconstrained_diag": {names[i]: {f"b{b}_k{k}": fit_beta(Gc[i], lag_cells(b, k)) for b in range(NB) for k in LAGS} for i in range(1, 5)},
           "energy_frac_by_band_eval": [float(Be[i].pow(2).sum() / Be[0].pow(2).sum()) for i in range(1, 5)],
           "eval": {}, "w4traj_diag": {}, "harmful_eval": []}
    for split, GG, key in (("eval", Ge, "eval"), ("w4traj", Gw, "w4traj_diag")):
        for bi, nm in enumerate(names):
            r = {}
            for k in LAGS:
                allc = [c for b in range(NB) for c in lag_cells(b, k)]
                r[f"k{k}"] = {"pooled": arm_block(GG[bi], allc, cb, shuffle=(bi == 0)), "cosnorm": cosnorm(GG[bi], allc),
                              "by_bucket": {f"b{b}": {**arm_block(GG[bi], lag_cells(b, k), cb, shuffle=(bi == 0)), **cosnorm(GG[bi], lag_cells(b, k))} for b in range(NB)}}
                if split == "eval" and bi == 0:
                    for b in range(NB):
                        x = r[f"k{k}"]["by_bucket"][f"b{b}"]
                        if x["cal"]["v"] > 1 or x["unit"]["v"] > 1:
                            res["harmful_eval"].append({"bucket": b, "lag": k, "unit": x["unit"]["v"], "cal": x["cal"]["v"]})
            res[key][nm] = r
            print(f"[lag] {split} {nm}: " + "; ".join(f"k{k} zero/unit/cal {r[f'k{k}']['pooled']['unit']['v']:.3f}/{r[f'k{k}']['pooled']['cal']['v']:.3f} cos {r[f'k{k}']['cosnorm']['cos_mean']:.3f}" for k in LAGS), flush=True)
    jdump("lag.json", res)
    plots(res)
    return res


def plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for split, key, ls in (("eval (FP traj.)", "eval", "-"), ("Stage-1 (W4 traj.)", "w4traj_diag", "--")):
        w = res[key]["whole"]
        for arm, col in (("unit", "C1"), ("cal", "C0")):
            v = [w[f"k{k}"]["pooled"][arm]["v"] for k in LAGS]
            lo = [w[f"k{k}"]["pooled"][arm]["ci"][0] for k in LAGS]
            hi = [w[f"k{k}"]["pooled"][arm]["ci"][1] for k in LAGS]
            ax[0].errorbar(LAGS, v, yerr=[np.subtract(v, lo), np.subtract(hi, v)], fmt="o" + ls, color=col, capsize=3, label=f"{arm}, {split}")
        if "shuffle" in w["k1"]["pooled"]:
            ax[0].plot(LAGS, [w[f"k{k}"]["pooled"]["shuffle"]["mean"] for k in LAGS], "x" + ls, color="C3", label=f"shuffled anchor, {split}")
        ax[1].plot(LAGS, [w[f"k{k}"]["cosnorm"]["cos_mean"] for k in LAGS], "o" + ls, color="C0", label=f"cos, {split}")
        ax[1].plot(LAGS, [w[f"k{k}"]["cosnorm"]["norm_ratio_mean"] for k in LAGS], "s" + ls, color="C2", label=f"norm ratio, {split}")
    ax[0].axhline(1, color="k", lw=0.8)
    ax[0].set(xscale="log", xticks=LAGS, xticklabels=LAGS, xlabel="lag k (calls)", ylabel="error energy remaining", title="whole output, pooled over anchors")
    ax[1].set(xscale="log", xticks=LAGS, xticklabels=LAGS, xlabel="lag k", title="cos(e_a, e_a+k) and |e_a+k|/|e_a|")
    for b in range(NB):
        ax[2].plot(LAGS, [res["eval"]["whole"][f"k{k}"]["by_bucket"][f"b{b}"]["cal"]["v"] for k in LAGS], "o-", label=f"anchors {10 * b}-{10 * b + 9}")
    ax[2].axhline(1, color="k", lw=0.8)
    ax[2].set(xscale="log", xticks=LAGS, xticklabels=LAGS, xlabel="lag k", title="calibrated reuse by anchor bucket (eval)")
    for a in ax:
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lag_whole.png"), dpi=110)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i in range(1, 5):
        w = res["eval"][f"band{i}"]
        ax.plot(LAGS, [w[f"k{k}"]["pooled"]["cal"]["v"] for k in LAGS], "o-", label=f"band {i} ({100 * res['energy_frac_by_band_eval'][i - 1]:.0f}% energy)")
    ax.axhline(1, color="k", lw=0.8)
    ax.set(xscale="log", xticks=LAGS, xticklabels=LAGS, xlabel="lag k", ylabel="energy remaining (calibrated, whole-output beta)", title="by radial band (eval)")
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lag_bands.png"), dpi=110)


def sched(K):
    """(i, anchor a, age j) for every non-anchor call."""
    return [(i, (i // K) * K, i % K) for i in range(T) if i % K]


def sim_arms(E, betaK, K, perm=None):
    """per-image non-anchor energy (anchors-only, reuse), temporal-sum energy (plain, anchors-only, reuse)."""
    S = E.shape[0]
    Ea = E if perm is None else E[torch.as_tensor(perm)]
    r_anchor = E.clone()
    r_anchor[:, ::K] = 0
    r_reuse = r_anchor.clone()
    for i, a, j in sched(K):
        r_reuse[:, i] = E[:, i] - betaK[(a // 10, j)] * Ea[:, a]
    na = [i for i in range(T) if i % K]
    f = lambda R: R[:, na].flatten(1).pow(2).sum(1).numpy()
    ts = lambda R: R.sum(1).flatten(1).pow(2).sum(1).numpy()
    return {"nonanchor_anchor": f(r_anchor), "nonanchor_reuse": f(r_reuse), "ts_plain": ts(E), "ts_anchor": ts(r_anchor), "ts_reuse": ts(r_reuse),
            "step_plain": E.flatten(1).pow(2).sum(1).numpy(), "step_anchor": r_anchor.flatten(1).pow(2).sum(1).numpy(), "step_reuse": r_reuse.flatten(1).pow(2).sum(1).numpy()}


@torch.no_grad()
def sim():
    Ecal, Eev, Ew = errors("cal"), errors("eval"), errors("w4traj")
    Gc = grams(Ecal)
    res = {"definition": "non-anchor reduction = 1 - sum_nonanchor ||e_i - beta e_a||^2 / sum_nonanchor ||e_i||^2 (reuse vs FP anchors only); "
                         "temporal-sum energy = sum_images ||sum_i r_i||^2. Neither is an endpoint metric.", "K": {}}
    for K in KS:
        cells = {(b, j): [(a, j) for a in range(0, T, K) if a // 10 == b and a + j <= T - 1] for b in range(NB) for j in range(1, K)}
        bu = {key: (fit_beta(Gc, c) if c else None) for key, c in cells.items()}
        bd = {key: (float(min(max(v, 0.0), 1.5)) if v is not None else 0.0) for key, v in bu.items()}
        rK = {"anchors": list(range(0, T, K)), "n_fp_calls": len(range(0, T, K)),
              "beta_unconstrained": {f"b{b}_j{j}": bu[(b, j)] for b, j in bu}, "beta_deployed": {f"b{b}_j{j}": bd[(b, j)] for b, j in bd}}
        for split, E in (("eval", Eev), ("w4traj_diag", Ew)):
            A = sim_arms(E, bd, K)
            ix = boot_idx(len(A["nonanchor_anchor"]))
            red = lambda i=None: 1 - (A["nonanchor_reuse"][i].sum() / A["nonanchor_anchor"][i].sum() if i is not None else A["nonanchor_reuse"].sum() / A["nonanchor_anchor"].sum())
            v = [red(i) for i in ix]
            tsr = lambda i=None: 1 - (A["ts_reuse"][i].sum() / A["ts_anchor"][i].sum() if i is not None else A["ts_reuse"].sum() / A["ts_anchor"].sum())
            vt = [tsr(i) for i in ix]
            P = perms(E.shape[0])
            sh = [sim_arms(E, bd, K, p) for p in P]
            r = {"nonanchor_reduction": {"v": float(red()), "ci": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]},
                 "nonanchor_wins": int(np.sum(A["nonanchor_reuse"] < A["nonanchor_anchor"])),
                 "temporal_sum_reduction_vs_anchor": {"v": float(tsr()), "ci": [float(np.percentile(vt, 2.5)), float(np.percentile(vt, 97.5))]},
                 "temporal_sum_energy": {k: float(A[k].sum()) for k in ("ts_plain", "ts_anchor", "ts_reuse")},
                 "per_step_energy": {k: float(A[k].sum()) for k in ("step_plain", "step_anchor", "step_reuse")},
                 "anchor_only_vs_plain_step_reduction": float(1 - A["step_anchor"].sum() / A["step_plain"].sum()),
                 "shuffled_nonanchor_reduction": {"mean": float(np.mean([1 - s["nonanchor_reuse"].sum() / s["nonanchor_anchor"].sum() for s in sh])),
                                                  "max": float(np.max([1 - s["nonanchor_reuse"].sum() / s["nonanchor_anchor"].sum() for s in sh]))},
                 "shuffled_temporal_sum_reduction_mean": float(np.mean([1 - s["ts_reuse"].sum() / s["ts_anchor"].sum() for s in sh])), "n_images": E.shape[0]}
            if split == "eval":
                r["gate_pass"] = bool(r["nonanchor_reduction"]["v"] >= 0.20 and r["nonanchor_reduction"]["ci"][0] > 0 and r["temporal_sum_reduction_vs_anchor"]["v"] > 0)
            rK[split] = r
            print(f"[sim] K={K} {split}: non-anchor reduction {100 * r['nonanchor_reduction']['v']:+.1f}% [{100 * r['nonanchor_reduction']['ci'][0]:+.1f}, "
                  f"{100 * r['nonanchor_reduction']['ci'][1]:+.1f}], temporal-sum {100 * r['temporal_sum_reduction_vs_anchor']['v']:+.1f}%, "
                  f"shuffled {100 * r['shuffled_nonanchor_reduction']['mean']:+.1f}%" + (f", gate {r['gate_pass']}" if "gate_pass" in r else ""), flush=True)
        res["K"][str(K)] = rK
    passing = [K for K in KS if res["K"][str(K)]["eval"]["gate_pass"]]
    if not passing:
        sel = None
    elif len(passing) == 1:
        sel = passing[0]
    else:
        r4, r8 = (res["K"][str(K)]["eval"]["nonanchor_reduction"]["v"] for K in KS)
        sel = 8 if abs(r4 - r8) < 0.02 else (4 if r4 > r8 else 8)
    res["gate_pass"], res["selected_K"] = bool(passing), sel
    res["frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    jdump("sim.json", res)
    co = {"lag_beta_deployed": (jload("lag.json") or {}).get("beta_deployed"), "selected_K": sel, "gate_pass": bool(passing),
          "beta_K_deployed": res["K"][str(sel)]["beta_deployed"] if sel else None, "s": None, "frozen_utc": res["frozen_utc"]}
    jdump("coefficients.json", co)
    print(f"[sim] gate pass: {bool(passing)}; selected K: {sel}", flush=True)


# ------------------------------------------------------------------------------------------------------------------------ live
def guided(model, xx, ts, y, n, amp=True):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(torch.cat([xx, xx]), torch.cat([ts, ts]), y).float()
    eps, rest = out[:, :3], out[:, 3:]
    c, u = eps[:n], eps[n:]
    return torch.cat([u + CFG * (c - u), rest[:n]], dim=1)


@torch.no_grad()
def live_sample(q, fp, d, noise, labels, mode, K=None, beta=None, s=0.0, record=()):
    """mode: 'w4' | 'fp' | 'anchor'. Same op order as sample.ddim_sample. Returns x0, counts, and anchor errors at the recorded calls."""
    x = noise.float().clone()
    n = len(x)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    cnt = {"w4": 0, "fp": 0}
    rec, st = {}, {"err": None, "a": None}
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            if mode == "w4" or (mode == "anchor" and _i % K):
                res = guided(q, xx, ts, y, n)
                cnt["w4"] += 1
                if mode == "anchor" and s != 0.0:
                    res = res.clone()
                    res[:, :4] -= (s * beta[(st["a"] // 10, _i - st["a"])]) * st["err"]
                return res
            res = guided(fp, xx, ts, y, n)
            cnt["fp"] += 1
            if mode == "anchor" and (s != 0.0 or _i in record):
                rq = guided(q, xx, ts, y, n)
                cnt["w4"] += 1
                st["err"], st["a"] = rq[:, :4] - res[:, :4], _i
                if _i in record:
                    rec[_i] = st["err"].cpu()
            return res
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt, rec


def models(dev):
    fp = load_dit(dev)
    q, _ = heig(dev)
    return fp, q


def beta_table(K):
    co = jload("coefficients.json")
    return {(int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): v for k, v in co["beta_K_deployed"].items()}


def model_bytes(m):
    return int(sum(p.numel() * p.element_size() for p in m.parameters()) + sum(b.numel() * b.element_size() for b in m.buffers()))


@torch.no_grad()
def checks(dev):
    guard("checks", 0.08)
    with Run(STAGE, "checks", dev, {"seeds": [SPLITS["check"][0], SPLITS["check"][-1]]}) as run:
        d = make_diffusion(T)
        fp, q = models(dev)
        R = torch.load(TRACES["eval"], map_location="cpu")
        s = SPLITS["check"]
        assert R["seeds"][:BS] == s
        nz, lab = noise_for_seeds(s, dev), labels_for_seeds(s)
        t0 = time.time()
        x1, c1, _ = live_sample(q, fp, d, nz, lab, "w4")
        tw = time.time() - t0
        t0 = time.time()
        x2, c2, rec = live_sample(q, fp, d, nz, lab, "anchor", K=1, beta=None, s=0.0, record=(0, 1))
        ta = time.time() - t0
        e_cache = (R["epsq"][:BS, :2] - R["eps"][:BS, :2])
        out = {"C1_w4_harness_equals_cached_x0q": bool(torch.equal(x1.cpu(), R["x0q"][:BS])), "C1_max_abs": float((x1.cpu() - R["x0q"][:BS]).abs().max()),
               "C2_K1_equals_cached_x0fp": bool(torch.equal(x2.cpu(), R["x0fp"][:BS])), "C2_max_abs": float((x2.cpu() - R["x0fp"][:BS]).abs().max()),
               "C3_anchor_error_equals_cached": [bool(torch.equal(rec[i], e_cache[:, i])) for i in (0, 1)],
               "C3_max_abs": [float((rec[i] - e_cache[:, i]).abs().max()) for i in (0, 1)],
               "counts": {"w4": c1, "K1": c2}, "wall_s_w4_run": tw, "wall_s_K1_run_plus_2_w4": ta}
        # latency per forward (batch 32, CFG-doubled 64), CUDA events
        xx = nz.float()
        ts = torch.full((BS,), 500, device=dev, dtype=torch.long)
        y = torch.cat([torch.as_tensor(lab, device=dev), torch.full((BS,), NULL_CLASS, device=dev, dtype=torch.long)])
        lat = {}
        for nm, m in (("fp", fp), ("w4_sim", q)):
            for _ in range(3):
                guided(m, xx, ts, y, BS)
            ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(10)]
            for a, b in ev:
                a.record(); guided(m, xx, ts, y, BS); b.record()
            torch.cuda.synchronize()
            v = [a.elapsed_time(b) for a, b in ev]
            lat[nm] = {"ms_median": float(np.median(v)), "ms_min": float(np.min(v))}
        out["forward_latency_batch32_cfg"] = lat
        out["resident_model_bytes"] = {"fp": model_bytes(fp), "w4_sim": model_bytes(q)}
        out["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated())
        out["pass"] = bool(out["C1_w4_harness_equals_cached_x0q"] and out["C2_K1_equals_cached_x0fp"])
        out.update(run.record())
    jdump("checks.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "config"}, indent=1), flush=True)


def rel(x, ref):
    return ((x - ref).flatten(1).pow(2).sum(1) / ref.flatten(1).pow(2).sum(1)).tolist()


def boot(a, b, ratio, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    g = np.random.default_rng(0)
    idx = [g.integers(0, len(a), len(a)) for _ in range(n)]
    if ratio:
        d = [a[i].mean() / b[i].mean() - 1 for i in idx]
        return {"rel_change": float(a.mean() / b.mean() - 1), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    d = [np.mean(a[i] - b[i]) for i in idx]
    return {"diff": float(np.mean(a - b)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


@torch.no_grad()
def live(dev, split):
    co = jload("coefficients.json")
    ch = jload("checks.json")
    if not (co and co["gate_pass"]):
        print("[wa1] sim gate did not pass: live pilot skipped", flush=True)
        return
    if not (ch and ch["pass"]):
        print("[wa1] live correctness checks did not pass: live pilot blocked", flush=True)
        sys.exit(6)
    K = co["selected_K"]
    beta = beta_table(K)
    if split == "dev":
        arms = {"w4": ("w4", 0.0), "anchor_only": ("anchor", 0.0), "reuse_s0.5": ("anchor", 0.5), "reuse_s1": ("anchor", 1.0)}
    else:
        dv = jload("live_dev.json")
        if not dv or dv["selected_s"] in (None, 0.0):
            print("[wa1] s = 0 won on development (or dev not run): confirmation not opened", flush=True)
            return
        arms = {"w4": ("w4", 0.0), "anchor_only": ("anchor", 0.0), "reuse": ("anchor", float(dv["selected_s"]))}
    seeds = SPLITS[split]
    nb = len(seeds) // BS
    per_batch_h = (ch["wall_s_w4_run"] * (1 + len(arms)) + 60) / 3600
    guard(f"live_{split}", per_batch_h * nb + 0.03)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, f"live_{split}", dev, {"K": K, "arms": arms, "seeds": [seeds[0], seeds[-1]]}) as run:
        d = make_diffusion(T)
        fp, q = models(dev)
        torch.cuda.reset_peak_memory_stats()
        lat = {"fp": []}
        lat.update({a: [] for a in arms})
        wall = {a: 0.0 for a in ["fp", *arms]}
        cnt = {}
        for b in range(0, len(seeds), BS):
            s = seeds[b:b + BS]
            nz, lab = noise_for_seeds(s, dev), labels_for_seeds(s)
            torch.cuda.synchronize(); t0 = time.time()
            x, c, _ = live_sample(q, fp, d, nz, lab, "fp")
            torch.cuda.synchronize(); wall["fp"] += time.time() - t0; cnt["fp"] = c
            lat["fp"].append(x.cpu())
            for a, (mode, sv) in arms.items():
                torch.cuda.synchronize(); t0 = time.time()
                x, c, _ = live_sample(q, fp, d, nz, lab, mode, K=K, beta=beta, s=sv)
                torch.cuda.synchronize(); wall[a] += time.time() - t0; cnt[a] = c
                lat[a].append(x.cpu())
            print(f"  live_{split}: seeds {s[0]}-{s[-1]} done ({spent():.3f} h logged before this run)", flush=True)
        lat = {k: torch.cat(v) for k, v in lat.items()}
        atomic_torch_save(lat, os.path.join(CK, f"live_{split}_latents.pt"))
        vae = load_vae(dev)
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        imf = decode(vae, lat["fp"].to(dev))
        out = {"K": K, "seeds": [seeds[0], seeds[-1]], "arms": {}, "calls_per_batch": cnt, "wall_s_per_arm_total": wall,
               "peak_mem_bytes": int(torch.cuda.max_memory_allocated()), "residual_buffer_bytes_per_image": 4 * 32 * 32 * 4}
        for a in arms:
            im = decode(vae, lat[a].to(dev))
            ps, lp = [], []
            for i in range(len(im)):
                u, v = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (u - v).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(u / 127.5 - 1, v / 127.5 - 1).flatten()[0]))
            out["arms"][a] = {"rel_mse": rel(lat[a], lat["fp"]), "psnr": ps, "lpips": lp}
            print(f"  {a}: latent rel. MSE {np.mean(out['arms'][a]['rel_mse']):.5f}  PSNR {np.mean(ps):.2f}  LPIPS {np.mean(lp):.4f}", flush=True)
        A = out["arms"]
        pairs = [(x, "anchor_only") for x in A if x.startswith("reuse")] + [(x, "w4") for x in A if x != "w4"]
        out["paired"] = {f"{x}_vs_{y}": {"latent": boot(A[x]["rel_mse"], A[y]["rel_mse"], True), "psnr": boot(A[x]["psnr"], A[y]["psnr"], False),
                                         "lpips": boot(A[x]["lpips"], A[y]["lpips"], False),
                                         "latent_wins": int(np.sum(np.asarray(A[x]["rel_mse"]) < np.asarray(A[y]["rel_mse"])))} for x, y in pairs}
        if split == "dev":
            m = {0.0: np.mean(A["anchor_only"]["rel_mse"]), 0.5: np.mean(A["reuse_s0.5"]["rel_mse"]), 1.0: np.mean(A["reuse_s1"]["rel_mse"])}
            out["mean_rel_mse_by_s"] = {str(k): float(v) for k, v in m.items()}
            out["selected_s"] = min(m, key=m.get)
            co["s"] = out["selected_s"]
            co["s_frozen_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            jdump("coefficients.json", co)
            print(f"  dev selection: s = {out['selected_s']} ({'STOP: s = 0 won' if out['selected_s'] == 0.0 else 'proceed to confirmation'})", flush=True)
        else:
            p = out["paired"]["reuse_vs_anchor_only"]
            out["promising"] = bool(p["latent"]["rel_change"] <= -0.05 and p["latent"]["ci"][1] < 0 and not (p["lpips"]["ci"][0] > 0))
        for k, v in out["paired"].items():
            print(f"  {k}: latent {100 * v['latent']['rel_change']:+.2f}% [{100 * v['latent']['ci'][0]:+.2f}, {100 * v['latent']['ci'][1]:+.2f}], "
                  f"PSNR {v['psnr']['diff']:+.3f}, LPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['ci'][0]:+.4f}, {v['lpips']['ci'][1]:+.4f}], wins {v['latent_wins']}", flush=True)
        out.update(run.record())
    jdump(f"live_{split}.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["provenance", "lag", "sim", "checks", "live_dev", "live_conf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd in ("provenance", "lag", "sim"):
        torch.set_num_threads(8)
        {"provenance": provenance, "lag": lag, "sim": sim}[a.cmd]()
        return
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"checks": lambda: checks(dev), "live_dev": lambda: live(dev, "dev"), "live_conf": lambda: live(dev, "conf")}[a.cmd]()


if __name__ == "__main__":
    main()
