"""Retrospective anchor-bridge correction (results/anchor_bridge/PROTOCOL.md). Stage tag br1.
aa1's C sampler (static A8 cheap model, FP anchors with reuse, feedback off) plus, at anchor a + K, a state correction
    x <- x - lam * (d_new - d_old) * sum_j (j/K) b_{a+j} prod_{m>j} A_{a+m}
estimated from consecutive anchor measurements (zero extra calls; CB1_re re-evaluates both models at the corrected state).
Stages: checks | screen | dev | conf
Run: python -m boa_dit.anchor_bridge.run <stage>
"""
import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.dit_joint import run as DJ
from boa_dit.model import NULL_CLASS, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import guided
from boa_dit.weight_anchor.shared import boot_idx, ci, contrast, summary

STAGE = "br1"
OUT = os.path.join(RESULTS, "anchor_bridge")
CK = os.path.join(CKPT_DIR, "anchor_bridge")
CAP_GPU_H = 0.5
CONF_RESERVE_H = 0.30
K, T = 4, 50
DEV, CONF, AA1DEV = list(range(60208, 60224)), list(range(98100, 98164)), list(range(60200, 60216))
BS = 32
STD = tuple(range(0, 49, K))                                                     # 0, 4, ..., 48
LAST49 = tuple(range(0, 45, K)) + (49,)                                          # 0, 4, ..., 44, 49
HASHES = {"base_model": (DJ.BASE_STATE, "f017af877a79015fb9670504da4fb9fb51df30e1d8e94f6b4a3deedd5aec7ced"),
          "base_carry": (DJ.BASE_CARRY, "b4a537b0570b97d7f50f8f7d9230f1f6e37c157b7277f589285dbd1ee41aedd5"),
          "tables": (DJ.TABLES, "b73030de42a83255338f4a4ab92dce52febf604f5b5b935779c1f2342298dd67"),
          "smoothing": (DJ.SMOOTH, "7ee15c0c72fc18569f2e72671089b20c1293d368dc193d1af49276c1a296db26")}


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


def guard(label, est_h, reserve_h=CONF_RESERVE_H):
    s = spent()
    ok = s + est_h + reserve_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "log": []}
    b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h,
                     "reserved_h": reserve_h, "ok": ok})
    jdump("budget.json", b)
    print(f"[br1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        sys.exit(5)


def provenance():
    out = {k: {"path": p, "sha256": DJ.sha(p), "expected": h} for k, (p, h) in HASHES.items()}
    assert all(v["sha256"] == v["expected"] for v in out.values()), out
    return out


def rl2(x, y):
    return float((x - y).double().norm() / y.double().norm().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------------- DDIM coefficients
def coeffs(d):
    """step i (respaced t = 49 - i): x_{i+1} = A_i x_i + b_i eps_i (DDIM, eta 0)."""
    A, b = [], []
    for i in range(T):
        t = T - 1 - i
        ab, abp = float(d.alphas_cumprod[t]), float(d.alphas_cumprod_prev[t])
        A.append(math.sqrt(abp / ab))
        b.append(math.sqrt(1 - abp) - math.sqrt(abp * (1 - ab) / ab))
    return A, b


def weights(A, b, a, k=K):
    """w_j = b_{a+j} prod_{m=j+1}^{k-1} A_{a+m} (propagation of a step-(a+j) prediction error to x_{a+k}), j = 1..k-1."""
    out = []
    for j in range(1, k):
        p = b[a + j]
        for m in range(j + 1, k):
            p *= A[a + m]
        out.append(p)
    return out


def bridge_coef(A, b, a, k=K):
    return sum((j / k) * w for j, w in zip(range(1, k), weights(A, b, a, k)))


# ------------------------------------------------------------------------------------------------------------------- models / sampler
def models(dev):
    fp = load_dit(dev)
    fp.requires_grad_(False)
    m, jq, _ = DJ.student(dev, None, "static")
    jq.t, jq.auto = None, True
    return fp, m


@torch.no_grad()
def sample(fp, cm, d, noise, labels, anchors=STD, lam=None, reeval=False, kind="corr", rec=None):
    """kind 'corr': anchored sampler with optional bridge (lam None = aa1 C); 'fp': FP50. rec: dict collecting states x_i (before step i)."""
    n = len(noise)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    A, b = coeffs(d)
    tmap = torch.as_tensor(d.timestep_map, device=noise.device, dtype=torch.long)
    x = noise.float().clone()
    cnt = {"cheap": 0, "fp": 0}
    st = {"d": None, "a": None, "pre": None}
    corr_norm = []
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        CTX.step, CTX.n = i, n
        if rec is not None:
            rec[i] = x.detach().clone()
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        if kind == "fp":
            def model_fn(xx, ts):
                cnt["fp"] += 1
                return guided(fp, xx, ts, y, n)
        elif i in anchors:
            ts_m = tmap[t]
            rc, rf = guided(cm, x, ts_m, y, n), guided(fp, x, ts_m, y, n)
            cnt["cheap"] += 1
            cnt["fp"] += 1
            dnew = rc[:, :4] - rf[:, :4]
            if lam is not None and st["d"] is not None and i - st["a"] == K:
                dx = lam * bridge_coef(A, b, st["a"]) * (dnew - st["d"])
                corr_norm.append(float(dx.norm()))
                x = x - dx
                if reeval:
                    rc, rf = guided(cm, x, ts_m, y, n), guided(fp, x, ts_m, y, n)
                    cnt["cheap"] += 1
                    cnt["fp"] += 1
                    dnew = rc[:, :4] - rf[:, :4]
            st["d"], st["a"], st["pre"] = dnew, i, rf

            def model_fn(xx, ts):
                return st["pre"]
        else:
            def model_fn(xx, ts):
                cnt["cheap"] += 1
                res = guided(cm, xx, ts, y, n).clone()
                res[:, :4] -= st["d"]
                return res
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt, corr_norm


ARMS = {"C": dict(), "C_last49": dict(anchors=LAST49), "CB1": dict(lam=1.0), "CB1_re": dict(lam=1.0, reeval=True), "FP50": dict(kind="fp")}
EXPECT = {"C": (50, 13), "C_last49": (50, 13), "CB1": (50, 13), "CBlam": (50, 13), "CB1_re": (62, 25), "FP50": (0, 50)}


def arm_kw(name, lam_star=None):
    return dict(lam=lam_star) if name == "CBlam" else dict(ARMS[name])


def gen(fp, cm, d, seeds, bs, arms, lam_star=None, t0=None, base=0.0, est_batch_h=0.0, timing=None):
    lat, cnt = {a: [] for a in arms}, {}
    for b0 in range(0, len(seeds), bs):
        s = seeds[b0:b0 + bs]
        if t0 is not None and base + (time.time() - t0) / 3600 + est_batch_h > CAP_GPU_H:
            print(f"[br1 runtime stop] before batch {s[0]}-{s[-1]}", flush=True)
            return {a: torch.cat(v) for a, v in lat.items() if v}, cnt, False
        nz, lab = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            torch.cuda.synchronize()
            ts = time.perf_counter()
            x, c, _ = sample(fp, cm, d, nz, lab, **arm_kw(a, lam_star))
            torch.cuda.synchronize()
            if timing is not None:
                timing.setdefault(a, []).append(time.perf_counter() - ts)
            assert torch.isfinite(x).all() and (c["cheap"], c["fp"]) == EXPECT[a], (a, c)
            lat[a].append(x.cpu())
            cnt[a] = c
            print(f"  seeds {s[0]}-{s[-1]} {a}: {time.perf_counter() - ts:.1f} s", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt, True


# ------------------------------------------------------------------------------------------------------------------- checks
def synthetic():
    out = {}
    for name, e in (("linear", [0.10, 0.11, 0.12, 0.13, 0.14]), ("random_walk", [0.10, 0.13, 0.09, 0.12, 0.14])):
        A, b = [1.0] * 8, [-0.1] * 8
        H = sum(w * (e[j] - e[0]) for j, w in zip(range(1, K), weights(A, b, 0)))
        dxh = bridge_coef(A, b, 0) * (e[K] - e[0])
        out[name] = {"state_error": H, "estimate": dxh, "residual_after": H - dxh}
    out["pass"] = abs(out["linear"]["residual_after"]) < 1e-12 and abs(out["random_walk"]["residual_after"] - 0.002) < 1e-12
    return out


@torch.no_grad()
def checks(dev):
    guard("checks", 0.04)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "checks", dev, {}) as run:
        ch = {"provenance": provenance(), "B4_synthetic": synthetic()}
        fp, cm = models(dev)
        d = make_diffusion(T)
        A, b = coeffs(d)
        ch["coefficients"] = {"A": A, "b": b, "bridge_coef": {a: bridge_coef(A, b, a) for a in range(0, 45, K)}}
        g = torch.Generator().manual_seed(0)
        x = torch.randn(4, 4, 32, 32, generator=g).to(dev)
        eps = torch.randn(4, 8, 32, 32, generator=g).to(dev)
        b1 = {}
        for i in (0, 25, 49):
            t = torch.full((4,), T - 1 - i, device=dev, dtype=torch.long)
            ref = d.ddim_sample(lambda xx, ts: eps, x, t, clip_denoised=False, eta=0.0)["sample"]
            b1[i] = rl2(A[i] * x + b[i] * eps[:, :4], ref)
        ch["B1_coeff_rel_l2"] = b1
        ref = torch.load(os.path.join(CKPT_DIR, "activation_anchor", "checks_dev_latents.pt"), map_location="cpu")
        nz, lab = noise_for_seeds(AA1DEV, dev), labels_for_seeds(AA1DEV)
        xc, c, _ = sample(fp, cm, d, nz, lab)
        ch["B2_C_bitwise_aa1"] = bool(torch.equal(xc.cpu(), ref["C"]))
        ch["B2_counts"] = c
        x0, c0, _ = sample(fp, cm, d, nz, lab, lam=0.0)
        ch["B2_lambda0_bitwise_C"] = bool(torch.equal(x0.cpu(), xc.cpu()))
        _, c3, _ = sample(fp, cm, d, nz[:4], lab[:4], anchors=LAST49)
        _, c4, cn = sample(fp, cm, d, nz[:4], lab[:4], lam=1.0, reeval=True)
        ch["B3_counts"] = {"C_last49": c3, "CB1_re": c4, "CB1_re_corrections": len(cn)}
        ch["pass"] = bool(ch["B4_synthetic"]["pass"] and all(v <= 1e-5 for v in b1.values()) and ch["B2_C_bitwise_aa1"] and ch["B2_lambda0_bitwise_C"]
                          and c == {"cheap": 50, "fp": 13} and c3 == {"cheap": 50, "fp": 13} and c4 == {"cheap": 62, "fp": 25} and len(cn) == 12)
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "coefficients", "config")}, indent=1), flush=True)
    if not ch["pass"]:
        sys.exit(6)


def need_checks():
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        print("[br1] checks must pass first", flush=True)
        sys.exit(6)


# ------------------------------------------------------------------------------------------------------------------- screen (S1 + S2)
@torch.no_grad()
def screen(dev):
    need_checks()
    guard("screen", 0.04)
    with Run(STAGE, "screen", dev, {"seeds": [DEV[0], DEV[-1]]}) as run:
        provenance()
        fp, cm = models(dev)
        d = make_diffusion(T)
        A, b = coeffs(d)
        tmap = torch.as_tensor(d.timestep_map, device=dev, dtype=torch.long)
        nz, lab = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        n = len(DEV)
        labt = torch.as_tensor(lab, device=dev)
        y = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        rec = {}
        xC, c, _ = sample(fp, cm, d, nz, lab, rec=rec)
        rec[T] = xC
        # same-state errors along C's trajectory (diagnostic calls)
        e = {}
        for i in range(T):
            CTX.step, CTX.n = i, n
            tsm = tmap[torch.full((n,), T - 1 - i, device=dev, dtype=torch.long)]
            e[i] = (guided(cm, rec[i], tsm, y, n)[:, :4] - guided(fp, rec[i], tsm, y, n)[:, :4]).double()
        # replay each interval from x_{a+1} with FP
        S = {k_: np.zeros(n) for k_ in ("dd", "hh", "dh", "HH", "Hh", "dH", "res_h", "res_H_vs_d")}
        per_int = {}
        for a in range(0, 45, K):
            x = rec[a + 1]
            for j in range(a + 1, a + K):
                CTX.step = j
                t = torch.full((n,), T - 1 - j, device=dev, dtype=torch.long)
                x = d.ddim_sample(lambda xx, ts: guided(fp, xx, ts, y, n), x, t, clip_denoised=False, eta=0.0)["sample"]
            dx = (rec[a + K] - x).double()                                     # actual window deviation x^C - x^F
            w = weights(A, b, a)
            H = sum(wj * (e[a + j] - e[a]) for j, wj in zip(range(1, K), w))   # coefficient-only true hold residual
            hhat = bridge_coef(A, b, a) * (e[a + K] - e[a])                    # bridge estimate
            f = lambda X, Y: (X * Y).flatten(1).sum(1).cpu().numpy()
            vals = {"dd": f(dx, dx), "hh": f(hhat, hhat), "dh": f(dx, hhat), "HH": f(H, H), "Hh": f(H, hhat), "dH": f(dx, H),
                    "res_h": f(dx - hhat, dx - hhat), "res_H_vs_d": f(dx - H, dx - H)}
            for k_ in S:
                S[k_] += vals[k_]
            per_int[a] = {"cos_dx_hhat": float(vals["dh"].sum() / np.sqrt(vals["dd"].sum() * vals["hh"].sum())),
                          "norm_hhat_over_dx": float(np.sqrt(vals["hh"].sum() / vals["dd"].sum())),
                          "cos_dx_H": float(vals["dH"].sum() / np.sqrt(vals["dd"].sum() * vals["HH"].sum())),
                          "norm_H_over_dx": float(np.sqrt(vals["HH"].sum() / vals["dd"].sum())),
                          "residual_after_bridge": float(vals["res_h"].sum() / vals["dd"].sum()),
                          "cos_H_hhat": float(vals["Hh"].sum() / np.sqrt(vals["HH"].sum() * vals["hh"].sum())), "dx_energy": float(vals["dd"].sum())}
        CTX.step = None
        idx = boot_idx(n)[:2000]
        lam_ls = lambda i: S["dh"][i].sum() / S["hh"][i].sum()
        al = np.arange(n)
        lam = float(lam_ls(al))
        lam_ci = ci([lam_ls(i) for i in idx])
        # S1 (cached): H vs hhat; residual of the bridge on H
        hres = S["HH"] - 2 * S["Hh"] + S["hh"]
        s1 = {"cos_H_hhat": float(S["Hh"].sum() / np.sqrt(S["HH"].sum() * S["hh"].sum())), "norm_hhat_over_H": float(np.sqrt(S["hh"].sum() / S["HH"].sum())),
              "residual_H_after_bridge": float(hres.sum() / S["HH"].sum()), "residual_H_after_bridge_ci": ci([hres[i].sum() / S["HH"][i].sum() for i in idx])}
        num = sum(float(((e[a + K] - e[a]) * (e[a] - e[a - K])).sum()) for a in range(K, 45, K))
        den = math.sqrt(sum(float((e[a + K] - e[a]).pow(2).sum()) for a in range(K, 45, K)) * sum(float((e[a] - e[a - K]).pow(2).sum()) for a in range(K, 45, K)))
        s1["forward_extrapolation_cosine"] = num / den
        # S2 (replay): dx vs hhat and vs H
        s2 = {"cos_dx_hhat": float(S["dh"].sum() / np.sqrt(S["dd"].sum() * S["hh"].sum())), "norm_hhat_over_dx": float(np.sqrt(S["hh"].sum() / S["dd"].sum())),
              "residual_after_bridge": float(S["res_h"].sum() / S["dd"].sum()),
              "residual_after_bridge_ci": ci([S["res_h"][i].sum() / S["dd"][i].sum() for i in idx]),
              "cos_dx_H": float(S["dH"].sum() / np.sqrt(S["dd"].sum() * S["HH"].sum())), "norm_H_over_dx": float(np.sqrt(S["HH"].sum() / S["dd"].sum())),
              "residual_after_H": float(S["res_H_vs_d"].sum() / S["dd"].sum()), "lambda_ls": lam, "lambda_ls_ci": lam_ci}
        lam_star = float(min(max(math.floor(10 * lam) / 10, 0.0), 1.0)) if lam_ci[0] > 0 else None
        out = {"seeds": [DEV[0], DEV[-1]], "S1_cached": s1, "S2_replay": s2, "per_interval": per_int, "lambda_star": lam_star,
               "lambda_rule": "lambda* = clip(floor(10 lambda_ls)/10, 0, 1) if the lambda_ls CI lower bound > 0, else CBlam dropped",
               "note": "C's own trajectory; diagnostic same-state FP calls and FP window replays are not part of the method; uncentered cosines pooled over "
                       "12 intervals and 16 images; 2,000-draw image bootstrap", "calls_C": c, **run.record()}
    jdump("screen.json", out)
    print(json.dumps({k: v for k, v in out.items() if k in ("S1_cached", "S2_replay", "lambda_star")}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------------------------- dev / conf
def reading(c):
    m, lp = c["rel_mse"], c["lpips"]
    if m["ratio"] <= 0.95 and m["ratio_ci"][1] < 1 and not lp["diff_ci"][0] > 0:
        return "useful (>= 5% lower mean latent MSE, CI excludes no improvement, no clear LPIPS regression)"
    if m["ratio_ci"][1] < 1:
        return "smaller positive" + (" (LPIPS clearly worse)" if lp["diff_ci"][0] > 0 else " (below the 5% threshold)")
    if m["ratio_ci"][0] > 1:
        return "harmful"
    return "inconclusive"


def arms_list(lam_star):
    return ["FP50", "C", "C_last49", "CB1"] + (["CBlam"] if lam_star is not None else []) + ["CB1_re"]


def pairs(lam_star):
    best = "CBlam" if lam_star is not None else "CB1"
    p = [(best, "C_last49"), ("CB1", "C"), ("C_last49", "C"), ("CB1", "CB1_re")]
    if lam_star is not None:
        p += [("CBlam", "C"), ("CBlam", "CB1")]
    return best, p


def show(Mx, C):
    for a, v in summary(Mx).items():
        print(f"  {a}: rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f} PSNR {v['psnr']['mean']:.2f} LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in C.items():
        m = v["rel_mse"]
        print(f"  {k_}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] wins {m['wins_x']}/{m['n']} "
              f"dLPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['diff_ci'][0]:+.4f}, {v['lpips']['diff_ci'][1]:+.4f}]", flush=True)


@torch.no_grad()
def dev_stage(dev):
    need_checks()
    sc = jload("screen.json")
    lam_star = sc["lambda_star"]
    guard("dev", 0.08)
    with Run(STAGE, "dev", dev, {"seeds": [DEV[0], DEV[-1]], "lambda_star": lam_star}) as run:
        provenance()
        fp, cm = models(dev)
        d = make_diffusion(T)
        lat, cnt, _ = gen(fp, cm, d, DEV, 16, arms_list(lam_star), lam_star)
        atomic_torch_save(lat, os.path.join(CK, "dev_latents.pt"))
        del fp, cm
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        best, P = pairs(lam_star)
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in P}
        e = lambda a: float(np.mean(Mx[a]["rel_mse"]))
        ref = min(e("C"), e("C_last49"))
        bridge = [a for a in ("CB1", "CBlam") if a in Mx]
        gate = {a: e(a) / ref for a in bridge}
        go = any(v <= 0.95 for v in gate.values())
        out = {"seeds": [DEV[0], DEV[-1]], "lambda_star": lam_star, "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C,
               "gate": {"ratios_vs_min_C_Clast49": gate, "proceed": go, "rule": "confirm iff E(CB1) or E(CBlam) <= 0.95 min(E(C), E(C_last49))"},
               **run.record()}
    jdump("dev.json", out)
    show(Mx, C)
    print(f"  gate: {gate} -> {'CONFIRM' if go else 'STOP'}", flush=True)


@torch.no_grad()
def conf(dev):
    fr, dv = jload("frozen.json"), jload("dev.json")
    if not (fr and dv and dv["gate"]["proceed"]):
        print("[br1] committed frozen.json and a passing dev gate required", flush=True)
        sys.exit(6)
    lam_star = fr["lambda_star"]
    guard("conf", 0.25, reserve_h=0.0)
    t0, base = time.time(), spent()
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "batch": BS, "lambda_star": lam_star}) as run:
        provenance()
        fp, cm = models(dev)
        d = make_diffusion(T)
        timing = {}
        lat, cnt, full = gen(fp, cm, d, CONF, BS, arms_list(lam_star), lam_star, t0=t0, base=base, est_batch_h=0.12, timing=timing)
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt" if full else "conf_latents_partial.pt"))
        if not full:
            jdump("conf_partial.json", {"calls": cnt, **run.record()})
            sys.exit(5)
        del fp, cm
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        best, P = pairs(lam_star)
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in P}
        out = {"seeds": [CONF[0], CONF[-1]], "lambda_star": lam_star, "primary": f"{best}_vs_C_last49", "arms": Mx, "summary": summary(Mx), "calls": cnt,
               "contrasts": C, "reading_primary": reading(C[f"{best}_vs_C_last49"]), "reading_CB1_vs_C": reading(C["CB1_vs_C"]),
               "reading_C_last49_vs_C": reading(C["C_last49_vs_C"]),
               "timing_s_per_batch32": {a: {"median": float(np.median(v)), "all": v} for a, v in timing.items()}, **run.record()}
    jdump("conf.json", out)
    show(Mx, C)
    for k_ in [x for x in out if x.startswith("reading")]:
        print(f"  {k_}: {out[k_]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "screen", "dev", "conf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"checks": checks, "screen": screen, "dev": dev_stage, "conf": conf}[a.cmd](dev)


if __name__ == "__main__":
    main()
