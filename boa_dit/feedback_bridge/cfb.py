"""Pilot J (cfb1): zero-reference DCT-4 feedback with jointly calibrated anchor-step gain, between-step gain and bridge strength
(results/coupled_feedback_bridge/PROTOCOL.md). theta = (g_anchor, g_between, lambda) on the fixed grid {0, .5, 1}^2 x {.5, 1, 1.5}.
Stages: profile | checks | fit | val | conf | diag
Run: python -m boa_dit.feedback_bridge.cfb <stage>       (queue: bash scripts/cfb1_queue.sh)
"""
import argparse
import csv
import itertools
import json
import os
import sys
import time

import numpy as np
import torch

from boa_dit.activation_bridge import run as AB
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.dit_joint import run as DJ
from boa_dit.feedback_bridge import core as FC
from boa_dit.feedback_bridge.core import spec
from boa_dit.model import NULL_CLASS, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor.run import guided, rel
from boa_dit.weight_anchor.shared import summary

STAGE = "cfb1"
OUT = os.path.join(RESULTS, "coupled_feedback_bridge")
CK = os.path.join(CKPT_DIR, "coupled_feedback_bridge")
CAP_GPU_H = 3.0
SHUTDOWN_RESERVE_H = 0.10
T = FC.T
FIT = list(range(120000, 120032))
VAL = list(range(120100, 120132))
CONF = list(range(121000, 121256))
DIAG = FIT[:8]
OVN = list(range(110000, 110032))
CHK = list(range(60200, 60204))
GAINS = (0.0, 0.5, 1.0)
LAMS = (0.5, 1.0, 1.5)
GRID = [(ga, gb, lam) for ga in GAINS for gb in GAINS for lam in LAMS]       # 27 points, fixed before data
FIXED = {"B0": (0.0, 0.0, 1.0), "N": (1.0, 1.0, 1.0), "H": (0.0, 1.0, 1.0)}


def name(th):
    return f"J({th[0]:g},{th[1]:g},{th[2]:g})"


def jspec(th):
    return spec("J", K=4, bridge=True, gains=th[:2])


FP50 = {"high_only": "fp"}


# ------------------------------------------------------------------------------------------------------------------- ledger
def jdump(n, obj):
    atomic_write_json(os.path.join(OUT, n), obj)


def jload(n):
    f = os.path.join(OUT, n)
    return json.load(open(f)) if os.path.exists(f) else None


def spent(tag=STAGE):
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == tag) if os.path.exists(f) else 0.0


def fits(label, est_h, reserve_h):
    s = spent()
    ok = s + est_h + reserve_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "combined_cap_gpu_h": 6.0, "log": []}
    b["log"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": label, "spent_before_h": s, "estimate_h": est_h,
                     "reserve_h": reserve_h, "ok": ok})
    b["spent_gpu_h"] = s
    jdump("budget.json", b)
    print(f"[cfb1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'SKIP'}", flush=True)
    return ok


def BS():
    return int(jload("frozen.json")["batch_size"])


def load_models(dev):
    M = {"fp": load_dit(dev), "fb": FC.CheapFB(dev)}
    M["fp"].requires_grad_(False)
    return M


def eq_states(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def snap(jq):
    return {k: v.detach().clone() for k, v in jq.state_out.items()}


def traj_s():
    p = jload("profile.json")
    return p["s_per_traj"]["J"], p["s_per_traj"]["FP50"]


def est_h(n_cfg, n_img, fp=True):
    j, f = traj_s()
    return (n_cfg * j + (f if fp else 0)) * (n_img / BS()) * 1.10 / 3600


# ------------------------------------------------------------------------------------------------------------------- profile
@torch.no_grad()
def profile(dev):
    """batch 32 memory and per-trajectory time for J and FP50 (8 steps, dev noise); rule: batch 32 if peak reserved <= 0.85 x device, else 16."""
    with Run(STAGE, "profile", dev, {}) as run:
        M = load_models(dev)
        d = make_diffusion(T)
        total = torch.cuda.get_device_properties(0).total_memory
        out = {"device_total_bytes": total, "rule": "batch 32 if J peak reserved <= 0.85 x device memory, else 16", "s_per_traj": {}}
        s = (list(range(60200, 60232)))
        nz, lb = noise_for_seeds(s, dev), labels_for_seeds(s)
        torch.cuda.reset_peak_memory_stats()
        for a, sp in (("J", jspec((0.5, 0.5, 1.0))), ("FP50", FP50)):
            ts = []
            for _ in range(2):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                FC.sample(M, sp, d, nz, lb, stop_at=8)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 8)
            out["s_per_traj"][a] = ts[1] * T
            print(f"  {a} bs32: {ts[1] * T:.1f} s / trajectory (est.)", flush=True)
        out["peak_reserved"] = torch.cuda.max_memory_reserved()
        out["batch_size"] = 32 if out["peak_reserved"] <= 0.85 * total else 16
        out.update(run.record())
    j, f = out["s_per_traj"]["J"], out["s_per_traj"]["FP50"]
    b = out["batch_size"]
    out["plan_h"] = {"fit": (27 * j + f) * len(FIT) / b * 1.1 / 3600, "val": (6 * j + f) * len(VAL) / b * 1.1 / 3600 + 0.03,
                     "conf": (6 * j + f) * len(CONF) / b * 1.1 / 3600 + 0.08, "checks": 0.20, "diag": 0.15}
    out["plan_total_h"] = sum(out["plan_h"].values()) + out["gpu_hours"]
    jdump("profile.json", out)
    return 0


# ------------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def checks(dev):
    with Run(STAGE, "checks", dev, {"batch_size": BS()}) as run:
        ch = {"provenance": AB.provenance()}
        d = make_diffusion(T)
        ch["synthetic_constant_linear"] = AB.synthetic(d)
        M = load_models(dev)
        jq = M["fb"].jq
        # reproduce overnight G, GB, ACB, FP50 at batch 32 (110000-110031) with J configurations (and with the original rules)
        L = torch.load(os.path.join(AB.CK, "s1_latents.pt"), map_location="cpu")
        nz, lb = noise_for_seeds(OVN, dev), labels_for_seeds(OVN)
        r = {}
        r["FP50"] = bool(torch.equal(FC.sample(M, FP50, d, nz, lb)[0].cpu(), L["FP50"][:32]))
        r["J(0,0)_nobridge=G"] = bool(torch.equal(FC.sample(M, spec("J", gains=(0, 0)), d, nz, lb)[0].cpu(), L["G"][:32]))
        r["J(0,0,1)=GB"] = bool(torch.equal(FC.sample(M, jspec((0, 0, 1)), d, nz, lb)[0].cpu(), L["GB"][:32]))
        x, _ = FC.sample(M, jspec((1, 1, 1)), d, nz, lb)
        sj = snap(jq)
        r["J(1,1,1)=ACB"] = bool(torch.equal(x.cpu(), L["ACB"][:32]))
        x2, _ = FC.sample(M, spec("fb", bridge=True), d, nz, lb)
        r["J(1,1,1)_state=ACB_state"] = bool(torch.equal(x2.cpu(), L["ACB"][:32]) and eq_states(sj, snap(jq)))
        del L
        ch["reproduce_overnight_bs32"] = r
        print(f"  overnight reproduction {r}", flush=True)
        nz, lb = noise_for_seeds(CHK, dev), labels_for_seeds(CHK)
        th = (0.5, 1.0, 1.5)
        # J(1,1) no bridge = AC incl. state (batch 4); lambda 0 = no-bridge incl. state
        xa, _ = FC.sample(M, spec("fb"), d, nz, lb); sa = snap(jq)
        xj, _ = FC.sample(M, spec("J", gains=(1, 1)), d, nz, lb); sj = snap(jq)
        ch["J(1,1)_nobridge=AC_incl_state"] = bool(torch.equal(xa, xj) and eq_states(sa, sj))
        x0, _ = FC.sample(M, spec("J", gains=th[:2]), d, nz, lb); s0 = snap(jq)
        x1, _ = FC.sample(M, jspec(th), d, nz, lb, lam=0.0); s1 = snap(jq)
        ch["lambda0_equals_no_bridge_incl_state"] = bool(torch.equal(x0, x1) and eq_states(s0, s1))
        # K1 = FP50, zero corrections
        f50, _ = FC.sample(M, FP50, d, nz, lb)
        xk, ck = FC.sample(M, spec("J", K=1, bridge=True, gains=th[:2]), d, nz, lb)
        ch["K1_equals_FP50"] = bool(torch.equal(xk, f50) and ck == {"cheap": 50, "high": 50, "corrections": 0})
        # counts, per-site updates, identical cheap / FP anchor inputs
        rec_in = {"cheap": [], "high": []}
        hk = [M["fb"].m.x_embedder.register_forward_pre_hook(lambda mod, inp: rec_in["cheap"].append((CTX.step, inp[0].detach().clone()))),
              M["fp"].x_embedder.register_forward_pre_hook(lambda mod, inp: rec_in["high"].append((CTX.step, inp[0].detach().clone())))]
        jq.calls = {k: 0 for k in DJ.KEYS}
        xr, c = FC.sample(M, jspec(th), d, nz, lb)
        hi = dict(rec_in["high"])
        same = [torch.equal(v, hi[s]) for s, v in rec_in["cheap"] if s in hi]
        for h in hk:
            h.remove()
        ch["counts"] = {"counts": c, "expected": FC.expect(jspec(th)), "quantizer_calls_per_site": sorted(set(jq.calls.values())),
                        "anchor_inputs_compared": len(same), "anchor_inputs_identical": all(same)}
        ch["counts_pass"] = c == FC.expect(jspec(th)) and sorted(set(jq.calls.values())) == [50] and len(same) == 13 and all(same)
        # independence and reset between batches
        nzp = nz.clone()
        nzp[3] = torch.randn(nzp[3].shape, generator=torch.Generator().manual_seed(7)).to(dev)
        xp, _ = FC.sample(M, jspec(th), d, nzp, lb)
        nzb, lbb = noise_for_seeds(list(range(60204, 60208)), dev), labels_for_seeds(list(range(60204, 60208)))
        xb1, _ = FC.sample(M, jspec(th), d, nzb, lbb)
        xb2, _ = FC.sample(M, jspec(th), d, nzb, lbb)
        ch["independence"] = bool(torch.equal(xp[:3], xr[:3]))
        ch["reset_between_batches"] = bool(torch.equal(xb1, xb2))
        # resume before (8) / after (9) an anchor
        rs = {}
        xr0, _ = FC.sample(M, jspec(th), d, nz, lb); st0 = snap(jq)
        for sa_ in (8, 9):
            s_ = FC.sample(M, jspec(th), d, nz, lb, stop_at=sa_)
            p = os.path.join(CK, f"resume_{sa_}.pt")
            atomic_torch_save(s_, p)
            xr_, cr = FC.sample(M, jspec(th), d, nz, lb, resume=torch.load(p, map_location="cpu"))
            rs[sa_] = bool(torch.equal(xr_, xr0) and eq_states(snap(jq), st0) and cr == FC.expect(jspec(th)))
            os.remove(p)
        ch["resume_bitwise"] = rs
        # instrumentation on/off
        Dg = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
        xd, _ = FC.sample(M, jspec(th), d, nz, lb, diag=Dg); sd = snap(jq)
        ch["instrumentation_on_off_bitwise"] = bool(torch.equal(xd, xr0) and eq_states(sd, st0))
        # gain placement: anchor call with g_anchor = 0 equals G on that call; held call with g_between = 1 equals AC given the same state
        n2 = 2
        xs = torch.randn(n2, 4, 32, 32, generator=torch.Generator().manual_seed(1)).to(dev)
        labt = torch.as_tensor(lb[:n2], device=dev)
        yy = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        tsx = torch.full((n2,), 500, device=dev, dtype=torch.long)
        CTX.step, CTX.n = 20, n2
        junk = {k: torch.randn(2 * n2, 4, v.shape[-1], generator=torch.Generator().manual_seed(5)).half().to(dev) for k, v in sa.items()}
        out = {}
        for nm, sp_, anchors in (("G", spec("G"), {20}), ("AC", spec("fb"), {20}), ("J_anchor", spec("J", gains=(0, 1)), {20}),
                                 ("J_held", spec("J", gains=(0, 1)), {16})):
            FC.configure(jq, sp_, anchors)
            jq.state_in, jq.state_out = {k: v.clone() for k, v in junk.items()}, {}
            out[nm] = (guided(M["fb"].m, xs, tsx, yy, n2), snap(jq))
        CTX.step = None
        ch["gain_anchor0_equals_G_call"] = bool(torch.equal(out["J_anchor"][0], out["G"][0]))
        ch["gain_between1_equals_AC_call_incl_state"] = bool(torch.equal(out["J_held"][0], out["AC"][0]) and eq_states(out["J_held"][1], out["AC"][1]))
        ch["state_bytes_per_image_incl_cfg"] = sum(v.numel() * v.element_size() for v in st0.values()) / len(CHK)
        crit = [ch["synthetic_constant_linear"]["pass"], *r.values(), ch["J(1,1)_nobridge=AC_incl_state"], ch["lambda0_equals_no_bridge_incl_state"],
                ch["K1_equals_FP50"], ch["counts_pass"], ch["independence"], ch["reset_between_batches"], *rs.values(), ch["instrumentation_on_off_bitwise"],
                ch["gain_anchor0_equals_G_call"], ch["gain_between1_equals_AC_call_incl_state"]]
        ch["pass"] = bool(all(crit))
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1, default=str), flush=True)
    return 0 if ch["pass"] else 6


# ------------------------------------------------------------------------------------------------------------------- generation
def gen(M, d, seeds, cfgs, part, j_s, f_s):
    """per (batch, configuration) checkpoint; cfgs: {arm name: spec-and-lambda tuple or 'FP50'}."""
    bs = BS()
    pf = os.path.join(CK, f"{part}.pt")
    P = torch.load(pf, map_location="cpu") if os.path.exists(pf) else {"lat": {}, "cnt": {}, "timing": {}, "bs": bs}
    assert P["bs"] == bs
    t0 = time.time()
    for b0 in range(0, len(seeds), bs):
        s = seeds[b0:b0 + bs]
        nz, lb = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a, th in cfgs.items():
            if (a, b0) in P["lat"]:
                continue
            est = (f_s if th == "FP50" else j_s) / 3600
            if spent() + (time.time() - t0) / 3600 + est + SHUTDOWN_RESERVE_H > CAP_GPU_H:
                print(f"[cfb1 runtime stop] before {a} batch {s[0]}", flush=True)
                return P, False
            sp, lam = (FP50, 1.0) if th == "FP50" else (jspec(th), th[2])
            torch.cuda.synchronize(); ts = time.perf_counter()
            x, c = FC.sample(M, sp, d, nz, lb, lam=lam)
            torch.cuda.synchronize(); dt = time.perf_counter() - ts
            assert torch.isfinite(x).all() and c == FC.expect(sp), (a, c)
            P["lat"][(a, b0)] = x.cpu(); P["cnt"][a] = c
            P["timing"].setdefault(a, []).append(dt)
            atomic_torch_save(P, pf)
            print(f"  seeds {s[0]}-{s[-1]} {a}: {dt:.1f} s", flush=True)
    return P, True


def collect(P, cfgs, seeds):
    bs = BS()
    return {a: torch.cat([P["lat"][(a, b0)] for b0 in range(0, len(seeds), bs)]) for a in cfgs}


def tiebreak_key(th, err):
    return (err, sum(g != 0 for g in th[:2]), th[0] + th[1], abs(th[2] - 1.0), th)


def select(errs, family):
    """smallest fit mean error; exact ties within relative 1e-6: fewer nonzero gains, lower gain sum, smaller |lambda - 1|, lexicographic tuple."""
    cand = {th: e for th, e in errs.items() if th in family}
    best = min(cand.values())
    tied = [th for th, e in cand.items() if e <= best * (1 + 1e-6)]
    return min(tied, key=lambda th: tiebreak_key(th, 0.0))


def fit_stage(dev):
    j, f = traj_s()
    plan = jload("profile.json")["plan_h"]
    need = est_h(len(GRID), len(FIT)) + plan["val"] + plan["conf"] + plan["diag"]
    if not fits("fit (complete grid + reserved validation, confirmation, diagnostics)", need, SHUTDOWN_RESERVE_H):
        jdump("fit.json", {"status": "skipped (complete grid + reserved confirmation do not fit 3 h; fit data not opened)", "required_h": need,
                           "spent_h": spent()})
        return 4
    cfgs = {"FP50": "FP50", **{name(th): th for th in GRID}}
    with Run(STAGE, "fit", dev, {"seeds": [FIT[0], FIT[-1]], "grid": GRID, "batch": BS()}) as run:
        AB.provenance()
        M = load_models(dev)
        d = make_diffusion(T)
        P, full = gen(M, d, FIT, cfgs, "fit_part", j, f)
        del M
        torch.cuda.empty_cache()
        if not full:
            jdump("fit.json", {"status": "partial (stopped by budget)", **run.record()})
            return 5
        lat = collect(P, cfgs, FIT)
        atomic_torch_save(lat, os.path.join(CK, "fit_latents.pt"))
        per = {name(th): rel(lat[name(th)], lat["FP50"]) for th in GRID}
        errs = {th: float(np.mean(per[name(th)])) for th in GRID}
        fam = {"B": [th for th in GRID if th[0] == 0 and th[1] == 0], "U": [th for th in GRID if th[0] == th[1]], "J": list(GRID)}
        sel = {k: select(errs, v) for k, v in fam.items()}
        out = {"status": "complete", "seeds": [FIT[0], FIT[-1]], "n": len(FIT), "batch": BS(), "metric": "latent rel. MSE vs paired FP50 (weight_anchor.run.rel)",
               "surface_mean": {name(th): errs[th] for th in GRID}, "surface_median": {name(th): float(np.median(per[name(th)])) for th in GRID},
               "per_image": per, "selected": {k: list(v) for k, v in sel.items()}, "selected_names": {k: name(v) for k, v in sel.items()},
               "selected_fit_error": {k: errs[v] for k, v in sel.items()},
               "J_chose_feedback_off": sel["J"][0] == 0 and sel["J"][1] == 0, "calls": P["cnt"],
               "timing_s_per_batch": {a: float(np.median(v)) for a, v in P["timing"].items()}, **run.record()}
    jdump("fit.json", out)
    for th in GRID:
        print(f"  {name(th)}: {errs[th]:.5f}", flush=True)
    print(f"  selected {out['selected_names']}", flush=True)
    return 0


def frozen_cfgs():
    fit = jload("fit.json")
    sel = {k: tuple(v) for k, v in fit["selected"].items()}
    cf = {}
    for k, th in list(FIXED.items()) + [(k, sel[k]) for k in ("B", "U", "J")]:
        cf.setdefault(name(th), []).append(k)
    return sel, cf                                                          # cf: unique configuration name -> roles


def quality(dev, stage, seeds, reserve):
    sel, cf = frozen_cfgs()
    j, f = traj_s()
    ths = {n_: next(th for th in GRID if name(th) == n_) for n_ in cf}
    est = est_h(len(ths), len(seeds)) + 0.03 * len(ths) * len(seeds) / 128
    if not fits(stage, est, SHUTDOWN_RESERVE_H + reserve):
        jdump(f"{stage}.json", {"status": "skipped (budget)", "estimate_h": est})
        return None
    cfgs = {"FP50": "FP50", **ths}
    with Run(STAGE, stage, dev, {"seeds": [seeds[0], seeds[-1]], "configs": list(ths), "batch": BS()}) as run:
        AB.provenance()
        M = load_models(dev)
        d = make_diffusion(T)
        P, full = gen(M, d, seeds, cfgs, f"{stage}_part", j, f)
        del M
        torch.cuda.empty_cache()
        if not full:
            jdump(f"{stage}.json", {"status": "partial (stopped by budget)", **run.record()})
            return None
        lat = collect(P, cfgs, seeds)
        atomic_torch_save(lat, os.path.join(CK, f"{stage}_latents.pt"))
        role = {r: n_ for n_, rs in cf.items() for r in rs}
        L = {"FP50": lat["FP50"], **{r: lat[n_] for r, n_ in role.items()}}             # role-named copies (deduplicated generation)
        pairs = [("J", "B"), ("J", "U"), ("U", "B"), ("B", "B0"), ("J", "N"), ("N", "B0"), ("H", "B0"), ("J", "H"), ("J", "B0")]
        pairs = [(x, y) for x, y in pairs if role[x] != role[y]]
        Mx, C = AB.evaluate(L, dev, pairs)
        out = {"status": "complete", "seeds": [seeds[0], seeds[-1]], "n": len(seeds), "batch": BS(), "roles": role, "dedup": cf,
               "arms": Mx, "summary": summary(Mx), "contrasts": C, "identical_role_pairs": [(x, y) for x, y in
               [("J", "B"), ("J", "U"), ("U", "B")] if role[x] == role[y]], "calls": P["cnt"],
               "timing_s_per_batch": {a: float(np.median(v)) for a, v in P["timing"].items()}, **run.record()}
    AB.show(Mx, C)
    return out


def val_stage(dev):
    if not (jload("fit.json") or {}).get("status") == "complete":
        jdump("val.json", {"status": "blocked (fit incomplete)"})
        return 4
    plan = jload("profile.json")["plan_h"]
    out = quality(dev, "val", VAL, plan["conf"] + plan["diag"])
    if out is None:
        return 4
    S, role = out["summary"], out["roles"]
    thJ = next(th for th in GRID if name(th) == role["J"])
    nonzero = thJ[0] != 0 or thJ[1] != 0
    if role["J"] == role["B"]:
        ratio, dlp = 1.0, 0.0
    else:
        ratio = S["J"]["rel_mse"]["mean"] / S["B"]["rel_mse"]["mean"]
        dlp = S["J"]["lpips"]["mean"] - S["B"]["lpips"]["mean"]
    out["gate"] = {"J_nonzero_feedback": nonzero, "ratio_J_over_B": ratio, "dlpips_J_minus_B": dlp,
                   "pass": bool(nonzero and ratio <= 0.95 and dlp <= 0.005),
                   "rule": "J nonzero feedback, mean error <= 0.95 x B, mean LPIPS <= LPIPS(B) + 0.005"}
    if not nonzero:
        out["gate"]["note"] = "the J family selected feedback off (its fit optimum is a bridge-only configuration)"
    jdump("val.json", out)
    print(f"  validation gate: {out['gate']}", flush=True)
    return 0


def conf_stage(dev):
    v = jload("val.json")
    if not (v and v.get("status") == "complete"):
        jdump("conf.json", {"status": "blocked (validation incomplete)"})
        return 4
    if not v["gate"]["pass"]:
        jdump("conf.json", {"status": "skipped (validation gate failed; confirmation cohort 121000-121255 not opened)", "gate": v["gate"]})
        return 4
    plan = jload("profile.json")["plan_h"]
    out = quality(dev, "conf", CONF, plan["diag"])
    if out is None:
        return 4
    C = out["contrasts"]
    prim = AB.reading(C["J_vs_B"])
    out["primary"] = "J_vs_B"
    out["reading_primary"] = prim
    out["reading_J_vs_U"] = (AB.reading(C["J_vs_U"]) if "J_vs_U" in C else "J = U (phase specificity not established)") if prim.startswith("useful") \
        else "not tested (primary not successful)"
    out["note"] = ("secondary intervals exploratory / unadjusted; improvement over N alone does not establish useful activation feedback; "
                   "if only lambda differs from B0 the effect is bridge recalibration")
    jdump("conf.json", out)
    return 0


# ------------------------------------------------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def diag_stage(dev):
    if not (jload("fit.json") or {}).get("status") == "complete":
        jdump("diag.json", {"status": "blocked (fit incomplete)"})
        return 4
    if not fits("diag", 0.15, SHUTDOWN_RESERVE_H):
        jdump("diag.json", {"status": "skipped (budget)"})
        return 4
    sel, _ = frozen_cfgs()
    pairs = sorted({(ga, gb) for ga in GAINS for gb in GAINS})
    with Run(STAGE, "diag", dev, {"seeds": [DIAG[0], DIAG[-1]]}) as run:
        M = load_models(dev)
        d = make_diffusion(T)
        Ac, bc = FC.coeffs(d)
        nz, lb = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        Df = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
        f50, _ = FC.sample(M, FP50, d, nz, lb, diag=Df)
        force = [Df["x"][i] for i in range(T)]
        out = {"status": "complete", "seeds": [DIAG[0], DIAG[-1]], "live": {}, "forced": {},
               "definitions": {"H_g": "sum_j w_j (e_{a+j} - e_a) over the 12 complete K4 intervals (abn1 s5)", "v_g": "(e_c - e_a) sum_j (j/k) w_j",
                               "frozen_window_objective": "E|H|^2 - 2 lam E<H,v> + lam^2 E|v|^2; lambda_LS = E<H,v>/E|v|^2 (NOT used for selection)",
                               "forced": "cheap and FP called on the FP50 latent trajectory (common forced trajectory), each gain pair with its own state history"},
               "limitation": "8 fit images; live diagnostics on each configuration's own trajectory; nothing here feeds selection"}
        live = {name((ga, gb, 1.0)): (ga, gb, 1.0) for ga, gb in pairs}
        live.update({f"{k}:{name(v)}": v for k, v in sel.items()})
        for nm, th in live.items():
            D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
            x, c = FC.sample(M, jspec(th), d, nz, lb, lam=th[2], diag=D)
            r = FC.interval_stats(D["e"], Ac, bc, lam=th[2])
            r["endpoint_rel_mse_mean"] = float(np.mean(rel(x.cpu(), f50.cpu())))
            r["jump_norms"] = D["jump"]
            out["live"][nm] = r
            print(f"  live {nm}: endpoint {r['endpoint_rel_mse_mean']:.5f} HH {r['HH']:.4g} Hv {r['Hv']:.4g} vv {r['vv']:.4g} lamLS {r['lambda_ls_frozen_window']:.3f}", flush=True)
        for ga, gb in pairs:
            D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
            FC.sample(M, jspec((ga, gb, 1.0)), d, nz, lb, diag=D, force=force)
            r = FC.interval_stats(D["e"], Ac, bc)
            out["forced"][f"g=({ga:g},{gb:g})"] = r
            print(f"  forced ({ga:g},{gb:g}): raw_anchor {r['raw_anchor']:.4g} raw_between {r['raw_between']:.4g} HH {r['HH']:.4g} cos {r['cos_H_v']:.3f}", flush=True)
        out.update(run.record())
    jdump("diag.json", out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "checks", "fit", "val", "conf", "diag"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if a.cmd in ("fit", "val", "conf", "diag"):
        ch = jload("checks.json")
        if not (ch and ch["pass"]):
            print("[cfb1] checks have not passed: stage blocked", flush=True)
            jdump(f"{a.cmd}.json", {"status": "blocked (checks not passed)"})
            sys.exit(4)
    rc = {"profile": profile, "checks": checks, "fit": fit_stage, "val": val_stage, "conf": conf_stage, "diag": diag_stage}[a.cmd](dev)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
