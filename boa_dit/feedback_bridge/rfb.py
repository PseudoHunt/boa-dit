"""Pilot R (rfb1): anchor-referenced activation feedback + anchor reuse + retrospective bridge (results/reference_feedback_bridge/PROTOCOL.md).
Arms (FP anchors, K4, lambda 1): G, GB, ACB (fixed old stack), RFB (R-full + bridge), RDB (R-DCT4 + bridge), RD (R-DCT4, bridge off); FP50 reference.
Stages: profile | checks | dev | conf | diag
Run: python -m boa_dit.feedback_bridge.rfb <stage>       (queue: bash scripts/rfb1_queue.sh)
"""
import argparse
import csv
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
from boa_dit.weight_anchor.run import guided
from boa_dit.weight_anchor.shared import summary

STAGE = "rfb1"
OUT = os.path.join(RESULTS, "reference_feedback_bridge")
CK = os.path.join(CKPT_DIR, "reference_feedback_bridge")
CAP_GPU_H = 3.0
SHUTDOWN_RESERVE_H = 0.10
T = FC.T
DEV = list(range(60200, 60232))                                              # inspected development (exploratory, freshness not claimed)
DIAG = list(range(60208, 60216))
CONF = list(range(122000, 122128))                                           # fresh confirmation (audited 2026-10-31, unused)
OVN = list(range(110000, 110032))                                            # overnight s1 batch 1 (evaluated) for bitwise reproduction at batch 32
CHK = list(range(60200, 60204))
ARMS = {"FP50": {"high_only": "fp"}, "G": spec("G"), "GB": spec("G", bridge=True), "AC": spec("fb"), "ACB": spec("fb", bridge=True),
        "RF": spec("R-full"), "RFB": spec("R-full", bridge=True), "RD": spec("R-dct"), "RDB": spec("R-dct", bridge=True)}
QUALITY_ARMS = ["FP50", "G", "GB", "ACB", "RFB", "RDB", "RD"]
PAIRS = [("RDB", "GB"), ("RFB", "GB"), ("RDB", "RFB"), ("RDB", "ACB"), ("RFB", "ACB"), ("RD", "G"), ("RDB", "RD"), ("ACB", "GB"), ("GB", "G")]


# ------------------------------------------------------------------------------------------------------------------- ledger
def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
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
    print(f"[rfb1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'SKIP'}", flush=True)
    return ok


def frozen():
    return jload("frozen.json") or {}


def BS():
    return int(frozen()["batch_size"])


def load_models(dev):
    M = {"fp": load_dit(dev), "fb": FC.CheapFB(dev)}
    M["fp"].requires_grad_(False)
    return M


def eq_states(a, b):
    return set(a) == set(b) and all(torch.equal(a[k], b[k]) for k in a)


def snap(jq):
    return {k: v.detach().clone() for k, v in jq.state_out.items()}, {k: v.detach().clone() for k, v in jq.ref.items()}


# ------------------------------------------------------------------------------------------------------------------- profile
@torch.no_grad()
def profile(dev):
    """R-full memory and per-step time at batch 32 / 16 / 8 (6 steps, dev noise), then G, ACB, RDB, FP50 per-step time at the chosen batch.
    Rule (frozen in PROTOCOL): largest of 32, 16, 8 whose R-full peak reserved memory <= 0.85 x device memory."""
    with Run(STAGE, "profile", dev, {}) as run:
        M = load_models(dev)
        d = make_diffusion(T)
        total = torch.cuda.get_device_properties(0).total_memory
        out = {"device_total_bytes": total, "rule": "largest of 32/16/8 with R-full peak reserved <= 0.85 x device memory", "rfull": {}}
        choice = None
        for bs in (32, 16, 8):
            s = (DEV + DEV)[:bs]
            nz, lb = noise_for_seeds(s, dev), labels_for_seeds(s)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            try:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                FC.sample(M, ARMS["RFB"], d, nz, lb, stop_at=6)
                torch.cuda.synchronize()
                r = {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(),
                     "s_per_step_incl_warmup": (time.perf_counter() - t0) / 6, "state_bytes_per_image_incl_cfg": M["fb"].jq.state_bytes() / bs, "oom": False}
            except torch.cuda.OutOfMemoryError:
                r = {"oom": True}
            M["fb"].jq.reset()
            torch.cuda.empty_cache()
            out["rfull"][bs] = r
            print(f"  R-full bs{bs}: {r}", flush=True)
            if choice is None and not r["oom"] and r["peak_reserved"] <= 0.85 * total:
                choice = bs
        out["batch_size"] = choice
        s = (DEV + DEV)[:choice]
        nz, lb = noise_for_seeds(s, dev), labels_for_seeds(s)
        tm = {}
        for a in ("FP50", "G", "ACB", "RDB", "RFB"):
            ts = []
            for rep in range(2):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                FC.sample(M, ARMS[a], d, nz, lb, stop_at=8)
                torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) / 8)
            tm[a] = {"s_per_step_warm": ts[1], "est_s_per_trajectory": ts[1] * T}
            print(f"  {a} bs{choice}: {ts[1] * T:.1f} s / trajectory (est.)", flush=True)
        out["timing_at_choice"] = tm
        out.update(run.record())
    jdump("profile.json", out)
    return 0


def est_traj_h(a):
    p = jload("profile.json")
    return p["timing_at_choice"]["FP50" if a == "FP50" else ("G" if a in ("G", "GB") else "ACB" if a in ("AC", "ACB") else
                                                            "RFB" if a.startswith("RF") else "RDB")]["est_s_per_trajectory"] / 3600


# ------------------------------------------------------------------------------------------------------------------- checks
def ddim_coeff_check(d, dev):
    Ac, bc = FC.coeffs(d)
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for i in range(T):
        x = torch.randn(2, 4, 32, 32, generator=g).to(dev)
        e = torch.randn(2, 8, 32, 32, generator=g).to(dev)
        t = torch.full((2,), T - 1 - i, device=dev, dtype=torch.long)
        y = d.ddim_sample(lambda xx, ts: e, x, t, clip_denoised=False, eta=0.0)["sample"]
        worst = max(worst, FC.rl2(y, Ac[i] * x + bc[i] * e[:, :4]))
    return {"max_rel_l2_vs_ddim_fp32": worst, "pass": worst < 1e-5, "A_all_ge_1": all(a >= 1 for a in Ac), "b_all_le_0": all(b <= 0 for b in bc),
            "K4_bridge_coef_sign": sorted({float(np.sign(FC.bridge_coef(Ac, bc, a, 4))) for a in range(0, 45, 4)})}


def alternating_example(d):
    """descriptive: alternating cheap error e_j = e0 + s (-1)^j over K4 intervals: bridge residual vs the uncorrected deviation (fp64)."""
    Ac, bc = FC.coeffs(d)
    g = torch.Generator().manual_seed(0)
    out = []
    for a in range(0, 45, 4):
        e0, s = torch.randn(5, generator=g, dtype=torch.float64), torch.randn(5, generator=g, dtype=torch.float64)
        e = lambda j: e0 + s * (-1) ** j
        dev_ = sum(w * (e(j) - e(0)) for j, w in zip(range(1, 4), FC.weights(Ac, bc, a, 4)))
        corr = FC.bridge_coef(Ac, bc, a, 4) * (e(4) - e(0))
        out.append(float((dev_ - corr).norm() / dev_.norm()))
    return {"residual_over_deviation_per_interval": out, "note": "e_4 - e_0 = 0 for an alternating error: the bridge applies no correction"}


def quantizer_unit(dev):
    """row independence of the stateful rules on synthetic inputs (two calls, perturb one row: other rows' outputs and states unchanged) and
    real state byte counts."""
    m = DJ.common_base_nocarry(dev)
    jq = FC.FBQuant(m, dev)
    jq.remove()
    jq.t, jq.auto = None, True
    key = "b5.mlp.fc2"
    D, Z, qm = jq.p[key]
    out = {}
    for rule in ("R-full", "R-dct", "J", "fb"):
        res = []
        for pert in (False, True):
            jq.reset(); jq.rule, jq.gains, jq.anchors, jq.zero_state = rule, (0.5, 1.0), {3}, False
            g = torch.Generator().manual_seed(11)
            ys = []
            for t in (3, 4, 5):
                x = (torch.randn(6, 256, 4608, generator=g) * float(D[t]) * 40).to(dev)
                if pert:
                    x[2] += 0.37 * float(D[t])
                CTX.step = t
                ys.append(jq._fn(key, x, None).float())
            res.append((ys, {k: v.clone() for k, v in jq.state_out.items()}, {k: v.clone() for k, v in jq.ref.items()}))
        keep = [0, 1, 3, 4, 5]
        out[rule] = bool(all(torch.equal(a[keep], b[keep]) for a, b in zip(res[0][0], res[1][0]))
                         and all(torch.equal(res[0][1][k][keep], res[1][1][k][keep]) for k in res[0][1])
                         and all(torch.equal(res[0][2][k][keep], res[1][2][k][keep]) for k in res[0][2]))
    CTX.step = None
    del m
    return out


@torch.no_grad()
def checks(dev):
    with Run(STAGE, "checks", dev, {"batch_size": BS()}) as run:
        ch = {"provenance": AB.provenance()}
        d = make_diffusion(T)
        ch["synthetic_constant_linear"] = AB.synthetic(d)
        ch["synthetic_alternating"] = alternating_example(d)
        ch["ddim_coefficients"] = ddim_coeff_check(d, dev)
        ch["row_independence_unit"] = quantizer_unit(dev)
        torch.cuda.empty_cache()
        M = load_models(dev)
        jq = M["fb"].jq
        # S1 reproduce overnight G, GB, ACB, FP50 (110000-110031, batch 32, same precision)
        L = torch.load(os.path.join(AB.CK, "s1_latents.pt"), map_location="cpu")
        nz, lb = noise_for_seeds(OVN, dev), labels_for_seeds(OVN)
        ch["S1_reproduce_overnight_bs32"] = {}
        for a in ("FP50", "G", "GB", "ACB"):
            x, c = FC.sample(M, ARMS[a], d, nz, lb)
            ch["S1_reproduce_overnight_bs32"][a] = bool(torch.equal(x.cpu(), L[a][:32]))
        del L
        print(f"  S1 {ch['S1_reproduce_overnight_bs32']}", flush=True)
        n4 = CHK
        nz, lb = noise_for_seeds(n4, dev), labels_for_seeds(n4)
        base = {}
        # S2 lambda 0 == corresponding no-bridge arm, incl. final rolling state and reference
        s2 = {}
        for b, nb in (("GB", "G"), ("ACB", "AC"), ("RFB", "RF"), ("RDB", "RD")):
            x0, _ = FC.sample(M, ARMS[nb], d, nz, lb)
            s0 = snap(jq)
            base[nb] = x0.cpu()
            x1, _ = FC.sample(M, ARMS[b], d, nz, lb, lam=0.0)
            s1 = snap(jq)
            s2[b] = bool(torch.equal(x0, x1) and eq_states(s0[0], s1[0]) and eq_states(s0[1], s1[1]))
        ch["S2_lambda0_equals_no_bridge"] = s2
        # S3 K1 == FP50, zero corrections, all stateful paths (batch 4 references)
        f50, _ = FC.sample(M, ARMS["FP50"], d, nz, lb)
        s3 = {}
        for cheap in ("G", "fb", "R-full", "R-dct", "J"):
            x, c = FC.sample(M, spec(cheap, K=1, bridge=True, gains=(0.5, 0.5)), d, nz, lb)
            s3[cheap] = bool(torch.equal(x, f50) and c == {"cheap": 50, "high": 50, "corrections": 0})
        ch["S3_K1_equals_FP50"] = s3
        # S4 counts, per-site quantizer calls, identical cheap/high anchor inputs
        s4 = {}
        rec_in = {"cheap": [], "high": []}
        hk = [M["fb"].m.x_embedder.register_forward_pre_hook(lambda mod, inp: rec_in["cheap"].append((CTX.step, inp[0].detach().clone()))),
              M["fp"].x_embedder.register_forward_pre_hook(lambda mod, inp: rec_in["high"].append((CTX.step, inp[0].detach().clone())))]
        for a in ("GB", "ACB", "RFB", "RDB", "RD"):
            jq.calls = {k: 0 for k in DJ.KEYS}
            rec_in["cheap"].clear(); rec_in["high"].clear()
            x, c = FC.sample(M, ARMS[a], d, nz, lb)
            hi = dict(rec_in["high"])
            same = [torch.equal(v, hi[s]) for s, v in rec_in["cheap"] if s in hi]
            s4[a] = {"counts": c, "expected": FC.expect(ARMS[a]), "quantizer_calls_per_site": sorted(set(jq.calls.values())),
                     "anchor_inputs_compared": len(same), "anchor_inputs_identical": all(same)}
        for h in hk:
            h.remove()
        ch["S4_counts"] = s4
        ch["S4_pass"] = all(v["counts"] == v["expected"] and v["quantizer_calls_per_site"] == [50] and v["anchor_inputs_compared"] == 13
                            and v["anchor_inputs_identical"] for v in s4.values())
        # S5 sample independence (perturb one sample) and reset between batches
        s5 = {}
        nzp = nz.clone()
        nzp[3] = torch.randn(nzp[3].shape, generator=torch.Generator().manual_seed(7)).to(dev)
        nzb, lbb = noise_for_seeds(list(range(60204, 60208)), dev), labels_for_seeds(list(range(60204, 60208)))
        for a in ("RFB", "RDB", "ACB"):
            x0, _ = FC.sample(M, ARMS[a], d, nz, lb)
            xp, _ = FC.sample(M, ARMS[a], d, nzp, lb)
            xb1, _ = FC.sample(M, ARMS[a], d, nzb, lbb)                     # after batch A
            xb2, _ = FC.sample(M, ARMS[a], d, nzb, lbb)                     # after batch B
            s5[a] = {"others_unchanged": bool(torch.equal(xp[:3], x0[:3])), "reset_between_batches": bool(torch.equal(xb1, xb2))}
            base[a] = x0.cpu()
        ch["S5_independence_reset"] = s5
        # S6 resume immediately before (8) and after (9) an anchor: latent, rolling state and reference
        s6 = {}
        for a, nn in (("RDB", 4), ("RFB", 2), ("ACB", 4)):
            sl = slice(0, nn)
            xr0, _ = FC.sample(M, ARMS[a], d, nz[sl], lb[sl])
            ref_st = snap(jq)
            for sa in (8, 9):
                s_ = FC.sample(M, ARMS[a], d, nz[sl], lb[sl], stop_at=sa)
                p = os.path.join(CK, f"resume_{a}_{sa}.pt")
                atomic_torch_save(s_, p)
                r = torch.load(p, map_location="cpu")
                xr, cr = FC.sample(M, ARMS[a], d, nz[sl], lb[sl], resume=r)
                st_ = snap(jq)
                s6[f"{a}@{sa}"] = bool(torch.equal(xr, xr0) and eq_states(st_[0], ref_st[0]) and eq_states(st_[1], ref_st[1]) and cr == FC.expect(ARMS[a]))
                os.remove(p)                                                 # temporary resume file written by this check
        ch["S6_resume_bitwise"] = s6
        # S7 instrumentation (diag dict + R accounting) on/off bitwise
        s7 = {}
        for a in ("RFB", "RDB"):
            Dg = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
            jq.racct = {}
            xd, _ = FC.sample(M, ARMS[a], d, nz, lb, diag=Dg)
            sd = snap(jq)
            acct = jq.racct
            jq.racct = None
            xn, _ = FC.sample(M, ARMS[a], d, nz, lb)
            sn = snap(jq)
            s7[a] = bool(torch.equal(xd, xn) and torch.equal(xn.cpu(), base[a]) and eq_states(sd[0], sn[0]) and eq_states(sd[1], sn[1]))
            ints = [v for k, v in acct.items() if isinstance(k, int) and k >= 0]
            tot = {}
            for v in ints:
                for kk, vv in v.items():
                    tot[kk] = tot.get(kk, 0.0) + vv
            ch[f"S9_accounting_{a}"] = {"identity_residual_rel_energy": tot["E[identity_residual]"] / tot["E[q-x]"],
                                       "max_abs_identity_residual": acct["max_abs_identity_residual"][0],
                                       "terms_present": sorted(tot), "pass": tot["E[identity_residual]"] / tot["E[q-x]"] <= 1e-8}
        ch["S7_instrumentation_on_off_bitwise"] = s7
        # S8 no anchors: R-dct == ordinary feedback (incl. state); R anchor call == G call (junk state/ref ignored)
        xa, _ = FC.sample(M, dict(ARMS["AC"], K=None), d, nz, lb)
        sa_ = snap(jq)
        xr, cr = FC.sample(M, dict(ARMS["RD"], K=None), d, nz, lb)
        sr_ = snap(jq)
        ch["S8_no_anchor_Rdct_equals_feedback"] = bool(torch.equal(xa, xr) and eq_states(sa_[0], sr_[0]) and sr_[1] == {}
                                                        and cr == {"cheap": 50, "high": 0, "corrections": 0})
        n2 = 2
        xs = torch.randn(n2, 4, 32, 32, generator=torch.Generator().manual_seed(1)).to(dev)
        labt = torch.as_tensor(lb[:n2], device=dev)
        yy = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        tsx = torch.full((n2,), 500, device=dev, dtype=torch.long)
        CTX.step, CTX.n = 20, n2
        FC.configure(jq, ARMS["G"], {20}); jq.reset()
        og = guided(M["fb"].m, xs, tsx, yy, n2)
        s8 = {}
        for a in ("RF", "RD"):
            FC.configure(jq, ARMS[a], {20})
            full = a == "RF"
            jq.state_in = {k: torch.randn((2 * n2, 256 if full else 4, v.shape[-1]), device=dev).half() for k, v in sa_[0].items()}
            jq.ref = {k: torch.randn_like(v.float()).half() for k, v in jq.state_in.items()}
            oa = guided(M["fb"].m, xs, tsx, yy, n2)
            s8[a] = bool(torch.equal(oa, og) and all(float(v.abs().max()) == 0 for v in jq.state_out.values()))
        CTX.step = None
        ch["S8_R_anchor_call_equals_G"] = s8
        # S10 state / reference bytes from real tensors (per image incl. both CFG rows, after a held call)
        s10 = {}
        for a in ("ACB", "RFB", "RDB"):
            s_ = FC.sample(M, ARMS[a], d, nz, lb, stop_at=2)
            s10[a] = {"rolling_state_bytes_per_image": sum(v.numel() * v.element_size() for v in s_["fb_state"].values()) / len(n4),
                      "reference_bytes_per_image": sum(v.numel() * v.element_size() for v in s_["ref"].values()) / len(n4),
                      "dtypes": sorted({str(v.dtype) for v in list(s_["fb_state"].values()) + list(s_["ref"].values())})}
        ch["S10_state_bytes"] = s10
        crit = [ch["synthetic_constant_linear"]["pass"], ch["synthetic_constant_linear"]["coef_K1_zero"], ch["ddim_coefficients"]["pass"],
                *ch["row_independence_unit"].values(), *ch["S1_reproduce_overnight_bs32"].values(), *s2.values(), *s3.values(), ch["S4_pass"],
                *[v["others_unchanged"] and v["reset_between_batches"] for v in s5.values()], *s6.values(), *s7.values(),
                ch["S9_accounting_RFB"]["pass"], ch["S9_accounting_RDB"]["pass"], ch["S8_no_anchor_Rdct_equals_feedback"], *s8.values()]
        ch["pass"] = bool(all(crit))
        ch["peak_mem"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1, default=str), flush=True)
    return 0 if ch["pass"] else 6


# ------------------------------------------------------------------------------------------------------------------- generation
def gen(M, d, seeds, arms, part, est_batch_h):
    bs = BS()
    pf = os.path.join(CK, f"{part}.pt")
    P = torch.load(pf, map_location="cpu") if os.path.exists(pf) else {"done": [], "lat": {a: [] for a in arms}, "cnt": {}, "timing": {}, "bs": bs}
    assert P["bs"] == bs
    t0 = time.time()
    for b0 in range(0, len(seeds), bs):
        if b0 in P["done"]:
            continue
        s = seeds[b0:b0 + bs]
        if spent() + (time.time() - t0) / 3600 + est_batch_h + SHUTDOWN_RESERVE_H > CAP_GPU_H:
            print(f"[rfb1 runtime stop] before batch {s[0]}-{s[-1]}", flush=True)
            return P, False
        nz, lb = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            torch.cuda.synchronize(); ts = time.perf_counter()
            x, c = FC.sample(M, ARMS[a], d, nz, lb)
            torch.cuda.synchronize(); dt = time.perf_counter() - ts
            assert torch.isfinite(x).all() and c == FC.expect(ARMS[a]), (a, c)
            P["lat"][a].append(x.cpu()); P["cnt"][a] = c
            P["timing"].setdefault(a, []).append(dt)
            print(f"  seeds {s[0]}-{s[-1]} {a}: {dt:.1f} s", flush=True)
        P["done"].append(b0)
        atomic_torch_save(P, pf)
    return P, True


def stage_est(arms, n):
    return sum(est_traj_h(a) for a in arms) * (n / BS()) * 1.10 + 0.04 * len(arms) * n / 128


def run_quality(dev, name, seeds, reserve_extra):
    arms = QUALITY_ARMS
    est = stage_est(arms, len(seeds))
    if not fits(name, est, SHUTDOWN_RESERVE_H + reserve_extra):
        jdump(f"{name}.json", {"status": "skipped (budget)", "estimate_h": est})
        return None
    with Run(STAGE, name, dev, {"seeds": [seeds[0], seeds[-1]], "arms": arms, "batch": BS()}) as run:
        AB.provenance()
        M = load_models(dev)
        d = make_diffusion(T)
        torch.cuda.reset_peak_memory_stats()
        P, full = gen(M, d, seeds, arms, f"{name}_part", sum(est_traj_h(a) for a in arms))
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        del M
        torch.cuda.empty_cache()
        if not full:
            r = {"status": "partial (stopped by budget)", "batches_done": len(P["done"]), **run.record()}
            jdump(f"{name}.json", r)
            return None
        lat = AB.cat(P, arms)
        atomic_torch_save(lat, os.path.join(CK, f"{name}_latents.pt"))
        Mx, C = AB.evaluate(lat, dev, PAIRS)
        out = {"status": "complete", "seeds": [seeds[0], seeds[-1]], "n": len(seeds), "batch": BS(), "arms": Mx, "summary": summary(Mx),
               "calls": P["cnt"], "contrasts": C, "timing_s_per_batch": {a: {"median": float(np.median(v)), "all": v} for a, v in P["timing"].items()},
               "peak_mem_harness": peak, **run.record()}
    AB.show(Mx, C)
    return out


def dev_stage(dev):
    conf_est = stage_est(QUALITY_ARMS, len(CONF))
    out = run_quality(dev, "dev", DEV, conf_est + 0.25)                       # confirmation (all six arms + metrics) and diagnostics reserved first
    if out is None:
        return 4
    S = out["summary"]
    gate = {}
    for a in ("RFB", "RDB"):
        gate[a] = {"ratio_vs_GB": S[a]["rel_mse"]["mean"] / S["GB"]["rel_mse"]["mean"], "dlpips_vs_GB": S[a]["lpips"]["mean"] - S["GB"]["lpips"]["mean"]}
        gate[a]["pass"] = gate[a]["ratio_vs_GB"] <= 0.95 and gate[a]["dlpips_vs_GB"] <= 0.005
    out["gate"] = gate
    out["gate_pass"] = any(v["pass"] for v in gate.values())
    out["gate_rule"] = "run confirmation iff RFB or RDB dev mean error <= 0.95 x GB and mean LPIPS <= LPIPS(GB) + 0.005"
    out["note"] = "development set 60200-60231 already inspected; exploratory; freshness not claimed"
    jdump("dev.json", out)
    print(f"  gate: {gate} -> {'RUN CONFIRMATION' if out['gate_pass'] else 'STOP R CONFIRMATION'}", flush=True)
    return 0


def conf_stage(dev):
    dv = jload("dev.json")
    if not (dv and dv.get("status") == "complete"):
        jdump("conf.json", {"status": "blocked (dev incomplete)"})
        return 4
    if not dv["gate_pass"]:
        jdump("conf.json", {"status": "skipped (development gate failed; confirmation cohort 122000-122127 not opened)", "gate": dv["gate"]})
        return 4
    out = run_quality(dev, "conf", CONF, 0.20)
    if out is None:
        return 4
    C = out["contrasts"]
    out["primary"] = "RDB_vs_GB"
    out["readings"] = {k: AB.reading(C[k]) for k in ("RDB_vs_GB", "RFB_vs_GB", "RDB_vs_RFB", "RDB_vs_ACB", "RD_vs_G", "RDB_vs_RD")}
    out["labels"] = {"RDB_vs_GB": "primary practical", "RFB_vs_GB": "secondary full-state mechanism", "RDB_vs_RFB": "compression",
                     "RDB_vs_ACB": "repair of the harmful stack only", "RD_vs_G": "interaction (no additive claim)", "RDB_vs_RD": "interaction"}
    out["note"] = "secondary intervals exploratory / unadjusted; nonsignificance is not equivalence"
    jdump("conf.json", out)
    return 0


# ------------------------------------------------------------------------------------------------------------------- diagnostics
@torch.no_grad()
def diag_stage(dev):
    if not fits("diag", 0.40, SHUTDOWN_RESERVE_H):
        jdump("diag.json", {"status": "skipped (budget)"})
        return 4
    arms = ["G", "GB", "ACB", "RFB", "RDB", "RD"]
    with Run(STAGE, "diag", dev, {"seeds": [DIAG[0], DIAG[-1]], "arms": arms}) as run:
        M = load_models(dev)
        jq = M["fb"].jq
        d = make_diffusion(T)
        Ac, bc = FC.coeffs(d)
        nz, lb = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        n = len(DIAG)
        labt = torch.as_tensor(lb, device=dev)
        y = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        f50, _ = FC.sample(M, ARMS["FP50"], d, nz, lb)
        out = {"status": "complete", "seeds": [DIAG[0], DIAG[-1]], "arms": {},
               "definitions": {"e_i": "cheap - FP guided prediction (channels 0-3) at the live latent (same state; FP call diagnostic only)",
                               "H": "sum_j w_j (e_{a+j} - e_a)", "v": "bridge estimate (e_c - e_a) sum_j (j/k) w_j",
                               "replay": "FP replay from x_{a+1} over the interval (starts 0, 20, 40) vs the live pre-correction x_c",
                               "R accounting": "per interval (anchor a), summed over all 112 sites and both CFG rows: energies of each term"},
               "limitation": "each arm on its own trajectory; 8 development images; diagnostics never feed live inference"}
        for name in arms:
            D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
            jq.racct = {} if name.startswith("R") else None
            jq.acct = {} if name in ("ACB",) else None
            x, c = FC.sample(M, ARMS[name], d, nz, lb, diag=D)
            racct, acct = jq.racct, jq.acct
            jq.racct, jq.acct = None, None
            r = FC.interval_stats(D["e"], Ac, bc)
            r["endpoint_rel_mse_vs_FP50_mean"] = float(np.mean([((x[i] - f50[i]).pow(2).sum() / f50[i].pow(2).sum()).item() for i in range(n)]))
            r["bridge_jump_norms"] = D["jump"]
            r["held_discrepancy_energy_by_step"] = {i: float(v.pow(2).sum()) for i, v in D["e"].items()}
            rep = {}
            for a in (0, 20, 40):
                xr = D["x"][a + 1]
                for j in range(a + 1, a + 4):
                    CTX.step, CTX.n = j, n
                    t = torch.full((n,), T - 1 - j, device=dev, dtype=torch.long)
                    xr = d.ddim_sample(lambda xx, ts: guided(M["fp"], xx, ts, y, n), xr, t, clip_denoised=False, eta=0.0)["sample"]
                dx = (D["x"][a + 4] - xr).double().cpu()
                w = FC.weights(Ac, bc, a, 4)
                H = sum(wj * (D["e"][a + j] - D["e"][a]) for j, wj in zip(range(1, 4), w))
                hh = FC.bridge_coef(Ac, bc, a, 4) * (D["e"][a + 4] - D["e"][a])
                rep[a] = {"cos_dx_v": float((dx * hh).sum() / (dx.norm() * hh.norm())), "v_over_dx": float(hh.norm() / dx.norm()),
                          "cos_dx_H": float((dx * H).sum() / (dx.norm() * H.norm())), "residual_after_bridge": float((dx - hh).pow(2).sum() / dx.pow(2).sum()),
                          "dx_energy": float(dx.pow(2).sum())}
            CTX.step = None
            r["replay"] = rep
            if racct:
                r["R_accounting_per_interval"] = {k: v for k, v in racct.items() if isinstance(k, int)}
                r["R_max_abs_identity_residual"] = racct.get("max_abs_identity_residual", [None])[0]
            if acct:
                tot = {}
                for v in acct.values():
                    for kk, vv in v.items():
                        tot[kk] = tot.get(kk, 0.0) + vv
                r["ACB_feedback_accounting_all_sites"] = tot
            r["calls"] = c
            out["arms"][name] = r
            print(f"  {name}: raw {r['raw']:.4g} held {r['held']:.4g} cos(H,v) {r['cos_H_v']:.3f} endpoint {r['endpoint_rel_mse_vs_FP50_mean']:.5f}", flush=True)
        # memory and one warmed synchronized runtime comparison at the frozen batch (fake-quant simulation; not packed-kernel evidence)
        bs = BS()
        s = (DEV + DEV)[:bs]
        nzb, lbb = noise_for_seeds(s, dev), labels_for_seeds(s)
        tm = {}
        for a in ("FP50", "GB", "ACB", "RFB", "RDB"):
            ts = []
            for _ in range(2):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                FC.sample(M, ARMS[a], d, nzb, lbb)
                torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
            sb = jq.state_bytes() / bs if "high_only" not in ARMS[a] else 0
            tm[a] = {"warm_s_per_trajectory": ts[1], "warmup_s": ts[0], "batch": bs, "state_plus_reference_bytes_per_image_incl_cfg": sb}
            print(f"  timing {a}: {ts[1]:.1f} s, state {sb / 1e6:.2f} MB/image", flush=True)
        out["timing_memory"] = tm
        out["timing_note"] = "one warm-up + one timed synchronized trajectory per arm, all models resident; fake-quant simulation, no packed-kernel claim"
        out.update(run.record())
    jdump("diag.json", out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["profile", "checks", "dev", "conf", "diag"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if a.cmd in ("dev", "conf", "diag"):
        ch = jload("checks.json")
        if not (ch and ch["pass"]):
            print("[rfb1] checks have not passed: stage blocked", flush=True)
            jdump(f"{a.cmd}.json", {"status": "blocked (checks not passed)"})
            sys.exit(4)
    rc = {"profile": profile, "checks": checks, "dev": dev_stage, "conf": conf_stage, "diag": diag_stage}[a.cmd](dev)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
