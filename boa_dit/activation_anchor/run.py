"""Activation feedback + measured anchoring (results/activation_anchor/PROTOCOL.md). Stage tag aa1.
Cheap model: common smoothed rank-8 W4 base + original carry + static A8 tables, JointQuant 'static' (Q) or 'fb' (DCT-4 feedback, A) from
dit_joint.run (unchanged). Teacher/anchor/scorer: FP model. Anchor loop as weight_anchor.w8.sample_arm (cheap at all 50 steps, FP at 13 anchors).
Arms: Q A C AC C_anchor_only AC_anchor_only FP50.
Stages: checks | dev | diag | conf
Run: python -m boa_dit.activation_anchor.run <stage>
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.dit_joint import run as DJ
from boa_dit.model import NULL_CLASS, LINEARS, block_linear, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import guided, ratio_ci
from boa_dit.weight_anchor.shared import boot_idx, ci, contrast, summary

STAGE = "aa1"
OUT = os.path.join(RESULTS, "activation_anchor")
CK = os.path.join(CKPT_DIR, "activation_anchor")
CAP_GPU_H = 1.5
CONF_RESERVE_H = 0.35
K = 4
DEV, CONF, DIAG, G1 = list(range(60200, 60216)), list(range(96100, 96164)), list(range(60200, 60208)), list(range(61000, 61032))
BS = 32
ARMS = {"Q": ("static", "plain"), "A": ("fb", "plain"), "C": ("static", "corr"), "AC": ("fb", "corr"),
        "C_anchor_only": ("static", "ao"), "AC_anchor_only": ("fb", "ao"), "FP50": (None, "fp")}
ORDER = ("FP50", "Q", "A", "C", "AC", "C_anchor_only", "AC_anchor_only")
HASHES = {"base_model": (os.path.join(CKPT_DIR, "ckpt_deflate_heig_r8_w4_s", "model.pt"), "f017af877a79015fb9670504da4fb9fb51df30e1d8e94f6b4a3deedd5aec7ced"),
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
    print(f"[aa1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        sys.exit(5)


def provenance():
    out = {k: {"path": os.path.relpath(p, os.path.dirname(RESULTS)) if p.startswith(os.path.dirname(RESULTS)) else os.path.relpath(p, CKPT_DIR),
               "sha256": DJ.sha(p), "expected": h} for k, (p, h) in HASHES.items()}
    assert all(v["sha256"] == v["expected"] for v in out.values()), out
    return out


# ------------------------------------------------------------------------------------------------------------------- models
def models(dev):
    """FP teacher + one cheap model per feedback mode (separate instances; FP carries no hooks)."""
    fp = load_dit(dev)
    fp.requires_grad_(False)
    cheap = {}
    for mode in ("static", "fb"):
        m, jq, _ = DJ.student(dev, None, mode)
        jq.t, jq.auto = None, True
        cheap[mode] = (m, jq)
    return fp, cheap


def fp_isolated(fp, cheap):
    no_hooks = all(len(block_linear(fp.blocks[l], ln)._forward_pre_hooks) == 0 and len(block_linear(fp.blocks[l], ln)._forward_hooks) == 0
                   for l in range(28) for ln in LINEARS)
    ptr = {p.data_ptr() for p in fp.parameters()}
    disjoint = all(p.data_ptr() not in ptr for m, _ in cheap.values() for p in m.parameters())
    return bool(no_hooks and disjoint)


# ------------------------------------------------------------------------------------------------------------------ sampler
@torch.no_grad()
def sample(fp, cm, jq, d, noise, labels, kind, k=K, zero_corr=False, zero_aux=False, rec=None, diag=None, stop_at=None, resume=None):
    """One trajectory. kind: 'plain' (cheap only), 'corr' (anchors + reuse), 'ao' (anchors only), 'fp' (FP only).
    cm/jq: cheap model and its JointQuant (jq None if cm is a stateless model, e.g. FP for G5). k=None: no anchors.
    diag: dict collecting raw / used channel-0..3 errors against a same-state FP call (diagnostic only).
    stop_at: return the full sampler state after that many steps; resume: continue from such a state."""
    n = len(noise)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    if resume is None:
        x, i0 = noise.float().clone(), 0
        st = {"d": None, "a": None}
        cnt = {"cheap": 0, "fp": 0}
        if jq is not None:
            jq.state_in, jq.state_out = {}, {}                                   # reset feedback state per batch
    else:
        x, i0 = resume["x"].to(noise.device), resume["i"]
        st = {"d": None if resume["d"] is None else resume["d"].to(noise.device), "a": resume["a"]}
        cnt = dict(resume["cnt"])
        torch.set_rng_state(resume["rng_cpu"]); torch.cuda.set_rng_state(resume["rng_cuda"])
        if jq is not None:
            jq.state_in = {kk: v.to(noise.device) for kk, v in resume["fb_state"].items()}
            jq.state_out = {}
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        if i < i0:
            continue
        if stop_at is not None and i == stop_at:
            return {"x": x.cpu(), "i": i, "d": None if st["d"] is None else st["d"].cpu(), "a": st["a"], "cnt": dict(cnt),
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "fb_state": {} if jq is None else {kk: v.detach().clone().cpu() for kk, v in jq.state_in.items()}}
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            def call(m, name):
                cnt[name] += 1
                if rec is not None:
                    rec.append({"i": _i, "model": name, "x": xx.detach().clone().cpu(), "t": ts.detach().clone().cpu(), "y": y.detach().clone().cpu(), "a": st["a"]})
                return guided(m, xx, ts, y, n)
            if kind == "fp":
                return call(fp, "fp")
            rc = call(cm, "cheap")                                              # cheap call every step: feedback state advances once
            anchor = kind != "plain" and k is not None and _i % k == 0
            if anchor:
                rf = call(fp, "fp")
                if kind == "corr":
                    dd = rc[:, :4] - rf[:, :4]
                    st["d"], st["a"] = (dd * 0.0 if zero_corr else dd), _i
                res = rf
            elif kind == "corr" and st["d"] is not None:
                res = rc.clone()
                res[:, :4] -= st["d"]
            else:
                res = rc
            if zero_aux:
                res = res.clone()
                res[:, 4:] = 0
            if diag is not None:
                ref = rf if anchor else guided(fp, xx, ts, y, n)                 # diagnostic-only FP call off-anchor
                diag["raw"].append((rc[:, :4] - ref[:, :4]).cpu())
                diag["used"].append((res[:, :4] - ref[:, :4]).cpu())
                diag["anchor"].append(bool(anchor))
            return res
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt


EXPECT = {"plain": {"cheap": 50, "fp": 0}, "corr": {"cheap": 50, "fp": 13}, "ao": {"cheap": 50, "fp": 13}, "fp": {"cheap": 0, "fp": 50}}


def run_arm(fp, cheap, d, nz, lab, arm, **kw):
    mode, kind = ARMS[arm]
    cm, jq = cheap[mode] if mode else (None, None)
    return sample(fp, cm, jq, d, nz, lab, kind, **kw)


def gen(fp, cheap, d, seeds, bs, arms=ORDER, t0=None, base=0.0, est_batch_h=0.0, timing=None):
    lat, cnt = {a: [] for a in arms}, {}
    clip = {}
    for b in range(0, len(seeds), bs):
        s = seeds[b:b + bs]
        if t0 is not None and base + (time.time() - t0) / 3600 + est_batch_h > CAP_GPU_H:
            print(f"[aa1 runtime stop] before batch {s[0]}-{s[-1]}", flush=True)
            return {a: torch.cat(v) for a, v in lat.items() if v}, cnt, clip, False
        nz, lab = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            mode = ARMS[a][0]
            if mode:
                cheap[mode][1].clip.zero_()
            torch.cuda.synchronize()
            ts = time.perf_counter()
            x, c = run_arm(fp, cheap, d, nz, lab, a)
            torch.cuda.synchronize()
            if timing is not None:
                timing.setdefault(a, []).append(time.perf_counter() - ts)
            assert torch.isfinite(x).all() and c == EXPECT[ARMS[a][1]], (a, c)
            if mode:
                cl = cheap[mode][1].clip
                clip.setdefault(a, []).append(float(cl[:, 0].sum() / cl[:, 1].sum().clamp_min(1)))
            lat[a].append(x.cpu())
            cnt[a] = c
        print(f"  seeds {s[0]}-{s[-1]}: {', '.join(arms)} done", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt, clip, True


# ------------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def checks(dev):
    guard("checks", 0.25)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "checks", dev, {"G1_seeds": [G1[0], G1[-1]], "dev": [DEV[0], DEV[-1]]}) as run:
        ch = {"provenance": provenance()}
        fp, cheap = models(dev)
        ch["fp_isolated_no_hooks_disjoint_params"] = fp_isolated(fp, cheap)
        d = make_diffusion(50)
        # G1: reproduce the joint pilot's cached static / DCT-4 / FP latents (61000-61031, batch 32)
        L = torch.load(os.path.join(DJ.CK, "eval_latents.pt"), map_location="cpu")
        xf_cache = torch.load(os.path.join(DJ.CK, "fp_sealed.pt"), map_location="cpu")
        nz, lab = noise_for_seeds(G1, dev), labels_for_seeds(G1)
        g1 = {}
        for arm, ref in (("Q", L["A"][:32]), ("A", L["B"][:32]), ("FP50", xf_cache[:32])):
            x, c = run_arm(fp, cheap, d, nz, lab, arm)
            g1[arm] = bool(torch.equal(x.cpu(), ref))
            print(f"  G1 {arm}: {g1[arm]}", flush=True)
        ch["G1_reproduce_joint_pilot_batch32"] = g1
        # development seeds, batch 16
        nz, lab = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        ref = {}
        for arm in ORDER:
            ref[arm], c = run_arm(fp, cheap, d, nz, lab, arm)
            ref[arm] = ref[arm].cpu()
            ch.setdefault("G6_counts", {})[arm] = c
        ch["G6_counts_pass"] = all(ch["G6_counts"][a] == EXPECT[ARMS[a][1]] for a in ORDER)
        atomic_torch_save(ref, os.path.join(CK, "checks_dev_latents.pt"))
        # G2: correction disabled == anchor-only
        ch["G2_zero_correction_equals_anchor_only"] = {}
        for arm, ao in (("C", "C_anchor_only"), ("AC", "AC_anchor_only")):
            x, _ = run_arm(fp, cheap, d, nz, lab, arm, zero_corr=True)
            ch["G2_zero_correction_equals_anchor_only"][arm] = bool(torch.equal(x.cpu(), ref[ao]))
        # G3: K = 1 -> FP50
        ch["G3_K1_equals_FP50"] = {}
        for arm in ("AC", "AC_anchor_only"):
            x, c = run_arm(fp, cheap, d, nz, lab, arm, k=1)
            ch["G3_K1_equals_FP50"][arm] = bool(torch.equal(x.cpu(), ref["FP50"]))
        # G4: no-anchor mode == Q / A
        ch["G4_no_anchor_equals_plain"] = {}
        for arm, pl in (("C", "Q"), ("AC", "A")):
            x, c = run_arm(fp, cheap, d, nz, lab, arm, k=None)
            ch["G4_no_anchor_equals_plain"][arm] = bool(torch.equal(x.cpu(), ref[pl]) and c == {"cheap": 50, "fp": 0})
        # G5: cheap := FP (stateless, deterministic) -> zero discrepancy, FP50 trajectory
        x, _ = sample(fp, fp, None, d, nz, lab, "corr")
        ch["G5_identical_low_high_equals_FP50"] = bool(torch.equal(x.cpu(), ref["FP50"]))
        dz = []
        x2, _ = sample(fp, fp, None, d, nz[:4], lab[:4], "corr", diag={"raw": dz, "used": [], "anchor": []})
        ch["G5_zero_discrepancy"] = bool(all(torch.count_nonzero(v) == 0 for v in dz))
        # G6: anchor inputs identical; one feedback update per cheap call
        jq = cheap["fb"][1]
        jq.calls = {kk: 0 for kk in DJ.KEYS}
        rec = []
        x, c = run_arm(fp, cheap, d, nz, lab, "AC", rec=rec)
        ok = True
        for i in range(50):
            ci_ = [r for r in rec if r["i"] == i]
            if i % K == 0:
                a_, b_ = [r for r in ci_ if r["model"] == "cheap"], [r for r in ci_ if r["model"] == "fp"]
                ok &= len(a_) == 1 and len(b_) == 1 and all(torch.equal(a_[0][f], b_[0][f]) for f in ("x", "t", "y"))
            else:
                ok &= len(ci_) == 1 and ci_[0]["model"] == "cheap" and ci_[0]["a"] == i - i % K
        del rec
        ch["G6_anchor_inputs_identical"] = bool(ok)
        ch["G6_fb_updates_per_site_per_trajectory"] = sorted(set(jq.calls.values()))
        ch["G6_AC_rerun_bitwise"] = bool(torch.equal(x.cpu(), ref["AC"]))
        # G7: save / resume
        s30 = sample(fp, cheap["fb"][0], jq, d, nz, lab, "corr", stop_at=30)
        p = os.path.join(CK, "resume_test.pt")
        atomic_torch_save(s30, p)
        loaded = torch.load(p, map_location="cpu")
        x, c = sample(fp, cheap["fb"][0], jq, d, nz, lab, "corr", resume=loaded)
        ch["G7_save_resume_bitwise"] = bool(torch.equal(x.cpu(), ref["AC"]))
        ch["G7_counts_after_resume"] = c
        ch["G7_saved_fields"] = sorted(loaded.keys())
        # G8: finite; variance channels unused
        ch["G8_finite"] = bool(all(torch.isfinite(v).all() for v in ref.values()))
        x, _ = run_arm(fp, cheap, d, nz, lab, "Q", zero_aux=True)
        ch["G8_zero_aux_equals_Q"] = bool(torch.equal(x.cpu(), ref["Q"]))
        flat = [ch["fp_isolated_no_hooks_disjoint_params"], *g1.values(), ch["G6_counts_pass"], *ch["G2_zero_correction_equals_anchor_only"].values(),
                *ch["G3_K1_equals_FP50"].values(), *ch["G4_no_anchor_equals_plain"].values(), ch["G5_identical_low_high_equals_FP50"], ch["G5_zero_discrepancy"],
                ch["G6_anchor_inputs_identical"], ch["G6_fb_updates_per_site_per_trajectory"] == [50], ch["G6_AC_rerun_bitwise"], ch["G7_save_resume_bitwise"],
                ch["G8_finite"], ch["G8_zero_aux_equals_Q"]]
        ch["pass"] = bool(all(flat))
        ch["peak_mem"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k_: v for k_, v in ch.items() if k_ not in ("config", "provenance")}, indent=1), flush=True)
    if not ch["pass"]:
        print("[aa1] checks FAILED: stop", flush=True)
        sys.exit(6)


# --------------------------------------------------------------------------------------------------------------- statistics
PAIRS = (("AC", "C"), ("A", "Q"), ("C", "Q"), ("AC", "Q"), ("AC", "A"), ("C", "C_anchor_only"), ("AC", "AC_anchor_only"), ("AC_anchor_only", "C_anchor_only"))


def interaction(Mx):
    q, a, c, ac = (np.asarray(Mx[x]["rel_mse"]) for x in ("Q", "A", "C", "AC"))
    idx = boot_idx(len(q))
    I = lambda i: (c[i].mean() - ac[i].mean()) - (q[i].mean() - a[i].mean())
    R = lambda i: (ac[i].mean() / c[i].mean()) / (a[i].mean() / q[i].mean())
    al = np.arange(len(q))
    return {"I_abs": float(I(al)), "I_abs_ci": ci([I(i) for i in idx]), "feedback_gain_with_anchors": float(c.mean() - ac.mean()),
            "feedback_gain_with_anchors_ci": ci([c[i].mean() - ac[i].mean() for i in idx]), "feedback_gain_without_anchors": float(q.mean() - a.mean()),
            "feedback_gain_without_anchors_ci": ci([q[i].mean() - a[i].mean() for i in idx]),
            "rel_ratio_of_ratios": float(R(al)), "rel_ratio_of_ratios_ci": ci([R(i) for i in idx]),
            "note": "I > 0: feedback's absolute benefit grows with anchoring; ratio of ratios < 1: feedback's relative benefit grows with anchoring"}


def reading(c):
    m, lp = c["rel_mse"], c["lpips"]
    if m["ratio"] <= 0.95 and m["ratio_ci"][1] < 1 and not lp["diff_ci"][0] > 0:
        return "useful additional feedback signal (>= 5% lower mean latent MSE, CI excludes 0, no clear LPIPS regression)"
    if m["ratio_ci"][1] < 1:
        return "smaller positive" + (" (LPIPS clearly worse)" if lp["diff_ci"][0] > 0 else " (below the 5% threshold)")
    if m["ratio_ci"][0] > 1:
        return "harmful (AC worse than C)"
    return "inconclusive"


def evaluate(lat, dev):
    Mx = metrics(lat, dev)
    C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in PAIRS}
    return Mx, C


def show(out):
    for a, v in out["summary"].items():
        print(f"  {a}: rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f} PSNR {v['psnr']['mean']:.2f} LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in out["contrasts"].items():
        m = v["rel_mse"]
        print(f"  {k_}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] diff {m['diff']:+.5f} wins {m['wins_x']}/{m['n']} "
              f"dLPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['diff_ci'][0]:+.4f}, {v['lpips']['diff_ci'][1]:+.4f}]", flush=True)
    print("  interaction:", json.dumps(out["interaction"]), flush=True)
    print("  reading AC vs C:", out["reading_AC_vs_C"], flush=True)


# ---------------------------------------------------------------------------------------------------------------------- dev
@torch.no_grad()
def dev_stage(dev):
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        sys.exit(6)
    guard("dev", 0.12)
    with Run(STAGE, "dev", dev, {"seeds": [DEV[0], DEV[-1]], "batch": 16}) as run:
        f = os.path.join(CK, "checks_dev_latents.pt")
        lat = torch.load(f, map_location="cpu")                                  # the G-stage reference runs: same seeds, batch, models, code
        cnt = ch["G6_counts"]
        Mx, C = evaluate(lat, dev)
        out = {"seeds": [DEV[0], DEV[-1]], "note": "development (already-inspected design seeds); exploratory; nothing selected; latents = the checks-stage "
               "reference runs (identical configuration)", "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C,
               "interaction": interaction(Mx), "reading_AC_vs_C": reading(C["AC_vs_C"]), **run.record()}
    jdump("dev.json", out)
    show(out)


# --------------------------------------------------------------------------------------------------------------------- diag
def lag_stats(E, mask_valid=None):
    """E: [N, 50, 4, 32, 32] float64 errors; pooled over all pairs (a, t=a+L) and images."""
    sq = lambda X: X.flatten(2).pow(2).sum(2)
    out = {}
    for L in (1, 2, 4, 8):
        a = torch.arange(0, 50 - L)
        t = a + L
        Et, Ea = E[:, t], E[:, a]
        et2, ea2 = sq(Et), sq(Ea)
        cross = (Et * Ea).flatten(2).sum(2)
        diff = sq(Et - Ea)
        S = {"E_t": float(et2.sum()), "E_a": float(ea2.sum()), "cross": float(cross.sum()), "sum_diff": float(diff.sum())}
        S["rho"] = S["cross"] / (S["E_t"] * S["E_a"]) ** 0.5
        S["identity_residual_rel"] = abs(S["sum_diff"] - (S["E_t"] + S["E_a"] - 2 * S["cross"])) / S["sum_diff"]
        S["ratio_diff_over_Et"] = ratio_ci(diff.sum(1).numpy(), et2.sum(1).numpy())
        S["E_t_over_E_a"] = S["E_t"] / S["E_a"]
        S["pairs_per_image"] = int(len(a))
        out[str(L)] = S
    return out


@torch.no_grad()
def diag(dev):
    guard("diag (optional)", 0.08)
    with Run(STAGE, "diag", dev, {"seeds": [DIAG[0], DIAG[-1]], "arms": ["Q", "A", "C", "AC"]}) as run:
        fp, cheap = models(dev)
        d = make_diffusion(50)
        nz, lab = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        out = {"seeds": [DIAG[0], DIAG[-1]], "pooling": "sums over all pairs (a, t=a+L), 0<=a<=49-L, and all 8 images; per-image sums for the bootstrap",
               "note": "within-arm, own-trajectory errors e_t = f_cheap - f_FP (channels 0-3) against same-state FP calls (diagnostic only); cross-arm "
                       "differences include state-distribution changes", "arms": {}}
        for arm in ("Q", "A", "C", "AC"):
            D = {"raw": [], "used": [], "anchor": []}
            mode = ARMS[arm][0]
            cheap[mode][1].clip.zero_()
            x, c = run_arm(fp, cheap, d, nz, lab, arm, diag=D)
            cl = cheap[mode][1].clip
            raw, used = torch.stack(D["raw"], 1).double(), torch.stack(D["used"], 1).double()
            anc = torch.tensor(D["anchor"])
            sq = lambda X: X.flatten(2).pow(2).sum(2)
            r = {"calls_inference": c, "clip_rate_representable_range": float(cl[:, 0].sum() / cl[:, 1].sum().clamp_min(1)),
                 "residual_clamp_and_discarded_subspace": "not measured (frozen quantizer exposes only the clip counter)",
                 "per_step_raw_energy_mean": sq(raw).mean(0).tolist(), "lags_raw": lag_stats(raw)}
            if ARMS[arm][1] == "corr":
                na = ~anc
                t_idx = torch.arange(50)[na]
                a_idx = t_idx - t_idx % K
                pred = sq(raw[:, t_idx] - raw[:, a_idx])                            # anchor-structured identity: e_t - e_a(t)
                r["corrected_energy_nonanchor"] = float(sq(used[:, na]).sum())
                r["raw_energy_nonanchor"] = float(sq(raw[:, na]).sum())
                r["corrected_over_raw_nonanchor"] = ratio_ci(sq(used[:, na]).sum(1).numpy(), sq(raw[:, na]).sum(1).numpy())
                r["anchor_identity_rel_residual"] = float((sq(used[:, na]) - pred).abs().sum() / pred.sum())
                r["per_step_used_energy_mean"] = sq(used).mean(0).tolist()
            out["arms"][arm] = r
            print(f"  {arm}: lag ratios " + ", ".join(f"L{L} {v['ratio_diff_over_Et']['v']:.3f} (rho {v['rho']:.3f})" for L, v in r["lags_raw"].items())
                  + (f"; corrected/raw {r['corrected_over_raw_nonanchor']['v']:.3f}" if "corrected_over_raw_nonanchor" in r else ""), flush=True)
        out.update(run.record())
    jdump("diag.json", out)


# --------------------------------------------------------------------------------------------------------------------- conf
@torch.no_grad()
def conf(dev):
    if not (jload("frozen.json") and jload("dev.json")):
        print("[aa1] dev and a committed frozen.json are required", flush=True)
        sys.exit(6)
    guard("conf", CONF_RESERVE_H, reserve_h=0.0)
    t0, base = time.time(), spent()
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "batch": BS}) as run:
        fp, cheap = models(dev)
        d = make_diffusion(50)
        torch.cuda.reset_peak_memory_stats()
        timing = {}
        lat, cnt, clip, full = gen(fp, cheap, d, CONF, BS, t0=t0, base=base, est_batch_h=0.16, timing=timing)
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt" if full else "conf_latents_partial.pt"))
        if not full:
            jdump("conf_partial.json", {"calls": cnt, **run.record()})
            sys.exit(5)
        del fp, cheap
        torch.cuda.empty_cache()
        Mx, C = evaluate(lat, dev)
        out = {"seeds": [CONF[0], CONF[-1]], "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C, "interaction": interaction(Mx),
               "reading_AC_vs_C": reading(C["AC_vs_C"]), "clip_rate": clip,
               "timing_s_per_batch32": {a: {"median": float(np.median(v)), "all": v} for a, v in timing.items()},
               "timing_note": "CUDA-synchronized wall time per trajectory, diagnostics off, all three models resident; simulation (fake-quant, fp32 weights)",
               "peak_mem_live": peak, **run.record()}
    jdump("conf.json", out)
    show(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "dev", "diag", "conf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"checks": checks, "dev": dev_stage, "diag": diag, "conf": conf}[a.cmd](dev)


if __name__ == "__main__":
    main()
