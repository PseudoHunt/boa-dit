"""Analytic carry correction + trajectory-response calibration (results/analytic_carry/PROTOCOL.md). Stage tag ac1.
Bed: the joint pilot's COMMON smoothed DiT-XL/2 W4 rank-8 base, static single8 A8, feedback off, DDIM-50, cfg 1.5 (boa_dit.dit_joint.run).
Only the existing carry factors change: V [d_in, 8], delta (= D) [d_out, 8]; carry(x) = (x V) D^T.
  E1: four depth-group coefficients on the factor interpolation C0 -> C_L u200 (joint pilot, Adam).
  E2: four depth-group coefficients on analytic ridge directions dD (V fixed), from paired student/FP layer statistics (no Adam).
Stages (resumable; each skips itself when its output exists):
  audit checks profile  e1_resp e1_select e1_confirm  e2_stats e2_solve e2_resp e2_select e2_confirm  report
Run: python -m boa_dit.analytic_carry.run <stage>
"""
import argparse
import csv
import hashlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as J
from boa_dit.followup.carry_opt import Sched, guided_eps
from boa_dit.model import block_linear, labels_for_seeds, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample
from boa_dit.temporal import act as ACT

STAGE = "ac1"
OUT = os.path.join(RESULTS, "analytic_carry")
CK = os.path.join(CKPT_DIR, "analytic_carry")
T, CFG = J.T, J.CFG
CAP_GPU_H = 4.0
ALLOC = {"audit_checks": 0.15, "e1": 1.10, "e2": 2.25, "reserve": 0.50}
SETS = {"e1": {"fit": list(range(82000, 82032)), "dev": list(range(82100, 82132)), "conf": list(range(82200, 82328))},
        "e2": {"fit": list(range(83000, 83032)), "dev": list(range(83100, 83132)), "conf": list(range(83200, 83328))}}
GROUPS = [range(0, 7), range(7, 14), range(14, 21), range(21, 28)]
LO, HI = -0.25, 1.0
STEPS = list(range(0, 50, 5)) + [49]                                         # local target / E2 statistics: 11 execution steps
NTOK = 64
H0, H_CHECK, H_RETRY = 0.1, 0.05, 0.2
COL_TOL, MIX_TOL, NOISE_MULT = 0.25, 0.25, 10.0
GAMMAS, SHRINK, ALPHAS = (0.001, 0.01, 0.1, 1.0), (1.0, 0.5, 0.25), (0.0, 0.1, 0.3, 0.5, 1.0)
MIXED = (1.0, -1.0, 1.0, -1.0)
SRC_E1 = os.path.join(CKPT_DIR, "dit_joint", "carry_C_L_u200.pt")
KEYS = J.KEYS


def group_of(key):
    return int(key.split(".")[0][1:]) // 7


def sha_file(p):
    return J.sha(p)


def sha_carry(c):
    h = hashlib.sha256()
    for k in KEYS:
        for n in ("V", "delta"):
            h.update(c[k][n].contiguous().numpy().tobytes())
    return h.hexdigest()


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def done(name):
    return os.path.exists(os.path.join(OUT, name))


# ------------------------------------------------------------------------------------------------ budget (shared 4.0 GPU-h ceiling)
def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


class Ceiling(Exception):
    pass


class Budget:
    """refuses a stage whose estimate does not fit; inside a stage, over() = spent + this stage's elapsed > ceiling (save partial, exit 5)."""

    def __init__(self, label, est_h, keep_h=0.0):
        self.label, self.t0 = label, time.time()
        s = spent()
        self.ok = s + est_h + keep_h <= CAP_GPU_H
        b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "allocations_h": ALLOC, "log": []}
        b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h,
                         "kept_for_later_h": keep_h, "ok": self.ok})
        b["spent_gpu_h"] = s
        jdump("budget.json", b)
        print(f"[ac1 budget] {label}: spent {s:.3f} GPU-h, estimate {est_h:.2f} h, kept {keep_h:.2f} h -> {'OK' if self.ok else 'STOP (ceiling)'}", flush=True)
        if not self.ok:
            raise Ceiling(label)

    def over(self):
        return spent() + (time.time() - self.t0) / 3600 > CAP_GPU_H


# ------------------------------------------------------------------------------------------------ student / teacher
class QuantNL(J.JointQuant):
    """JointQuant 'static' without the per-call clip-count synchronization (values unchanged; checked bitwise against JointQuant)."""

    def _fn(self, key, x, mod):
        t = self.step_t()
        self.calls[key] += 1
        D, Z, qm = self.p[key]
        with torch.no_grad():
            val = ACT.q_static(x.float().detach(), D[t], Z[t], torch.zeros((), device=x.device), qm)
        return val.to(x.dtype)


class Student:
    """common smoothed W4 base + static A8 (feedback off) + a carry dict installed through the deployed install() path."""

    def __init__(self, dev, quant_cls=QuantNL):
        self.dev = dev
        self.m = J.common_base_nocarry(dev)
        self.jq = quant_cls(self.m, dev, "static")
        self.hs = []

    def set_carry(self, carry):
        for h in self.hs:
            h.remove()
        self.hs = install(self.m, carry)

    def sample(self, seeds, bs=32):
        return J.sample_config(self.m, self.jq, seeds, self.dev, bs)

    @torch.no_grad()
    def eps(self, x, lab, i, sch):
        self.jq.t = i
        e = guided_eps(self.m, x, lab, i, sch)
        self.jq.t = None
        return e


def load_c0():
    return torch.load(J.BASE_CARRY, map_location="cpu")


def carry_e1(a, C0, C1):
    """factor interpolation per depth group: V = V0 + a_g (V1 - V0), D = D0 + a_g (D1 - D0) in fp32, exported fp16 (rank 8)."""
    out = {}
    for k in KEYS:
        ag = float(a[group_of(k)])
        out[k] = {n: (C0[k][n].float() + ag * (C1[k][n].float() - C0[k][n].float())).half() for n in ("V", "delta")}
    return out


def carry_e2(a, C0, dD):
    """V fixed; D = D0 + a_g dD in fp32, exported fp16."""
    return {k: {"V": C0[k]["V"].clone(), "delta": (C0[k]["delta"].float() + float(a[group_of(k)]) * dD[k].float()).half()} for k in KEYS}


def make_carry(exp, a):
    C0 = load_c0()
    if exp == "e1":
        return carry_e1(a, C0, torch.load(SRC_E1, map_location="cpu"))
    return carry_e2(a, C0, torch.load(os.path.join(CK, "e2_dD.pt"), map_location="cpu"))


def fp_end(seeds, dev, tag):
    f = os.path.join(CK, f"fp_{tag}.pt")
    if os.path.exists(f):
        return torch.load(f)
    fp = J.fp_model(dev)
    d = make_diffusion(T)
    with torch.no_grad():
        x = torch.cat([ddim_sample(fp, d, noise_for_seeds(seeds[b:b + 32], dev), labels_for_seeds(seeds[b:b + 32]), CFG).cpu() for b in range(0, len(seeds), 32)])
    del fp
    torch.cuda.empty_cache()
    atomic_torch_save(x, f)
    return x


def rel_mse(x, xf):
    return ((x - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1))


def baseline_states(S, seeds, dev, tag):
    """the ORIGINAL-carry student's own trajectory states at STEPS (and its endpoint), cached."""
    f = os.path.join(CK, f"states_{tag}.pt")
    if os.path.exists(f):
        return torch.load(f)
    S.set_carry(load_c0())
    d = make_diffusion(T)
    S.jq.t = None
    x, st = ddim_sample(S.m, d, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), CFG, record_states=True)
    out = {"states": {i: st[i].cpu() for i in STEPS}, "end": x.cpu()}
    atomic_torch_save(out, f)
    return out


# ------------------------------------------------------------------------------------------------ audit (CPU only): coherence field, DNS budget
def stage_audit(dev):
    res = {}
    ev = json.load(open(os.path.join(RESULTS, "dit_joint", "eval.json")))
    co = {}
    for name, v in ev["diag"].items():
        C, I = np.asarray(v["C"]), np.asarray(v["I"])
        co[name] = {"C_over_I_old (sum C / sum I; C = T*||mean_t e_t||^2)": v["C_over_I"],
                    "C_over_I_recomputed_from_stored_lists": float(C.sum() / I.sum()),
                    "coherent_fraction_ratio_of_sums (= C_over_I / T)": float(C.sum() / I.sum() / T),
                    "coherent_fraction_mean_of_imagewise_ratios": float(np.mean(C / I / T)),
                    "n_images": len(C)}
    res["coherence"] = {"definition": "c = ||mean_t e_t||^2 / mean_t ||e_t||^2 = C_old / (T*I), T = 50; old field kept unchanged",
                        "raw_error_tensors_available": False,
                        "note": "diag_ci stored only per-image C and I, not e_t; consistency checked against the stored lists only", "configs": co}
    rows = [r for r in csv.reader(open("/home/boa-dit-dnsr/results/gpu_hours.csv")) if r and r[0] == "r1"]
    res["dns_response_budget"] = {"source": "dns-response-pilot results/gpu_hours.csv (r1 rows; unique attempts incl. FAILED)",
                                  "rows": [[r[1], r[3], r[4], float(r[5])] for r in rows], "actual_total_gpu_h": float(sum(float(r[5]) for r in rows)),
                                  "budget_json_spent_gpu_h": json.load(open("/home/boa-dit-dnsr/results/dns_response/budget.json"))["spent_gpu_h"],
                                  "explanation": "budget.json logs the spend BEFORE each stage, so the last stage (fid_plain) is missing from its total"}
    res["feedback_refit_overlap"] = {"status": "skipped",
                                     "reason": "aligned prediction-error tensors for A, B, C_L, D_L were not cached (eval_latents.pt holds final latents only); "
                                               "no new GPU collection for this optional diagnostic"}
    jdump("audit.json", res)
    print(json.dumps({k: (v if k != "coherence" else {n: round(c["coherent_fraction_ratio_of_sums (= C_over_I / T)"], 4) for n, c in v["configs"].items()})
                      for k, v in res.items() if k != "dns_response_budget"}, indent=1))
    print("DNS r1 actual total GPU-h:", res["dns_response_budget"]["actual_total_gpu_h"])


# ------------------------------------------------------------------------------------------------ ridge solve (E2) and its unit checks
def ridge(Szz, Sez, lam_rel=1e-4):
    """dD = Sez (Szz + lam I)^-1 via a linear solve (float64); lam = lam_rel * trace(Szz)/8. Zero feature energy -> dD = 0."""
    r = Szz.shape[0]
    tr = float(torch.trace(Szz))
    if tr <= 0:
        return torch.zeros_like(Sez), 0.0
    lam = lam_rel * tr / r
    A = Szz + lam * torch.eye(r, dtype=Szz.dtype)
    return torch.linalg.solve(A, Sez.T).T, lam                               # A symmetric: (A^-1 Sez^T)^T = Sez A^-1


def surrogate(see, Sez, Szz, dD):
    """mean ||E0 - dD Z||^2 = see - 2 tr(dD^T Sez) + tr(dD Szz dD^T)."""
    return float(see - 2 * (dD * Sez).sum() + ((dD @ Szz) * dD).sum())


def ridge_unit_checks():
    g = torch.Generator().manual_seed(0)
    out = {}
    N, r, do = 500, 8, 40
    Z = torch.randn(r, N, generator=g, dtype=torch.float64)
    E0 = torch.randn(do, N, generator=g, dtype=torch.float64)
    Szz, Sez = Z @ Z.T / N, E0 @ Z.T / N
    dD, lam = ridge(Szz, Sez)
    out["stationarity_rel"] = float((dD @ (Szz + lam * torch.eye(r, dtype=torch.float64)) - Sez).norm() / Sez.norm())
    Aug = torch.cat([Z.T / N ** 0.5, lam ** 0.5 * torch.eye(r, dtype=torch.float64)])
    Bug = torch.cat([E0.T / N ** 0.5, torch.zeros(r, do, dtype=torch.float64)])
    ref = torch.linalg.lstsq(Aug, Bug).solution.T
    out["vs_direct_lstsq_rel"] = float((dD - ref).norm() / ref.norm())
    Dt = torch.randn(do, r, generator=g, dtype=torch.float64)
    dD0, _ = ridge(Szz, Dt @ Szz, lam_rel=0.0)
    out["lam0_exact_fit_rel"] = float((dD0 - Dt).norm() / Dt.norm())
    see = float((E0 ** 2).sum() / N)
    out["surrogate_decreases"] = surrogate(see, Sez, Szz, dD) <= see
    C0 = load_c0()
    dDz = {k: torch.zeros_like(C0[k]["delta"], dtype=torch.float32) for k in KEYS}
    c = carry_e2([0.3, 0.1, -0.2, 1.0], C0, dDz)
    out["V_unchanged_and_rank8"] = all(torch.equal(c[k]["V"], C0[k]["V"]) and c[k]["delta"].shape == C0[k]["delta"].shape and c[k]["V"].shape[1] == 8 for k in KEYS)
    out["pass"] = out["stationarity_rel"] < 1e-10 and out["vs_direct_lstsq_rel"] < 1e-8 and out["lam0_exact_fit_rel"] < 1e-8 and out["surrogate_decreases"] \
        and out["V_unchanged_and_rank8"]
    return out


# ------------------------------------------------------------------------------------------------ checks
@torch.no_grad()
def stage_checks(dev):
    res = {}
    with Run(STAGE, "ac1_checks", dev, {}) as run:
        Budget("checks", 0.12)
        base = json.load(open(os.path.join(RESULTS, "dit_joint", "base.json")))
        cal = json.load(open(os.path.join(RESULTS, "dit_joint", "calib_act.json")))
        res["hash_base_model"] = sha_file(J.BASE_STATE) == base["model_sha256"]
        res["hash_base_carry"] = sha_file(J.BASE_CARRY) == base["carry_sha256"]
        res["hash_tables"] = sha_file(J.TABLES) == cal["tables_sha256"]
        res["hash_smoothing"] = sha_file(J.SMOOTH) == base["smoothing_sha256"]
        res["source_C_L_u200_sha256"] = sha_file(SRC_E1)
        C0, C1 = load_c0(), torch.load(SRC_E1, map_location="cpu")
        res["carry_shapes_rank8"] = all(C0[k]["V"].shape[1] == 8 and C0[k]["delta"].shape[1] == 8 and C1[k]["V"].shape == C0[k]["V"].shape
                                        and C1[k]["delta"].shape == C0[k]["delta"].shape for k in KEYS) and len(C0) == 112 and sorted(C0) == sorted(KEYS)
        res["ridge_unit_checks"] = ridge_unit_checks()
        seeds4 = SETS["e1"]["fit"][:4]
        # QuantNL == JointQuant static (bitwise), reference sample = the deployed common base with its original carry file
        S_ref = Student(dev, J.JointQuant)
        S_ref.set_carry(C0)
        ref = S_ref.sample(seeds4, bs=4)
        S_ref.jq.remove()
        for h in S_ref.hs:
            h.remove()
        del S_ref
        S = Student(dev)
        S.set_carry(C0)
        x1 = S.sample(seeds4, bs=4)
        res["quantNL_equals_JointQuant_static_bitwise"] = bool(torch.equal(x1, ref))
        x2 = S.sample(seeds4, bs=4)
        res["repeat_baseline_rel_diff_4img"] = float((x1 - x2).norm() / x1.norm())
        x32 = S.sample(SETS["e1"]["fit"], bs=32)[:4]
        res["batch4_vs_batch32_rel_diff (context)"] = float((x1 - x32).norm() / x1.norm())
        # zero coefficients through the deployed export path (save -> load -> install)
        for exp, mk in (("e1", lambda a: carry_e1(a, C0, C1)), ("e2", lambda a: carry_e2(a, C0, {k: torch.randn(C0[k]["delta"].shape) for k in KEYS}))):
            f = os.path.join(CK, f"_check_{exp}_zero.pt")
            atomic_torch_save(mk([0, 0, 0, 0]), f)
            cz = torch.load(f, map_location="cpu")
            res[f"{exp}_zero_tensors_bitwise"] = all(torch.equal(cz[k][n], C0[k][n]) for k in KEYS for n in ("V", "delta"))
            S.set_carry(cz)
            res[f"{exp}_zero_sample_bitwise"] = bool(torch.equal(S.sample(seeds4, bs=4), x1))
            os.remove(f)
        c1 = carry_e1([1, 1, 1, 1], C0, C1)
        res["e1_ones_tensors_bitwise_C_L_u200"] = all(torch.equal(c1[k][n], C1[k][n]) for k in KEYS for n in ("V", "delta"))
        S.set_carry(C1)
        xs = S.sample(seeds4, bs=4)
        S.set_carry(c1)
        res["e1_ones_sample_bitwise_C_L_u200"] = bool(torch.equal(S.sample(seeds4, bs=4), xs))
        # exactly 50 denoiser calls; one carry hook per linear (no doubled rank / extra branch)
        S.set_carry(C0)
        S.jq.calls = {k: 0 for k in KEYS}
        S.sample(seeds4[:1], bs=1)
        res["calls_per_site_per_sample"] = sorted(set(S.jq.calls.values()))
        hooks = [len(block_linear(S.m.blocks[int(k.split('.')[0][1:])], k.split('.', 1)[1])._forward_hooks) for k in KEYS]
        pre = [len(block_linear(S.m.blocks[int(k.split('.')[0][1:])], k.split('.', 1)[1])._forward_pre_hooks) for k in KEYS]
        res["forward_hooks_per_linear"] = sorted(set(hooks))
        res["pre_hooks_per_linear (smoothing at fc1/fc2 + A8)"] = {"fc": sorted(set(p for p, k in zip(pre, KEYS) if "mlp" in k)),
                                                                     "attn": sorted(set(p for p, k in zip(pre, KEYS) if "attn" in k))}
        # CFG ordering / sampler indexing: guided_eps on recorded state i == the sampler's own CFG-combined eps at step i
        d = make_diffusion(T)
        sch = Sched(d, dev)
        cap = {}
        S.jq.t = None
        _, st = ddim_sample(S.m, d, noise_for_seeds(seeds4, dev), labels_for_seeds(seeds4), CFG, record_states=True,
                            eps_hook=lambda i, xx, o: (cap.__setitem__(i, o[:, :4].clone()), o)[1])
        lab = torch.as_tensor(labels_for_seeds(seeds4), device=dev)
        res["guided_eps_vs_sampler_eps_rel"] = {i: float((S.eps(st[i], lab, i, sch) - cap[i]).norm() / cap[i].norm()) for i in (0, 25, 49)}
        # E2 site pairing: shapes, token subsample, the FP teacher has no quantization hooks
        fp = J.fp_model(dev)
        res["fp_teacher_pre_hooks"] = sorted(set(len(block_linear(fp.blocks[int(k.split('.')[0][1:])], k.split('.', 1)[1])._forward_pre_hooks) for k in KEYS))
        res["bias_equal_student_fp"] = all(torch.equal(block_linear(S.m.blocks[int(k.split('.')[0][1:])], k.split('.', 1)[1]).bias,
                                                       block_linear(fp.blocks[int(k.split('.')[0][1:])], k.split('.', 1)[1]).bias) for k in KEYS)
        acc = Acc(C0)
        acc.add_step(S, fp, st[25][:2], lab[:2], 25, sch, img0=0)
        res["pairing_sites"] = len(acc.n)
        res["pairing_rows_per_site"] = sorted(set(int(v) for v in acc.n.values()))
        res["pairing_finite"] = all(bool(torch.isfinite(acc.Syz[k]).all()) and acc.syy[k] > 0 for k in KEYS)
        res["tolerances"] = {"bitwise": "QuantNL, zero coefficients, E1 ones, export path", "guided_eps_vs_sampler": "<= 1e-4 relative",
                             "repeat": "reported", "ridge": "stationarity < 1e-10, vs lstsq < 1e-8, lam=0 exact < 1e-8"}
        res["pass"] = all([res["hash_base_model"], res["hash_base_carry"], res["hash_tables"], res["hash_smoothing"], res["carry_shapes_rank8"],
                           res["ridge_unit_checks"]["pass"], res["quantNL_equals_JointQuant_static_bitwise"], res["e1_zero_tensors_bitwise"],
                           res["e1_zero_sample_bitwise"], res["e2_zero_tensors_bitwise"], res["e2_zero_sample_bitwise"],
                           res["e1_ones_tensors_bitwise_C_L_u200"], res["e1_ones_sample_bitwise_C_L_u200"], res["calls_per_site_per_sample"] == [T],
                           res["forward_hooks_per_linear"] == [1], max(res["guided_eps_vs_sampler_eps_rel"].values()) <= 1e-4,
                           res["fp_teacher_pre_hooks"] == [0], res["pairing_sites"] == 112, res["pairing_rows_per_site"] == [2 * 2 * NTOK], res["pairing_finite"]])
        res.update(run.record())
    jdump("checks.json", res)
    print(json.dumps(res, indent=1, default=str), flush=True)
    if not res["pass"]:
        raise SystemExit(6)


# ------------------------------------------------------------------------------------------------ profile
@torch.no_grad()
def stage_profile(dev):
    res = {"gpu": torch.cuda.get_device_name(0)}
    with Run(STAGE, "ac1_profile", dev, {}) as run:
        Budget("profile", 0.06)
        S = Student(dev)
        S.set_carry(load_c0())
        seeds = SETS["e1"]["fit"]
        S.sample(seeds[:4], bs=4)                                            # warm-up
        for bs in (32, 8):
            torch.cuda.synchronize()
            t0 = time.time()
            S.sample(seeds[:bs], bs=bs)
            torch.cuda.synchronize()
            res[f"sec_per_sampling_bs{bs}"] = time.time() - t0
        d = make_diffusion(T)
        sch = Sched(d, dev)
        x = noise_for_seeds(seeds, dev)
        lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            S.eps(x, lab, 10, sch)
        torch.cuda.synchronize()
        res["sec_per_forward_bs32"] = (time.time() - t0) / 3
        fp = J.fp_model(dev)
        acc = Acc(load_c0())
        torch.cuda.synchronize()
        t0 = time.time()
        acc.add_step(S, fp, x[:8], lab[:8], 10, sch, img0=0)
        torch.cuda.synchronize()
        res["sec_per_stats_step_bs8"] = time.time() - t0
        res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        s32, s8, f32 = res["sec_per_sampling_bs32"], res["sec_per_sampling_bs8"], res["sec_per_forward_bs32"]
        resp = (2 * s32 + 21 * s8 + 8 * s32 + len(STEPS) * 10 * f32) / 3600   # fp+base, h-check (+retry margin), full R, local responses
        sel = (1 + 31 + 3) * s32 / 3600                                        # fp dev, <= 31 distinct candidates, 3 fit endpoints
        conf = (4 + 20) * s32 / 3600 + 0.03                                    # fp + 5 configs x 128, decode/LPIPS
        stats = (2 * (s32 + len(STEPS) * 4 * res["sec_per_stats_step_bs8"])) / 3600
        res["projected_h"] = {"e1_resp": resp, "e1_select": sel, "e1_confirm": conf, "e2_stats": stats, "e2_solve": 0.01, "e2_resp": resp,
                              "e2_select": sel, "e2_confirm": conf}
        res["projected_total_h"] = sum(res["projected_h"].values())
        res["spent_before_h"] = spent()
        res["fits_ceiling"] = res["spent_before_h"] + res["projected_total_h"] <= CAP_GPU_H - ALLOC["reserve"]
        res.update(run.record())
    jdump("profile.json", res)
    print(json.dumps(res, indent=1), flush=True)


def est(stage):
    p = jload("profile.json")
    return float(p["projected_h"][stage]) if p else 0.5


def keep_for_e2(exp):
    """E1 stages must leave the projected E2 core untouched."""
    if exp != "e1":
        return 0.0
    return sum(est(s) for s in ("e2_stats", "e2_solve", "e2_resp", "e2_select", "e2_confirm"))


# ------------------------------------------------------------------------------------------------ E2 statistics
def tok_index(n_img, img0, step, dev):
    """deterministic shared token subsample: tokens 4k + ((image index + step) mod 4), k = 0..63, same for both CFG rows of an image."""
    off = (torch.arange(n_img, device=dev) + img0 + step) % 4
    return torch.arange(NTOK, device=dev)[None, :] * 4 + off[:, None]          # [n_img, 64]


class Acc:
    """streams per-site means: Szz = <z z^T>, Syz = <(Y - Yq) z^T>, syy = <||Y - Yq||^2>, z = V0^T x (student's consumed input),
    Yq = student linear output without the carry (W4 x + b), Y = FP teacher linear output on its own activation; float64."""

    def __init__(self, C0):
        self.V = {k: C0[k]["V"].double() for k in KEYS}
        self.Szz, self.Syz, self.syy, self.n = {}, {}, {}, {}

    def add_step(self, S, fp, x, lab, i, sch, img0):
        dev = x.device
        n = len(x)
        idx = tok_index(n, img0, i, dev).repeat(2, 1)                        # rows: cond images then uncond images
        cap, hs = {}, []

        def gather(t):
            return torch.gather(t, 1, idx[:, :, None].expand(-1, -1, t.shape[-1])).reshape(-1, t.shape[-1]).double()

        for k in KEYS:
            l, ln = k.split(".", 1)

            def sh(mod, inp, out, k=k):                                       # prepended: sees W4 x + b before the carry hook
                cap[k] = (gather(inp[0]) @ self.V[k].to(dev), gather(out))

            def th(mod, inp, out, k=k):
                z, yq = cap.pop(k)
                dlt = gather(out) - yq
                self.Szz[k] = self.Szz.get(k, 0) + (z.T @ z).cpu()
                self.Syz[k] = self.Syz.get(k, 0) + (dlt.T @ z).cpu()
                self.syy[k] = self.syy.get(k, 0) + float(dlt.pow(2).sum())
                self.n[k] = self.n.get(k, 0) + z.shape[0]
            hs.append(block_linear(S.m.blocks[int(l[1:])], ln).register_forward_hook(sh, prepend=True))
            hs.append(block_linear(fp.blocks[int(l[1:])], ln).register_forward_hook(th))
        try:
            S.eps(x, lab, i, sch)
            guided_eps(fp, x, lab, i, sch)
        finally:
            for h in hs:
                h.remove()

    def means(self):
        return {k: {"Szz": self.Szz[k] / self.n[k], "Syz": self.Syz[k] / self.n[k], "syy": self.syy[k] / self.n[k], "n": self.n[k]} for k in KEYS}


@torch.no_grad()
def stage_e2_stats(dev):
    with Run(STAGE, "ac1_e2_stats", dev, {}) as run:
        Budget("e2_stats", est("e2_stats"))
        S = Student(dev)
        fp = J.fp_model(dev)
        sch = Sched(make_diffusion(T), dev)
        rec = {}
        for split in ("fit", "dev"):
            seeds = SETS["e2"][split]
            st = baseline_states(S, seeds, dev, f"e2_{split}")
            S.set_carry(load_c0())
            acc = Acc(load_c0())
            lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
            t0 = time.time()
            for i in STEPS:
                for b in range(0, len(seeds), 8):
                    acc.add_step(S, fp, st["states"][i][b:b + 8].to(dev), lab[b:b + 8], i, sch, img0=b)
                print(f"  e2 stats {split}: step {i} done ({time.time() - t0:.0f} s)", flush=True)
            f = os.path.join(CK, f"e2_stats_{split}.pt")
            atomic_torch_save(acc.means(), f)
            rec[split] = {"file": f, "sha256": sha_file(f), "seeds": [seeds[0], seeds[-1]], "steps": STEPS, "tokens_per_row": NTOK,
                          "rows_per_site": int(acc.n[KEYS[0]])}
        rec["token_rule"] = "token 4k + ((image index within split + step) mod 4), k = 0..63, shared across sites and both CFG rows"
        rec.update(run.record())
    jdump("e2_stats.json", rec)
    print(json.dumps(rec, indent=1), flush=True)


def stage_e2_solve(dev):
    C0 = load_c0()
    st = {s: torch.load(os.path.join(CK, f"e2_stats_{s}.pt")) for s in ("fit", "dev")}
    dD, diag = {}, {}
    for k in KEYS:
        D0 = C0[k]["delta"].double()
        f = st["fit"][k]
        Sez = f["Syz"] - D0 @ f["Szz"]
        d, lam = ridge(f["Szz"], Sez)
        dD[k] = d.float()
        d_exp = (C0[k]["delta"].float() + d.float()).half().double() - D0
        row = {"lambda": lam, "trace_Szz": float(torch.trace(f["Szz"])), "cond_Szz": float(torch.linalg.cond(f["Szz"])) if lam > 0 else None,
               "cond_reg": float(torch.linalg.cond(f["Szz"] + lam * torch.eye(8, dtype=torch.float64))) if lam > 0 else None,
               "norm_D0": float(D0.norm()), "norm_dD": float(d.norm())}
        for s in ("fit", "dev"):
            g = st[s][k]
            see = float(g["syy"] - 2 * (D0 * g["Syz"]).sum() + ((D0 @ g["Szz"]) * D0).sum())
            sez = g["Syz"] - D0 @ g["Szz"]
            row[s] = {"E0_energy": see, "after": surrogate(see, sez, g["Szz"], d), "after_exported_fp16": surrogate(see, sez, g["Szz"], d_exp),
                      "update_output_energy": float(((d @ g["Szz"]) * d).sum()), "target_energy_Y_minus_Yq": float(g["syy"])}
        row["fit_regularized_not_increased"] = row["fit"]["after"] + lam * float(d.pow(2).sum()) <= row["fit"]["E0_energy"] + 1e-12 * abs(row["fit"]["E0_energy"])
        diag[k] = row
    f = os.path.join(CK, "e2_dD.pt")
    atomic_torch_save(dD, f)
    grp = {}
    for g in range(4):
        ks = [k for k in KEYS if group_of(k) == g]
        grp[g] = {s: {q: float(sum(diag[k][s][q] for k in ks)) for q in ("E0_energy", "after", "after_exported_fp16", "update_output_energy")} for s in ("fit", "dev")}
        for s in ("fit", "dev"):
            grp[g][s]["rel_improvement"] = 1 - grp[g][s]["after"] / grp[g][s]["E0_energy"]
            grp[g][s]["rel_improvement_exported"] = 1 - grp[g][s]["after_exported_fp16"] / grp[g][s]["E0_energy"]
    res = {"dD_file": f, "dD_sha256": sha_file(f), "lambda_rule": "1e-4 * trace(mean Z Z^T) / 8; no damping change", "groups": grp,
           "all_sites_regularized_not_increased": all(diag[k]["fit_regularized_not_increased"] for k in KEYS),
           "zero_energy_sites": [k for k in KEYS if diag[k]["lambda"] == 0], "max_cond_reg": max(diag[k]["cond_reg"] or 0 for k in KEYS), "sites": diag}
    jdump("e2_solve.json", res)
    print(json.dumps({"groups": grp, "all_not_increased": res["all_sites_regularized_not_increased"], "max_cond_reg": res["max_cond_reg"]}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ responses (shared §5)
def unit(g, h):
    a = [0.0] * 4
    a[g] = h
    return a


def fit_box(G, r, lam):
    """min a^T (G + lam I) a + 2 r^T a, a in [LO, HI]^4 (exact bounded least squares on the Cholesky form)."""
    from scipy.optimize import lsq_linear
    if np.trace(G) <= 0:
        return np.zeros(4)
    A = np.linalg.cholesky(G + lam * np.eye(4)).T
    b = -np.linalg.solve(A.T, r)
    return lsq_linear(A, b, bounds=(LO, HI), method="bvls", tol=1e-12).x


def gram(E, R):
    """E [M], R [M, 4] stacked and already weighted for the mean convention: G = R^T R, r = R^T E, c = E^T E."""
    return (R.T @ R), (R.T @ E), float(E @ E)


def capacity(E, R):
    U, s, _ = np.linalg.svd(R, full_matrices=False)
    tol = max(R.shape) * np.finfo(np.float64).eps * (s[0] if len(s) else 0)
    Ur = U[:, s > tol]
    p = Ur @ (Ur.T @ E)
    return {"singular_values": s.tolist(), "svd_tol": float(tol), "rank": int((s > tol).sum()), "fraction_of_fit_sq_error_explained": float(p @ p / (E @ E))}


def endpoint_vec(x, xf):
    return ((x - xf).flatten(1) / xf.flatten(1).norm(dim=1, keepdim=True)).double()


@torch.no_grad()
def stage_resp(dev, exp):
    with Run(STAGE, f"ac1_{exp}_resp", dev, {}) as run:
        Budget(f"{exp}_resp", est(f"{exp}_resp"), keep_for_e2(exp))
        fit = SETS[exp]["fit"]
        S = Student(dev)
        xf = fp_end(fit, dev, f"{exp}_fit")
        st = baseline_states(S, fit, dev, f"{exp}_fit")
        S.set_carry(load_c0())
        x0 = S.sample(fit)
        res = {"bitwise_base_equals_cached_endpoint": bool(torch.equal(x0, st["end"])), "fit_base_latent": float(rel_mse(x0, xf).mean())}
        f8 = fit[:8]
        xf8 = xf[:8]
        # repeat-run noise (same batch configuration) and the h checks on the first eight fit images (batch 8)
        r1, r2 = S.sample(fit[:4], bs=4), S.sample(fit[:4], bs=4)
        noise = float((endpoint_vec(r1, xf[:4]) - endpoint_vec(r2, xf[:4])).norm())
        b8 = S.sample(f8, bs=8)
        res["repeat_noise_norm_4img"] = noise
        res["batch8_vs_batch32_rel (context)"] = float((b8 - x0[:8]).norm() / x0[:8].norm())

        def cols(h, seeds, xfs, bs):
            out = []
            for g in range(4):
                p = []
                for sgn in (1, -1):
                    S.set_carry(make_carry(exp, unit(g, sgn * h)))
                    p.append(S.sample(seeds, bs=bs))
                out.append(((endpoint_vec(p[0], xfs) - endpoint_vec(p[1], xfs)) / (2 * h), float((endpoint_vec(p[0], xfs) - endpoint_vec(p[1], xfs)).norm())))
            return out

        def mixed(h):
            S.set_carry(make_carry(exp, [m * h for m in MIXED]))
            return endpoint_vec(S.sample(f8, bs=8), xf8) - endpoint_vec(b8, xf8)

        checks, h_ok = [], None
        prev = cols(H0, f8, xf8, 8)
        for h_new, ref_h in ((H_CHECK, H0), (H_RETRY, H0)):
            new = cols(h_new, f8, xf8, 8)
            rel = [float((new[g][0] - prev[g][0]).norm() / prev[g][0].norm().clamp_min(1e-30)) for g in range(4)]
            above = [min(new[g][1], prev[g][1]) > NOISE_MULT * noise for g in range(4)]
            h_fit = ref_h if h_new == H_CHECK else H_RETRY
            dm = mixed(h_fit)
            Rh = prev if h_fit == H0 else new
            pred = h_fit * sum(MIXED[g] * Rh[g][0] for g in range(4))
            mix_rel = float((dm - pred).norm() / dm.norm().clamp_min(1e-30))
            usable = [rel[g] <= COL_TOL and above[g] for g in range(4)]
            ok = any(usable) and mix_rel <= MIX_TOL
            checks.append({"h_fit": h_fit, "compared_h": [ref_h, h_new], "col_rel_diff": rel, "above_noise": above, "usable": usable,
                           "col_norms_h_fit": [float(Rh[g][0].norm()) for g in range(4)], "mixed_rel_err": mix_rel, "ok": ok})
            print(f"  {exp} h-check {checks[-1]}", flush=True)
            if ok:
                h_ok, usable_ok = h_fit, usable
                break
        res["h_checks"] = checks
        res["h"] = h_ok
        h_loc = h_ok or H0
        # full endpoint response on the 32 fit images (batch 32)
        n = len(fit)
        e = endpoint_vec(x0, xf)
        E = (e.flatten() / n ** 0.5).numpy()
        if h_ok is not None:
            Rcols = []
            for g in range(4):
                p = []
                for sgn in (1, -1):
                    S.set_carry(make_carry(exp, unit(g, sgn * h_ok)))
                    p.append(endpoint_vec(S.sample(fit), xf))
                c = ((p[0] - p[1]) / (2 * h_ok)).flatten() / n ** 0.5
                Rcols.append(c.numpy() if usable_ok[g] else np.zeros_like(E))
                print(f"  {exp} endpoint column {g}: norm {np.linalg.norm(Rcols[-1]):.4f}", flush=True)
            Re = np.stack(Rcols, 1)
            res["endpoint"] = {"capacity": capacity(E, Re), "usable": usable_ok}
            np.savez(os.path.join(CK, f"{exp}_endpoint_resp.npz"), E=E, R=Re)
        else:
            res["endpoint"] = {"status": "approximation check FAILED at h = 0.1 and 0.2: R fitting stopped for this experiment"}
        # local response at the frozen baseline states (fit), the same four perturbations
        sch = Sched(make_diffusion(T), dev)
        fp = J.fp_model(dev)
        lab = torch.as_tensor(labels_for_seeds(fit), device=dev)
        el, Rl, nrm = [], [], []
        for i in STEPS:
            x = st["states"][i].to(dev)
            et = guided_eps(fp, x, lab, i, sch)
            S.set_carry(load_c0())
            e_i = (S.eps(x, lab, i, sch) - et).flatten(1).double()
            nrm.append(float(e_i.pow(2).sum(1).mean()))
            cols_i = []
            for g in range(4):
                p = []
                for sgn in (1, -1):
                    S.set_carry(make_carry(exp, unit(g, sgn * h_loc)))
                    p.append(S.eps(x, lab, i, sch).flatten(1).double())
                cols_i.append(((p[0] - p[1]) / (2 * h_loc)).flatten())
            el.append(e_i.flatten())
            Rl.append(torch.stack(cols_i, 1))
        floor = 1e-6 * float(np.mean(nrm))
        w = [1.0 / (len(STEPS) * n * max(v, floor)) ** 0.5 for v in nrm]
        El = torch.cat([el[j] * w[j] for j in range(len(STEPS))]).cpu().numpy()
        Rl_ = torch.cat([Rl[j] * w[j] for j in range(len(STEPS))]).cpu().numpy()
        res["local"] = {"h": h_loc, "step_norm": nrm, "floor": floor, "capacity": capacity(El, Rl_)}
        np.savez(os.path.join(CK, f"{exp}_local_resp.npz"), E=El, R=Rl_)
        # candidates (fixed families, deduplicated by exported carry hash)
        cand = {"B": [0, 0, 0, 0], "A": [1, 1, 1, 1]}
        for al in ALPHAS:
            cand[f"S_{al}"] = [al] * 4
        fams = {"L": (El, Rl_)}
        if h_ok is not None:
            fams["R"] = (E, Re)
        res["fits"] = {}
        for fam, (Ev, Rv) in fams.items():
            G, r, c = gram(Ev, Rv)
            res["fits"][fam] = {"G": G.tolist(), "r": r.tolist(), "c": c, "trace": float(np.trace(G))}
            cand[f"{fam}_zero"] = [0, 0, 0, 0]
            for gm in GAMMAS:
                a = fit_box(G, r, gm * np.trace(G) / 4)
                pred = float(a @ G @ a + 2 * r @ a + c)
                res["fits"][fam][f"gamma_{gm}"] = {"a": a.tolist(), "predicted_objective": pred, "baseline_objective": c}
                for s in SHRINK:
                    cand[f"{fam}_g{gm}_s{s}"] = (s * a).tolist()
        res["candidates"] = cand
        res.update(run.record())
    jdump(f"{exp}_resp.json", res)
    print(json.dumps({k: v for k, v in res.items() if k not in ("candidates",)}, indent=1, default=str)[:6000], flush=True)


# ------------------------------------------------------------------------------------------------ development selection
@torch.no_grad()
def stage_select(dev, exp):
    rs = jload(f"{exp}_resp.json")
    with Run(STAGE, f"ac1_{exp}_select", dev, {}) as run:
        bud = Budget(f"{exp}_select", est(f"{exp}_select"), keep_for_e2(exp))
        dv = SETS[exp]["dev"]
        xf = fp_end(dv, dev, f"{exp}_dev")
        S = Student(dev)
        f = os.path.join(OUT, f"{exp}_select_partial.json")
        res = json.load(open(f)) if os.path.exists(f) else {"models": {}, "cand": {}}
        for name, a in rs["candidates"].items():
            hsh = sha_carry(make_carry(exp, a))
            res["cand"][name] = {"a": a, "hash": hsh}
            if hsh in res["models"]:
                continue
            if bud.over():
                atomic_write_json(f, res)
                raise Ceiling(f"{exp}_select")
            S.set_carry(make_carry(exp, a))
            v = rel_mse(S.sample(dv), xf)
            res["models"][hsh] = {"dev_latent": float(v.mean()), "per_image": v.tolist()}
            print(f"  {exp} dev {name}: latent {float(v.mean()):.5f}", flush=True)
            atomic_write_json(f, res)
        base = res["models"][res["cand"]["B"]["hash"]]["dev_latent"]
        sel = {}
        for fam in ("S", "L", "R"):
            names = [n for n in res["cand"] if n.startswith(fam + "_")]
            if not names:
                sel[fam] = None
                continue
            key = lambda n: (res["models"][res["cand"][n]["hash"]]["dev_latent"], float(np.linalg.norm(res["cand"][n]["a"])))
            best = min(names, key=key)
            sel[fam] = {"name": best, "a": res["cand"][best]["a"], "hash": res["cand"][best]["hash"],
                        "dev_latent": res["models"][res["cand"][best]["hash"]]["dev_latent"], "n_candidates": len(names),
                        "n_distinct_models": len({res["cand"][n]["hash"] for n in names}), "identity": bool(np.allclose(res["cand"][best]["a"], 0))}
        # predicted (from the FIT Gram) vs actual DEV mean change for every candidate of L / R, and S
        pred = {}
        for name, c in res["cand"].items():
            out = {"actual_dev_change": res["models"][c["hash"]]["dev_latent"] - base}
            a = np.asarray(c["a"])
            for fam in rs["fits"]:
                ff = rs["fits"][fam]
                G, r = np.asarray(ff["G"]), np.asarray(ff["r"])
                out[f"predicted_change_from_{fam}_fit_quadratic"] = float(a @ G @ a + 2 * r @ a)
            pred[name] = out
        # actual vs predicted on FIT for the selected S / L / R (endpoint response model)
        fitchk = {}
        fit = SETS[exp]["fit"]
        xff = fp_end(fit, dev, f"{exp}_fit")
        st = baseline_states(S, fit, dev, f"{exp}_fit")
        fit_base = float(rel_mse(st["end"], xff).mean())
        for fam, s in sel.items():
            if s is None or s["identity"]:
                continue
            S.set_carry(make_carry(exp, s["a"]))
            act = float(rel_mse(S.sample(fit), xff).mean())
            a = np.asarray(s["a"])
            p = {}
            for ffam in rs["fits"]:
                G, r = np.asarray(rs["fits"][ffam]["G"]), np.asarray(rs["fits"][ffam]["r"])
                p[ffam] = float(a @ G @ a + 2 * r @ a)
            fitchk[fam] = {"actual_fit_change": act - fit_base, "predicted_change": p, "fit_base": fit_base}
        out = {"dev_seeds": [dv[0], dv[-1]], "dev_base_latent": base, "selected": sel, "candidates": res["cand"], "models": res["models"],
               "predicted_vs_actual_dev": pred, "fit_actual_vs_predicted": fitchk, **run.record()}
    jdump(f"{exp}_select.json", out)
    print(json.dumps({"base": base, "selected": sel, "fit": fitchk}, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ confirmation
@torch.no_grad()
def stage_confirm(dev, exp):
    import lpips
    sel = jload(f"{exp}_select.json")
    with Run(STAGE, f"ac1_{exp}_confirm", dev, {}) as run:
        bud = Budget(f"{exp}_confirm", est(f"{exp}_confirm"), keep_for_e2(exp))
        conf = SETS[exp]["conf"]
        cfgs = {"B": [0, 0, 0, 0], "A": [1, 1, 1, 1]}
        for fam in ("S", "L", "R"):
            if sel["selected"].get(fam):
                cfgs[fam] = sel["selected"][fam]["a"]
        frozen = {k: sha_carry(make_carry(exp, a)) for k, a in cfgs.items()}
        assert all(frozen[f] == sel["selected"][f]["hash"] for f in ("S", "L", "R") if f in frozen), "export hash differs from the frozen selection"
        xf = fp_end(conf, dev, f"{exp}_conf")
        vae = load_vae(dev)
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        S = Student(dev)
        f = os.path.join(OUT, f"{exp}_confirm.json")
        res = json.load(open(f)) if os.path.exists(f) else {"seeds": [conf[0], conf[-1]], "frozen_hashes": frozen, "configs": {}}
        lat_f = os.path.join(CK, f"{exp}_confirm_latents.pt")
        lats = torch.load(lat_f) if os.path.exists(lat_f) else {"fp": xf}
        by_hash = {}
        for name, a in cfgs.items():
            h = frozen[name]
            if name in res["configs"]:
                by_hash.setdefault(h, name)
                continue
            if h in by_hash:
                res["configs"][name] = {"same_as": by_hash[h], "a": a}
                continue
            if bud.over():
                atomic_write_json(f, res)
                raise Ceiling(f"{exp}_confirm")
            S.set_carry(make_carry(exp, a))
            t0 = time.time()
            x = S.sample(conf)
            sec = (time.time() - t0) / len(conf)
            res["configs"][name] = {**J.metrics(x, xf, vae, net, dev), "a": a, "hash": h, "sec_per_image": sec}
            by_hash[h] = name
            lats[name] = x
            print(f"  {exp} confirm {name}: latent {np.mean(res['configs'][name]['rel_mse']):.5f} PSNR {np.mean(res['configs'][name]['psnr']):.2f} "
                  f"LPIPS {np.mean(res['configs'][name]['lpips']):.4f}", flush=True)
            atomic_torch_save(lats, lat_f)
            atomic_write_json(f, res)
        grids(exp, lats, res, vae, dev)
        res.setdefault("runs", []).append(run.record())
    atomic_write_json(f, res)


def grids(exp, lats, res, vae, dev):
    """fixed examples (set positions 0, 42, 85, chosen before metrics) + the three worst images of the R configuration (or L if R absent)."""
    from PIL import Image
    from boa_dit.model import decode
    conf = SETS[exp]["conf"]
    names = ["fp"] + [k for k in ("B", "A", "S", "L", "R") if k in lats]
    ref = "R" if "R" in res["configs"] and "rel_mse" in res["configs"]["R"] else "L"
    worst = list(np.argsort(res["configs"][ref]["rel_mse"])[::-1][:3]) if ref in res["configs"] and "rel_mse" in res["configs"][ref] else []
    cols = [0, 42, 85] + [int(i) for i in worst]
    rows = []
    for n in names:
        im = decode(vae, lats[n][cols].to(dev)).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
        rows.append(np.concatenate(list(im), 1))
    Image.fromarray(np.concatenate(rows, 0)).save(os.path.join(OUT, f"{exp}_grid.png"))
    res["grid"] = {"rows": names, "columns_seeds": [conf[c] for c in cols], "fixed": [conf[c] for c in (0, 42, 85)], "worst_by": ref}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(CK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    s = a.stage
    outputs = {"audit": "audit.json", "checks": "checks.json", "profile": "profile.json", "e2_stats": "e2_stats.json", "e2_solve": "e2_solve.json",
               **{f"{e}_{q}": f"{e}_{q}.json" for e in ("e1", "e2") for q in ("resp", "select")}}
    if s in outputs and done(outputs[s]):
        print(f"  {s}: done ({outputs[s]}), skipped", flush=True)
        return
    if s.endswith("_confirm") and done(f"{s}.json") and "runs" in jload(f"{s}.json"):
        print(f"  {s}: done, skipped", flush=True)
        return
    if s == "report":
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "make_analytic_carry.py")],
                       check=True)
        return
    fn = {"audit": stage_audit, "checks": stage_checks, "profile": stage_profile, "e2_stats": stage_e2_stats, "e2_solve": stage_e2_solve}
    try:
        if s in fn:
            fn[s](dev)
        else:
            exp, q = s.split("_", 1)
            {"resp": stage_resp, "select": stage_select, "confirm": stage_confirm}[q](dev, exp)
    except Ceiling as c:
        print(f"CEILING: stage {c} not started/finished; partial outputs kept", flush=True)
        raise SystemExit(5)


if __name__ == "__main__":
    main()
