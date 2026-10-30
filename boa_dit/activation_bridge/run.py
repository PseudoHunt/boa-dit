"""Overnight: activation feedback, anchor reuse and retrospective bridge correction (results/activation_bridge_overnight/PROTOCOL.md). Tag abn1.
Cheap: unrefitted common smoothed rank-8 W4 base + carry, static single8 tables; LocalQuant (local_refresh, parent JointQuant arithmetic unchanged)
in 'static' (C), 'fb' (AC) or 'fb' with the previous state forced to zero every call (G). High: FP (load_dit) or validated H8 (weight_anchor.w8.h8_model).
Sampler: anchors every K calls with reuse of d = Q(x_a) - H(x_a) on guided channels 0-3; bridge at the next anchor c (k = c - a):
    x_c <- x_c - (d_c - d_a) * sum_{j=1}^{k-1} (j/k) b_{a+j} prod_{m=j+1}^{k-1} A_{a+m}
then the DDIM update with the already-computed anchor prediction (br1 convention). One cheap call per step (feedback advances once).
Stages: checks | s1 | s2 | s3 | s4 | s5 | s6 | s7 | report
Run: python -m boa_dit.activation_bridge.run <stage>      (queue: bash scripts/abn1_queue.sh)
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
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as DJ
from boa_dit.local_refresh.run import LocalQuant
from boa_dit.model import NULL_CLASS, decode, labels_for_seeds, load_dit, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor import w8 as W8
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import guided
from boa_dit.weight_anchor.shared import boot_idx, ci, contrast, summary

STAGE = "abn1"
OUT = os.path.join(RESULTS, "activation_bridge_overnight")
CK = os.path.join(CKPT_DIR, "activation_bridge_overnight")
CAP_GPU_H = 6.0
WALL_H = 8.0
SHUTDOWN_RESERVE_H = 0.15
T = 50
BS = 32
DEV, DIAG = list(range(60208, 60224)), list(range(60208, 60216))
CONF = list(range(110000, 110256))
CONF128 = CONF[:128]
FID_SEEDS = list(range(112000, 114000))
FID_LABELS = [i // 2 for i in range(2000)]                                      # two images per ImageNet class
AA1DEV = list(range(60200, 60216))
HASHES = {"base_model": (DJ.BASE_STATE, "f017af877a79015fb9670504da4fb9fb51df30e1d8e94f6b4a3deedd5aec7ced"),
          "base_carry": (DJ.BASE_CARRY, "b4a537b0570b97d7f50f8f7d9230f1f6e37c157b7277f589285dbd1ee41aedd5"),
          "tables": (DJ.TABLES, "b73030de42a83255338f4a4ab92dce52febf604f5b5b935779c1f2342298dd67"),
          "smoothing": (DJ.SMOOTH, "7ee15c0c72fc18569f2e72671089b20c1293d368dc193d1af49276c1a296db26"),
          "h8": (W8.H8_FILE, "c574916e3f8a34754cc7d95e21db3c0de3d5498f3c46a4d2120109477aa248ad"),
          "heig8_w4_model": (W8.W4_STATE, "facee154c8a3e1a983c0f547fc8fd11a8eecae820ab63eea04fa26f580633761"),
          "carry_C_L_u200": (os.path.join(DJ.CK, "carry_C_L_u200.pt"), "ac20161034dbae00139cf472d1a10f2be77e39d201c83fa9e842ef4d06b55e69"),
          "carry_E_L_u200": (os.path.join(DJ.CK, "carry_E_L_u200.pt"), "77b5d0ac43b9ed3eb393d2a2c0f73fa5d8f9135731c3a0c555ac80300639572d"),
          "fid_ref_stats": (os.path.join(RESULTS, "stage0", "ref_stats.npz"), "a11af8b0f80b2d77"),
          "fid_ref_images": (os.path.join(CKPT_DIR, "VIRTUAL_imagenet256_labeled.npz"), "b32732719497e426")}
CARRY = {"0": DJ.BASE_CARRY, "C_L": HASHES["carry_C_L_u200"][0], "E_L": HASHES["carry_E_L_u200"][0]}


# ------------------------------------------------------------------------------------------------------------------- arms
def A(cheap, high="fp", K=4, bridge=False, carry="0"):
    return {"cheap": cheap, "high": high, "K": K, "bridge": bridge, "carry": carry}


ARMS = {
    # Stage 1: FP anchors, K4
    "C": A("static"), "AC": A("fb"), "CB": A("static", bridge=True), "ACB": A("fb", bridge=True), "G": A("G"), "GB": A("G", bridge=True),
    "FP50": {"high_only": "fp"},
    # Stage 2: W8 anchors, K4
    "C.w8": A("static", "h8"), "AC.w8": A("fb", "h8"), "CB.w8": A("static", "h8", bridge=True), "ACB.w8": A("fb", "h8", bridge=True),
    "GB.w8": A("G", "h8", bridge=True), "H8_50": {"high_only": "h8"},
    # Stage 3: K8
    "C.k8": A("static", K=8), "AC.k8": A("fb", K=8), "CB.k8": A("static", K=8, bridge=True), "ACB.k8": A("fb", K=8, bridge=True),
    "C.w8.k8": A("static", "h8", 8), "AC.w8.k8": A("fb", "h8", 8), "CB.w8.k8": A("static", "h8", 8, True), "ACB.w8.k8": A("fb", "h8", 8, True),
    # Stage 4: existing local refits + bridge
    "LCB": A("static", bridge=True, carry="C_L"), "LACB": A("fb", bridge=True, carry="E_L"),
}
STAGE_ARMS = {"s1": ["FP50", "C", "AC", "CB", "ACB", "G", "GB"],
              "s2": ["H8_50", "C.w8", "AC.w8", "CB.w8", "ACB.w8", "GB.w8"],
              "s3": ["C.k8", "AC.k8", "CB.k8", "ACB.k8", "C.w8.k8", "AC.w8.k8", "CB.w8.k8", "ACB.w8.k8"],
              "s4": ["LCB", "LACB"]}
PLAN_H = {"checks": 0.5, "s1": 0.9, "s2": 0.6, "s3": 0.7, "s4": 0.4, "s5": 0.3, "s6": 1.2, "s7": 0.2}


def expect(spec):
    if "high_only" in spec:
        return {"cheap": 0, "high": 50, "corrections": 0}
    K = spec["K"]
    n_anchor = len(range(0, T, K)) if K else 0
    corr = (n_anchor - 1) if (spec["bridge"] and K and K > 1) else 0
    return {"cheap": 50, "high": n_anchor, "corrections": corr}


# ------------------------------------------------------------------------------------------------------------------- ledger
def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


def launch_time():
    f = os.path.join(CK, "launch_utc.txt")
    if not os.path.exists(f):
        os.makedirs(CK, exist_ok=True)
        open(f, "w").write(str(time.time()))
    return float(open(f).read())


def wall_left_h():
    return WALL_H - (time.time() - launch_time()) / 3600


def budget_log(entry):
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "wall_h": WALL_H, "planning_h": PLAN_H, "log": []}
    b["log"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **entry})
    b["spent_gpu_h"] = spent()
    jdump("budget.json", b)


def fits(label, est_h, reserve_h):
    s, w = spent(), wall_left_h()
    ok = s + est_h + reserve_h <= CAP_GPU_H and est_h + reserve_h <= w
    budget_log({"stage": label, "spent_before_h": s, "wall_left_h": w, "estimate_h": est_h, "reserve_h": reserve_h, "ok": ok})
    print(f"[abn1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H}; wall left {w:.2f} h -> {'OK' if ok else 'SKIP'}", flush=True)
    return ok


def provenance():
    out = {k: {"path": p, "sha256": DJ.sha(p), "expected": h} for k, (p, h) in HASHES.items()}
    bad = {k: v for k, v in out.items() if not v["sha256"].startswith(v["expected"])}
    assert not bad, bad
    return out


def rl2(x, y):
    return float((x - y).double().norm() / y.double().norm().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------------- DDIM coefficients
def coeffs(d):
    """step i (respaced t = 49 - i): x_{i+1} = A_i x_i + b_i eps_i (DDIM, eta 0)."""
    Ac, bc = [], []
    for i in range(T):
        t = T - 1 - i
        ab, abp = float(d.alphas_cumprod[t]), float(d.alphas_cumprod_prev[t])
        Ac.append(math.sqrt(abp / ab))
        bc.append(math.sqrt(1 - abp) - math.sqrt(abp * (1 - ab) / ab))
    return Ac, bc


def weights(Ac, bc, a, k):
    """w_j = b_{a+j} prod_{m=j+1}^{k-1} A_{a+m}, j = 1..k-1."""
    out = []
    for j in range(1, k):
        p = bc[a + j]
        for m in range(j + 1, k):
            p *= Ac[a + m]
        out.append(p)
    return out


def bridge_coef(Ac, bc, a, k):
    return sum((j / k) * w for j, w in zip(range(1, k), weights(Ac, bc, a, k)))


# ------------------------------------------------------------------------------------------------------------------- models
class Cheap:
    """common base + LocalQuant ('static' or 'fb'); carry swappable by removing and re-installing its hooks."""

    def __init__(self, dev, mode):
        self.m = DJ.common_base_nocarry(dev)
        self.jq = LocalQuant(self.m, dev, mode)                                # registered after the smoothing pre-hooks
        self.jq.t, self.jq.auto = None, True
        self.cur = None
        self.hs = []
        self.use("0")

    def use(self, carry):
        if carry != self.cur:
            for h in self.hs:
                h.remove()
            self.hs = install(self.m, torch.load(CARRY[carry], map_location="cpu"))
            self.cur = carry


def load_models(dev, need=("fp", "h8", "static", "fb")):
    M = {}
    if "fp" in need:
        M["fp"] = load_dit(dev)
        M["fp"].requires_grad_(False)
    if "h8" in need:
        M["h8"] = W8.h8_model(dev)
    if "static" in need:
        M["static"] = Cheap(dev, "static")
    if "fb" in need:
        M["fb"] = Cheap(dev, "fb")
    return M


def needs(arms):
    s = set()
    for a in arms:
        sp = ARMS[a]
        if "high_only" in sp:
            s.add(sp["high_only"])
            continue
        s.add(sp["high"])
        s.add("static" if sp["cheap"] == "static" else "fb")
    return s


# ------------------------------------------------------------------------------------------------------------------- sampler
@torch.no_grad()
def sample(M, spec, d, noise, labels, lam=1.0, diag=None, stop_at=None, resume=None, rec=None):
    """One trajectory. spec: ARMS entry (K None = no anchors). Returns x, counts. diag: dict for Stage-5 diagnostics (passive)."""
    n = len(noise)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    Ac, bc = coeffs(d)
    tmap = torch.as_tensor(d.timestep_map, device=noise.device, dtype=torch.long)
    if "high_only" in spec:
        hm, cm, jq, anchors = M[spec["high_only"]], None, None, None
    else:
        hm = M[spec["high"]] if spec.get("high") else None
        ch = M["static" if spec["cheap"] == "static" else "fb"]
        ch.use(spec["carry"])
        cm, jq = ch.m, ch.jq
        if jq is not None:
            jq.zero_state = spec["cheap"] == "G"
        anchors = set(range(0, T, spec["K"])) if spec["K"] else set()
    if resume is None:
        x, i0 = noise.float().clone(), 0
        st = {"d": None, "a": None, "pre": None}
        cnt = {"cheap": 0, "high": 0, "corrections": 0}
        if jq is not None:
            jq.state_in, jq.state_out = {}, {}
    else:
        x, i0 = resume["x"].to(noise.device), resume["i"]
        st = {"d": None if resume["d"] is None else resume["d"].to(noise.device), "a": resume["a"], "pre": None}
        cnt = dict(resume["cnt"])
        torch.set_rng_state(resume["rng_cpu"]); torch.cuda.set_rng_state(resume["rng_cuda"])
        if jq is not None:
            jq.state_in, jq.state_out = {k: v.to(noise.device) for k, v in resume["fb_state"].items()}, {}
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        if i < i0:
            continue
        if stop_at is not None and i == stop_at:
            return {"x": x.cpu(), "i": i, "d": None if st["d"] is None else st["d"].cpu(), "a": st["a"], "cnt": dict(cnt),
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "fb_state": {} if jq is None else {k: v.detach().clone().cpu() for k, v in jq.state_in.items()}}
        CTX.step, CTX.n = i, n
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        if diag is not None:
            diag["x"][i] = x.detach().clone()
        if cm is None:
            def model_fn(xx, ts):
                cnt["high"] += 1
                return guided(hm, xx, ts, y, n)
        elif i in anchors:
            tsm = tmap[t]
            rc = guided(cm, x, tsm, y, n)                                    # the single cheap call of this step (feedback commits here)
            rh = guided(hm, x, tsm, y, n)
            cnt["cheap"] += 1
            cnt["high"] += 1
            if rec is not None:
                rec.append((x.clone(), tsm.clone(), y.clone()))
            dnew = rc[:, :4] - rh[:, :4]
            if diag is not None:
                diag["e"][i] = dnew.double().cpu()
                diag["anchor"].add(i)
            if spec.get("bridge") and st["d"] is not None:
                k = i - st["a"]
                coef = bridge_coef(Ac, bc, st["a"], k)
                if k > 1:
                    dx = lam * coef * (dnew - st["d"])
                    x = x - dx
                    cnt["corrections"] += 1
                    if diag is not None:
                        diag["jump"][i] = float(dx.double().norm())
                else:
                    assert coef == 0.0
            st["d"], st["a"], st["pre"] = dnew, i, rh

            def model_fn(xx, ts):
                return st["pre"]
        else:
            def model_fn(xx, ts, _i=i):
                cnt["cheap"] += 1
                rc = guided(cm, xx, ts, y, n)
                if diag is not None:                                        # diagnostic same-state FP call (stateless; never used live)
                    diag["e"][_i] = (rc[:, :4] - guided(diag["fp"], xx, ts, y, n)[:, :4]).double().cpu()
                if st["d"] is None:
                    return rc
                res = rc.clone()
                res[:, :4] -= st["d"]
                return res
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    if diag is not None:
        diag["x"][T] = x.detach().clone()
    return x, cnt


# ------------------------------------------------------------------------------------------------------------------- generation
def gen(M, d, seeds, arms, part_name, labels=None, bs=BS, est_batch_h=0.0, timing=None):
    """batch-by-batch over all arms; per-batch latents checkpointed to ckpt/.../<part_name>.pt and resumed."""
    pf = os.path.join(CK, f"{part_name}.pt")
    P = torch.load(pf, map_location="cpu") if os.path.exists(pf) else {"done": [], "lat": {a: [] for a in arms}, "cnt": {}, "timing": {}}
    t0 = time.time()
    for b0 in range(0, len(seeds), bs):
        if b0 in P["done"]:
            continue
        s = seeds[b0:b0 + bs]
        el = (time.time() - t0) / 3600
        if spent() + el + est_batch_h + SHUTDOWN_RESERVE_H > CAP_GPU_H or est_batch_h + SHUTDOWN_RESERVE_H > wall_left_h():
            print(f"[abn1 runtime stop] before batch {s[0]}-{s[-1]}", flush=True)
            return P, False
        nz = noise_for_seeds(s, "cuda:0")
        lab = labels_for_seeds(s) if labels is None else np.asarray(labels[b0:b0 + bs])
        for a in arms:
            torch.cuda.synchronize()
            ts = time.perf_counter()
            x, c = sample(M, ARMS[a], d, nz, lab)
            torch.cuda.synchronize()
            dt = time.perf_counter() - ts
            assert torch.isfinite(x).all() and c == expect(ARMS[a]), (a, c, expect(ARMS[a]))
            P["lat"][a].append(x.cpu())
            P["cnt"][a] = c
            P["timing"].setdefault(a, []).append(dt)
            print(f"  seeds {s[0]}-{s[-1]} {a}: {dt:.1f} s", flush=True)
        P["done"].append(b0)
        atomic_torch_save(P, pf)
    return P, True


def cat(P, arms):
    return {a: torch.cat(P["lat"][a]) for a in arms}


# ------------------------------------------------------------------------------------------------------------------- checks
def synthetic(d):
    """fp64: constant discrepancy -> zero bridge; linear drift -> the corrected state equals the high-only state, K4 and K8 (real coefficients)."""
    Ac, bc = coeffs(d)
    g = torch.Generator().manual_seed(0)
    out = {}
    for K in (4, 8):
        for a in range(0, T - K, K):
            for kind in ("constant", "linear"):
                x0 = torch.randn(3, 5, generator=g, dtype=torch.float64)
                e0 = torch.randn(3, 5, generator=g, dtype=torch.float64)
                s = torch.randn(3, 5, generator=g, dtype=torch.float64)
                e = lambda j: e0 + (s * j if kind == "linear" else 0.0)        # cheap error at step a + j; high prediction eps = 0
                xt, xc = x0.clone(), x0.clone()
                xt = Ac[a] * xt                                               # anchor step uses the high prediction (eps 0) in both
                xc = Ac[a] * xc
                for j in range(1, K):
                    xt = Ac[a + j] * xt
                    xc = Ac[a + j] * xc + bc[a + j] * (e(j) - e(0))            # held correction leaves e_j - e_a
                corr = bridge_coef(Ac, bc, a, K) * (e(K) - e(0))
                r = float(((xc - corr) - xt).norm() / max(float((xc - xt).norm()), 1e-300)) if kind == "linear" else float(corr.norm())
                out.setdefault(f"K{K}_{kind}", []).append(r)
    res = {k: max(v) for k, v in out.items()}
    res["pass"] = res["K4_constant"] == 0.0 and res["K8_constant"] == 0.0 and res["K4_linear"] < 1e-12 and res["K8_linear"] < 1e-12
    res["coef_K1_zero"] = all(bridge_coef(Ac, bc, a, 1) == 0.0 for a in range(T - 1))
    return res


@torch.no_grad()
def checks(dev):
    with Run(STAGE, "checks", dev, {}) as run:
        ch = {"provenance": provenance()}
        d = make_diffusion(T)
        ch["C9_synthetic"] = synthetic(d)
        M = load_models(dev)
        fpm, jqs = M["fp"], M["static"].jq
        jqf = M["fb"].jq
        # C1 reproduce cached C, AC, FP50 (aa1, 60200-60215 batch 16), Q / A plain (C3), CB1 (br1, 60208-60223 batch 16)
        aa = torch.load(os.path.join(CKPT_DIR, "activation_anchor", "checks_dev_latents.pt"), map_location="cpu")
        nz, lb = noise_for_seeds(AA1DEV, dev), labels_for_seeds(AA1DEV)
        c1, ref = {}, {}
        for name, key in (("C", "C"), ("AC", "AC"), ("FP50", "FP50")):
            ref[name], c = sample(M, ARMS[name], d, nz, lb)
            c1[name] = bool(torch.equal(ref[name].cpu(), aa[key]))
        ch["C1_reproduce_aa1"] = c1
        br = torch.load(os.path.join(CKPT_DIR, "anchor_bridge", "dev_latents.pt"), map_location="cpu")
        nzd, lbd = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        x, c = sample(M, ARMS["CB"], d, nzd, lbd)
        ch["C1_reproduce_br1_CB1"] = bool(torch.equal(x.cpu(), br["CB1"]))
        print(f"  C1 {c1} CB1 {ch['C1_reproduce_br1_CB1']}", flush=True)
        # C2 lambda 0 == C / AC (latent and final feedback state)
        c2 = {}
        for name in ("CB", "ACB"):
            base = name[:-1]
            xa, _ = sample(M, ARMS[base], d, nz, lb)
            sa = {k: v.clone() for k, v in (jqf if "A" in name else jqs).state_out.items()}
            xb, _ = sample(M, ARMS[name], d, nz, lb, lam=0.0)
            sb = (jqf if "A" in name else jqs).state_out
            c2[name] = bool(torch.equal(xa.cpu(), ref[base].cpu()) and torch.equal(xb.cpu(), xa.cpu()) and all(torch.equal(sa[k], sb[k]) for k in sa))
        ch["C2_lambda0_equals_base_incl_state"] = c2
        # C3 no anchors == plain static / feedback
        c3 = {}
        for name, key in (("C", "Q"), ("AC", "A")):
            sp = dict(ARMS[name], K=None)
            x, c = sample(M, sp, d, nz, lb)
            c3[name] = bool(torch.equal(x.cpu(), aa[key]) and c == {"cheap": 50, "high": 0, "corrections": 0})
        ch["C3_no_anchor_equals_plain"] = c3
        # C4 K1 == H50 (FP and W8, feedback off/on), zero corrections; C10 H8-50 reproduces wa8 cache
        w8dev = torch.load(os.path.join(CKPT_DIR, "weight_anchor_w8", "dev_latents.pt"), map_location="cpu")
        nzw, lbw = noise_for_seeds(W8.DEV, dev), labels_for_seeds(W8.DEV)
        x8, _ = sample(M, ARMS["H8_50"], d, nzw, lbw)
        ch["C10_H8_50_reproduces_wa8"] = bool(torch.equal(x8.cpu(), w8dev["H8"]))
        n4 = slice(0, 4)
        h50 = {"fp": ref["FP50"][n4], "h8": sample(M, ARMS["H8_50"], d, nz[n4], lb[n4])[0].cpu()}
        f50_4 = sample(M, ARMS["FP50"], d, nz[n4], lb[n4])[0].cpu()
        c4 = {}
        for high in ("fp", "h8"):
            for cheap in ("static", "fb"):
                x, c = sample(M, A(cheap, high, K=1, bridge=True), d, nz[n4], lb[n4])
                tgt = f50_4 if high == "fp" else h50["h8"]
                c4[f"{cheap}/{high}"] = bool(torch.equal(x.cpu(), tgt) and c == {"cheap": 50, "high": 50, "corrections": 0})
        ch["C4_K1_equals_H50"] = c4
        ch["C4_note"] = "FP50 reference for K1 regenerated at batch 4 (same batch as the K1 runs)"
        # C5 identical stateless cheap/high -> zero discrepancy and bridge
        Mi = {"fp": fpm, "static": type("S", (), {"m": fpm, "jq": None, "use": lambda self, c: None})()}
        sp = A("static", "fp", 4, True)
        D5 = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": fpm}
        x, c = sample(Mi, sp, d, nz[n4], lb[n4], diag=D5)
        ch["C5_identical_stateless_zero"] = bool(torch.equal(x.cpu(), f50_4) and all(float(v.abs().max()) == 0.0 for v in D5["e"].values())
                                                 and all(v == 0.0 for v in D5["jump"].values()))
        # C6 counts (K4, K8), quantizer updates per site, corrections
        c6 = {}
        for name in ("ACB", "ACB.k8", "CB", "GB"):
            jq = jqs if ARMS[name]["cheap"] == "static" else jqf
            jq.calls = {k: 0 for k in DJ.KEYS}
            x, c = sample(M, ARMS[name], d, nz[n4], lb[n4])
            c6[name] = {"counts": c, "expected": expect(ARMS[name]), "quantizer_calls_per_site": sorted(set(jq.calls.values()))}
        ch["C6_counts"] = c6
        ch["C6_pass"] = all(v["counts"] == v["expected"] and v["quantizer_calls_per_site"] == [50] for v in c6.values())
        # C7 anchor inputs identical; per-sample independence (perturb one sample, others unchanged)
        rec = []
        x0, _ = sample(M, ARMS["ACB"], d, nz[n4], lb[n4], rec=rec)
        ch["C7_anchor_calls_recorded"] = len(rec)
        nzp = nz[n4].clone()
        nzp[3] = torch.randn(nzp[3].shape, generator=torch.Generator().manual_seed(7)).to(dev)
        xp, _ = sample(M, ARMS["ACB"], d, nzp, lb[n4])
        ch["C7_other_samples_unchanged_max_abs"] = float((xp[:3] - x0[:3]).abs().max())
        ch["C7_independent_bitwise"] = bool(torch.equal(xp[:3], x0[:3]))
        # C8 save / resume before (i = 8) and after (i = 9) an anchor, ACB
        c8 = {}
        for sa in (8, 9):
            s_ = sample(M, ARMS["ACB"], d, nz[n4], lb[n4], stop_at=sa)
            p = os.path.join(CK, f"resume_{sa}.pt")
            atomic_torch_save(s_, p)
            r = torch.load(p, map_location="cpu")
            xr, cr = sample(M, ARMS["ACB"], d, nz[n4], lb[n4], resume=r)
            str_ = {k: v.clone() for k, v in jqf.state_out.items()}
            sample(M, ARMS["ACB"], d, nz[n4], lb[n4])
            c8[sa] = bool(torch.equal(xr, x0) and all(torch.equal(str_[k], jqf.state_out[k]) for k in str_) and cr == expect(ARMS["ACB"]))
        ch["C8_resume_bitwise"] = c8
        # C11 G: zero carried state, same guard/range path as AC
        n2 = 2
        xs = torch.randn(n2, 4, 32, 32, generator=torch.Generator().manual_seed(1)).to(dev)
        labt = torch.as_tensor(lb[:n2], device=dev)
        yy = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        tsx = torch.full((n2,), 500, device=dev, dtype=torch.long)
        CTX.step, CTX.n = 20, n2
        cmf = M["fb"].m
        jqf.zero_state = False
        jqf.state_in = {k: torch.zeros(2 * n2, jqf.U.shape[1], (4608 if k.endswith("fc2") else 1152), device=dev, dtype=torch.float16) for k in DJ.KEYS}
        oa = guided(cmf, xs, tsx, yy, n2)
        jqf.zero_state, jqf.state_in = True, {k: torch.randn_like(v) for k, v in jqf.state_in.items()}
        og = guided(cmf, xs, tsx, yy, n2)
        ch["C11_G_ignores_state_equals_zero_state_AC"] = bool(torch.equal(oa, og))
        ch["C11_G_mode_fb_path"] = jqf.mode == "fb"
        CTX.step = None
        # C and G functionally identical? (dev 60200-60215 trajectories)
        xg, _ = sample(M, ARMS["G"], d, nz, lb)
        ch["C_vs_G_bitwise"] = bool(torch.equal(xg.cpu(), ref["C"].cpu()))
        ch["C_vs_G_rel_l2"] = rl2(xg.cpu(), ref["C"].cpu())
        # C12 diagnostics on/off leave endpoint and state unchanged (ACB)
        Dg = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": fpm}
        jqf.acct = {}
        xd, _ = sample(M, ARMS["ACB"], d, nz[n4], lb[n4], diag=Dg)
        sd = {k: v.clone() for k, v in jqf.state_out.items()}
        jqf.acct = None
        xn, _ = sample(M, ARMS["ACB"], d, nz[n4], lb[n4])
        ch["C12_diag_on_off_bitwise"] = bool(torch.equal(xd, xn) and torch.equal(xn, x0) and all(torch.equal(sd[k], jqf.state_out[k]) for k in sd))
        crit = [ch["C9_synthetic"]["pass"], ch["C9_synthetic"]["coef_K1_zero"], *c1.values(), ch["C1_reproduce_br1_CB1"], *c2.values(), *c3.values(),
                ch["C10_H8_50_reproduces_wa8"], *c4.values(), ch["C5_identical_stateless_zero"], ch["C6_pass"], ch["C7_anchor_calls_recorded"] == 13,
                *c8.values(), ch["C11_G_ignores_state_equals_zero_state_AC"], ch["C11_G_mode_fb_path"], ch["C12_diag_on_off_bitwise"]]
        ch["C7_independence_note"] = "reported; bitwise expected (per-row state, per-sample bridge); a nonzero value would indicate batch-content-dependent kernels"
        ch["pass"] = bool(all(crit))
        ch["peak_mem"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1, default=str), flush=True)
    return 0 if ch["pass"] else 6


def checks_ok():
    ch = jload("checks.json")
    return bool(ch and ch["pass"])


# ------------------------------------------------------------------------------------------------------------------- statistics
def reading(c):
    m, lp = c["rel_mse"], c["lpips"]
    if m["ratio"] <= 0.95 and m["ratio_ci"][1] < 1 and not lp["diff_ci"][0] > 0:
        return "useful (>= 5% lower mean latent MSE, CI upper < 1, no clear LPIPS regression)"
    if m["ratio_ci"][1] < 1:
        return "smaller positive" + (" (LPIPS clearly worse)" if lp["diff_ci"][0] > 0 else " (below the 5% threshold)")
    if m["ratio_ci"][0] > 1:
        return "harmful"
    return "inconclusive"


def inter(Mx, g, gb, a, ab):
    """I = [E(gb) - E(ab)] - [E(g) - E(a)]: feedback's absolute gain with bridge minus without."""
    G_, GB_, A_, AB_ = (np.asarray(Mx[x]["rel_mse"]) for x in (g, gb, a, ab))
    idx = boot_idx(len(G_))
    f = lambda i: (GB_[i].mean() - AB_[i].mean()) - (G_[i].mean() - A_[i].mean())
    al = np.arange(len(G_))
    return {"I": float(f(al)), "I_ci": ci([f(i) for i in idx]), "gain_with_bridge": float(GB_.mean() - AB_.mean()),
            "gain_with_bridge_ci": ci([GB_[i].mean() - AB_[i].mean() for i in idx]), "gain_without_bridge": float(G_.mean() - A_.mean()),
            "gain_without_bridge_ci": ci([G_[i].mean() - A_[i].mean() for i in idx]), "definition": f"[E({gb}) - E({ab})] - [E({g}) - E({a})]"}


def noninferior(c, margin=1.05, lp_margin=0.005):
    m, lp = c["rel_mse"], c["lpips"]
    if m["ratio_ci"][1] < margin and lp["diff_ci"][1] < lp_margin:
        return "noninferior within margins"
    if m["ratio_ci"][0] >= margin or lp["diff_ci"][0] >= lp_margin:
        return "inferior beyond margin"
    return "unresolved"


def evaluate(lat, dev, pairs):
    Mx = metrics(lat, dev)
    return Mx, {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in pairs}


def show(Mx, C):
    for a, v in summary(Mx).items():
        print(f"  {a}: rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f} PSNR {v['psnr']['mean']:.2f} LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in C.items():
        m = v["rel_mse"]
        print(f"  {k_}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] wins {m['wins_x']}/{m['n']} "
              f"dLPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['diff_ci'][0]:+.4f}, {v['lpips']['diff_ci'][1]:+.4f}]", flush=True)


def per_batch_h(arms, timing_src=None):
    """estimate seconds per 32-image batch for arms from measured timings (s1 / checks), with fallbacks."""
    fb_s = {"static": 60.0, "fb": 92.0, "G": 92.0}
    meas = {}
    for f in ("s1_part.pt",):
        p = os.path.join(CK, f)
        if os.path.exists(p):
            P = torch.load(p, map_location="cpu")
            meas = {a: float(np.median(v)) for a, v in P["timing"].items()}
    tot = 0.0
    for a in arms:
        sp = ARMS[a]
        if a in meas:
            tot += meas[a]
        elif "high_only" in sp:
            tot += meas.get("FP50", 21.0)
        else:
            base = {"static": "C", "fb": "AC", "G": "G"}[sp["cheap"]]
            tot += meas.get(base, fb_s[sp["cheap"]])
    return tot / 3600


# ------------------------------------------------------------------------------------------------------------------- stage runner
def run_stage(dev, stage, seeds, pairs_fn, extra_fn=None, reuse=None):
    arms = STAGE_ARMS[stage]
    pb = per_batch_h(arms)
    est = pb * len(seeds) / BS * 1.08 + 0.05                                    # + decode / LPIPS / bootstrap
    if not fits(stage, est, SHUTDOWN_RESERVE_H):
        jdump(f"{stage}.json", {"status": "skipped (budget/wall)", "estimate_h": est})
        return 4
    with Run(STAGE, stage, dev, {"seeds": [seeds[0], seeds[-1]], "arms": arms, "batch": BS}) as run:
        provenance()
        M = load_models(dev, needs(arms))
        d = make_diffusion(T)
        torch.cuda.reset_peak_memory_stats()
        P, full = gen(M, d, seeds, arms, f"{stage}_part", est_batch_h=pb)
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        del M
        torch.cuda.empty_cache()
        if not full:
            jdump(f"{stage}.json", {"status": "partial (stopped by budget/wall)", "batches_done": len(P["done"]), **run.record()})
            return 5
        lat = cat(P, arms)
        if reuse:
            lat.update(reuse())
        atomic_torch_save(lat, os.path.join(CK, f"{stage}_latents.pt"))
        Mx, C = evaluate(lat, dev, pairs_fn(lat))
        out = {"status": "complete", "seeds": [seeds[0], seeds[-1]], "n": len(seeds), "arms": Mx, "summary": summary(Mx), "calls": P["cnt"], "contrasts": C,
               "timing_s_per_batch32": {a: {"median": float(np.median(v)), "all": v} for a, v in P["timing"].items()},
               "timing_note": "CUDA-synchronized wall per 32-image trajectory, all stage models resident (harness peak); fake-quant simulation",
               "peak_mem_harness": peak, **run.record()}
        if extra_fn:
            out.update(extra_fn(Mx, C))
    jdump(f"{stage}.json", out)
    show(Mx, C)
    return 0


def first128(stage, arms):
    def f():
        L = torch.load(os.path.join(CK, f"{stage}_latents.pt"), map_location="cpu")
        return {a: L[a][:128] for a in arms}
    return f


def s1(dev):
    pairs = lambda lat: [("ACB", "CB"), ("ACB", "GB"), ("AC", "G"), ("ACB", "AC"), ("CB", "C"), ("AC", "C"), ("GB", "G"), ("G", "C"), ("GB", "CB")]
    extra = lambda Mx, C: {"primary": "ACB_vs_CB", "reading_primary": reading(C["ACB_vs_CB"]), "reading_ACB_vs_GB": reading(C["ACB_vs_GB"]),
                           "reading_AC_vs_G": reading(C["AC_vs_G"]), "interaction_matched": inter(Mx, "G", "GB", "AC", "ACB"),
                           "interaction_package": inter(Mx, "C", "CB", "AC", "ACB")}
    return run_stage(dev, "s1", CONF, pairs, extra)


def s1_done():
    r = jload("s1.json")
    return bool(r and r.get("status") == "complete")


def s2(dev):
    if not s1_done():
        jdump("s2.json", {"status": "blocked (Stage 1 incomplete)"})
        return 4
    ch = jload("checks.json")
    if ch["C_vs_G_bitwise"]:
        STAGE_ARMS["s2"].remove("GB.w8")
    pairs = lambda lat: ([("CB.w8", "C.w8"), ("ACB.w8", "CB.w8")] + ([("ACB.w8", "GB.w8")] if "GB.w8" in lat else []) +
                         [("CB.w8", "CB"), ("ACB.w8", "ACB"), ("CB.w8", "H8_50"), ("ACB.w8", "H8_50"), ("AC.w8", "C.w8"), ("H8_50", "CB")])
    extra = lambda Mx, C: {"reading_CB.w8_vs_C.w8": reading(C["CB.w8_vs_C.w8"]), "reading_ACB.w8_vs_CB.w8": reading(C["ACB.w8_vs_CB.w8"]),
                           "note": "W8 anchors measure cheap-minus-W8; all fidelity vs FP50 (first 128 of the Stage-1 FP50, same batches); H8_50 row = W8's own error"}
    return run_stage(dev, "s2", CONF128, pairs, extra, reuse=first128("s1", ["FP50", "CB", "ACB", "C", "AC"]))


def s3(dev):
    if not s1_done():
        jdump("s3.json", {"status": "blocked (Stage 1 incomplete)"})
        return 4
    w8 = jload("s2.json") or {}

    def reuse():
        r = first128("s1", ["FP50", "C", "CB", "AC", "ACB"])()
        if w8.get("status") == "complete":
            L = torch.load(os.path.join(CK, "s2_latents.pt"), map_location="cpu")
            r.update({a: L[a] for a in ("C.w8", "CB.w8")})
        return r

    def pairs(lat):
        p = [("CB.k8", "C.k8"), ("ACB.k8", "AC.k8"), ("ACB.k8", "CB.k8"), ("AC.k8", "C.k8"), ("CB.w8.k8", "C.w8.k8"), ("ACB.w8.k8", "AC.w8.k8"),
             ("ACB.w8.k8", "CB.w8.k8"), ("CB.k8", "C"), ("ACB.k8", "C"), ("CB.k8", "CB"), ("ACB.k8", "CB")]
        if "C.w8" in lat:
            p += [("CB.w8.k8", "C.w8"), ("ACB.w8.k8", "C.w8"), ("CB.w8.k8", "CB.w8"), ("ACB.w8.k8", "CB.w8")]
        return p

    def extra(Mx, C):
        ni = {k: noninferior(C[k]) for k in C if k.endswith("_vs_C") or k.endswith("_vs_C.w8")}
        return {"noninferiority_vs_ordinary_K4": ni, "margin": "upper 95% CI of E(bridge_K8)/E(C_K4) < 1.05 and ΔLPIPS upper bound < 0.005 (exploratory)",
                "calls": "K8: 50 cheap + 7 high; K4: 50 + 13"}
    return run_stage(dev, "s3", CONF128, pairs, extra, reuse=reuse)


def s4(dev):
    if not s1_done():
        jdump("s4.json", {"status": "blocked (Stage 1 incomplete)"})
        return 4
    pairs = lambda lat: [("LCB", "CB"), ("LACB", "ACB"), ("LACB", "LCB")]
    extra = lambda Mx, C: {"reading_LCB_vs_CB": reading(C["LCB_vs_CB"]), "reading_LACB_vs_ACB": reading(C["LACB_vs_ACB"]),
                           "note": "LACB/LCB changes both the refit checkpoint and feedback; not attributable to feedback alone"}
    return run_stage(dev, "s4", CONF128, pairs, extra, reuse=first128("s1", ["FP50", "CB", "ACB"]))


# ------------------------------------------------------------------------------------------------------------------- s5 diagnostics
@torch.no_grad()
def s5(dev):
    if not fits("s5", 0.12, SHUTDOWN_RESERVE_H):
        jdump("s5.json", {"status": "skipped (budget/wall)"})
        return 4
    arms = ["C", "AC", "CB", "ACB", "G", "GB"]
    with Run(STAGE, "s5", dev, {"seeds": [DIAG[0], DIAG[-1]], "arms": arms}) as run:
        M = load_models(dev, {"fp", "static", "fb"})
        d = make_diffusion(T)
        Ac, bc = coeffs(d)
        nz, lb = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        n = len(DIAG)
        labt = torch.as_tensor(lb, device=dev)
        y = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        out = {"seeds": [DIAG[0], DIAG[-1]], "status": "complete", "arms": {},
               "definitions": {"uncentered_cosine": "<u, v> / (|u| |v|) pooled over intervals, images, channels, pixels",
                               "centered_correlation": "Pearson correlation of the flattened vectors after subtracting each vector's mean",
                               "H": "sum_j w_j (e_{a+j} - e_a), w_j = b_{a+j} prod_{m>j} A_{a+m}", "Hhat": "bridge estimate (e_c - e_a) sum_j (j/k) w_j",
                               "replay": "FP replay from x_{a+1} over the interval (starts 0, 20, 40) vs the live pre-correction x_c"},
               "limitation": "each arm on its own trajectory; development images; diagnostics never feed live inference"}
        for name in arms:
            sp = ARMS[name]
            jq = M["static" if sp["cheap"] == "static" else "fb"].jq
            D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
            jq.acct = {} if sp["cheap"] != "static" else None
            x, c = sample(M, sp, d, nz, lb, diag=D)
            acct = jq.acct
            jq.acct = None
            e = D["e"]
            S = {k: 0.0 for k in ("raw", "held", "jump", "HH", "hh", "Hh", "res")}
            cen = []
            for a in range(0, 45, 4):
                cc = a + 4
                w = weights(Ac, bc, a, 4)
                H = sum(wj * (e[a + j] - e[a]) for j, wj in zip(range(1, 4), w))
                hh = bridge_coef(Ac, bc, a, 4) * (e[cc] - e[a])
                S["raw"] += sum(float(e[a + j].pow(2).sum()) for j in range(0, 4))
                S["held"] += sum(float((e[a + j] - e[a]).pow(2).sum()) for j in range(1, 4))
                S["jump"] += float((e[cc] - e[a]).pow(2).sum())
                S["HH"] += float(H.pow(2).sum()); S["hh"] += float(hh.pow(2).sum()); S["Hh"] += float((H * hh).sum())
                S["res"] += float((H - hh).pow(2).sum())
                cen.append((H.flatten(), hh.flatten()))
            Hc, hc = torch.cat([u for u, _ in cen]), torch.cat([v for _, v in cen])
            r = {"raw_discrepancy_energy": S["raw"], "held_error_energy": S["held"], "anchor_jump_energy": S["jump"],
                 "H_energy": S["HH"], "Hhat_energy": S["hh"], "residual_over_H": S["res"] / S["HH"], "cos_H_Hhat": S["Hh"] / math.sqrt(S["HH"] * S["hh"]),
                 "centered_corr_H_Hhat": float(torch.corrcoef(torch.stack([Hc, hc]))[0, 1]), "magnitude_Hhat_over_H": math.sqrt(S["hh"] / S["HH"]),
                 "bridge_jump_norms": D["jump"], "discrepancy_age_steps": "0..3 within each K4 interval", "calls": c}
            rep = {}
            for a in (0, 20, 40):
                xr = D["x"][a + 1]
                for j in range(a + 1, a + 4):
                    CTX.step, CTX.n = j, n
                    t = torch.full((n,), T - 1 - j, device=dev, dtype=torch.long)
                    xr = d.ddim_sample(lambda xx, ts: guided(M["fp"], xx, ts, y, n), xr, t, clip_denoised=False, eta=0.0)["sample"]
                dx = (D["x"][a + 4] - xr).double().cpu()
                hh = bridge_coef(Ac, bc, a, 4) * (e[a + 4] - e[a])
                rep[a] = {"cos_dx_Hhat": float((dx * hh).sum() / (dx.norm() * hh.norm())), "Hhat_over_dx": float(hh.norm() / dx.norm()),
                          "residual_after_bridge": float((dx - hh).pow(2).sum() / dx.pow(2).sum()), "dx_energy": float(dx.pow(2).sum())}
            CTX.step = None
            r["replay"] = rep
            if acct:
                tot = {}
                for v in acct.values():
                    for kk, vv in v.items():
                        tot[kk] = tot.get(kk, 0.0) + vv
                for kk in [q for q in tot if q.startswith("rate_")]:
                    tot[kk] /= tot["n"]
                r["feedback_accounting_all_sites"] = tot
            else:
                r["feedback_accounting_all_sites"] = "not applicable (static quantizer has no feedback state)"
            out["arms"][name] = r
            print(f"  {name}: raw {S['raw']:.4g} held {S['held']:.4g} jump {S['jump']:.4g} cos(H,Hhat) {r['cos_H_Hhat']:.3f} res {r['residual_over_H']:.3f}", flush=True)
        out.update(run.record())
    jdump("s5.json", out)
    return 0


# ------------------------------------------------------------------------------------------------------------------- s6 FID-2k
@torch.no_grad()
def s6(dev):
    from boa_dit.eval_fid import fid_from_stats, inception_stats
    from boa_dit.fidboot import FIDRef, paired_boot
    if not s1_done():
        jdump("s6.json", {"status": "blocked (Stage 1 incomplete)"})
        return 4
    arms = ["FP50", "CB", "ACB"]
    pb = per_batch_h(arms)
    est = pb * len(FID_SEEDS) / BS * 1.05 + 0.10
    if not fits("s6", est, SHUTDOWN_RESERVE_H + PLAN_H["s7"]):
        jdump("s6.json", {"status": "skipped (full three-arm FID-2k does not fit the remaining budget/wall)", "required_estimate_h": est,
                          "spent_at_decision_h": spent(), "wall_left_h": wall_left_h()})
        return 4
    with Run(STAGE, "s6", dev, {"seeds": [FID_SEEDS[0], FID_SEEDS[-1]], "labels": "i // 2"}) as run:
        provenance()
        M = load_models(dev, needs(arms))
        d = make_diffusion(T)
        P, full = gen(M, d, FID_SEEDS, arms, "s6_part", labels=FID_LABELS, est_batch_h=pb)
        del M
        torch.cuda.empty_cache()
        if not full:
            jdump("s6.json", {"status": "partial (stopped by budget/wall); no FID reported", "batches_done": len(P["done"]), **run.record()})
            return 5
        lat = cat(P, arms)
        vae = load_vae(dev)
        ref = np.load(HASHES["fid_ref_stats"][0])
        feats, fid = {}, {}
        from pytorch_fid.inception import InceptionV3
        net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
        for a in arms:
            fs = []
            for b0 in range(0, len(lat[a]), 200):
                im = decode(vae, lat[a][b0:b0 + 200].to(dev))
                fs.append(net(im.to(dev).float() / 255)[0].squeeze(-1).squeeze(-1).double().cpu())
            F = torch.cat(fs).numpy()
            feats[a] = F
            fid[a] = fid_from_stats(F.mean(0), np.cov(F, rowvar=False), ref["mu"], ref["sigma"])
            print(f"  FID-2k {a}: {fid[a]:.3f}", flush=True)
        boot = paired_boot(FIDRef(ref["mu"], ref["sigma"]), feats, [("ACB", "CB"), ("CB", "FP50"), ("ACB", "FP50")], n_boot=1000, seed=0)
        out = {"status": "complete", "fid2k": fid, "paired_boot": boot, "n": len(FID_SEEDS), "labels": "two per class (label = index // 2)",
               "reference": {"stats": HASHES["fid_ref_stats"], "images": HASHES["fid_ref_images"], "extractor": "pytorch-fid InceptionV3 pool3, resize to 299"},
               "note": "generated-sample bootstrap only; conditional on the reference and n; small-sample FID bias not removed",
               "timing_s_per_batch32": {a: float(np.median(v)) for a, v in P["timing"].items()}, **run.record()}
    jdump("s6.json", out)
    return 0


# ------------------------------------------------------------------------------------------------------------------- s7 timing
@torch.no_grad()
def s7(dev):
    if not s1_done():
        jdump("timing.json", {"status": "blocked (Stage 1 incomplete)"})
        return 4
    arms = ["FP50", "C", "CB", "AC", "ACB"]
    meas = {a: v["median"] for a, v in (jload("s1.json") or {}).get("timing_s_per_batch32", {}).items()}
    est32 = sum(meas.get(a, 90.0) for a in arms) * 6 / 3600
    est1 = 0.08
    do32 = fits("s7 (batch 1 + 32)", est1 + est32 + 0.05, SHUTDOWN_RESERVE_H)
    if not do32 and not fits("s7 (batch 1 only)", est1 + 0.03, SHUTDOWN_RESERVE_H):
        jdump("timing.json", {"status": "skipped (budget/wall)"})
        return 4
    with Run(STAGE, "s7", dev, {"batches": [1, 32] if do32 else [1], "repeats": "1 warm-up + 5"}) as run:
        d = make_diffusion(T)
        out = {"status": "complete", "note": "diagnostics off; warm-up + 5 synchronized whole-trajectory repeats (denoising only); only the arm's models "
               "resident; fake-quant simulation (fp32 dequantized weights), not packed-kernel evidence", "arms": {}}
        for bs in ([1, 32] if do32 else [1]):
            seeds = DEV[:1] if bs == 1 else (DEV + DEV)[:32]
            nz, lb = noise_for_seeds(seeds, dev), labels_for_seeds(seeds)
            for a in arms:
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                m0 = torch.cuda.memory_allocated()
                tl = time.perf_counter()
                M = load_models(dev, needs([a]))
                torch.cuda.synchronize()
                load_s = time.perf_counter() - tl
                model_bytes = torch.cuda.memory_allocated() - m0
                ts = []
                for _ in range(6):
                    torch.cuda.synchronize()
                    t_ = time.perf_counter()
                    x, c = sample(M, ARMS[a], d, nz, lb)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t_)
                sb = 0
                if "high_only" not in ARMS[a] and ARMS[a]["cheap"] != "static":
                    sb = M["fb"].jq.state_bytes() / bs
                out["arms"][f"{a}@bs{bs}"] = {"load_s": load_s, "warmup_s": ts[0], "timed_s": ts[1:], "median_s": float(np.median(ts[1:])),
                                             "spread_s": [float(min(ts[1:])), float(max(ts[1:]))], "per_image_s": float(np.median(ts[1:])) / bs,
                                             "peak_allocated": torch.cuda.max_memory_allocated() - m0, "peak_reserved": torch.cuda.max_memory_reserved(),
                                             "model_bytes_resident": int(model_bytes), "feedback_state_bytes_per_image_incl_cfg": sb,
                                             "bridge_state_bytes_per_image": 4 * 4 * 32 * 32 if ARMS[a].get("bridge") else 0, "calls": c}
                print(f"  {a} bs{bs}: median {np.median(ts[1:]):.2f} s", flush=True)
                del M
            g = lambda k: out["arms"].get(k, {}).get("median_s")
            if g(f"CB@bs{bs}") and g(f"C@bs{bs}"):
                out[f"bridge_increment_s@bs{bs}"] = {"static": g(f"CB@bs{bs}") - g(f"C@bs{bs}"), "feedback": g(f"ACB@bs{bs}") - g(f"AC@bs{bs}")}
                out[f"feedback_overhead_s@bs{bs}"] = {"anchored": g(f"AC@bs{bs}") - g(f"C@bs{bs}"), "bridge": g(f"ACB@bs{bs}") - g(f"CB@bs{bs}")}
                out[f"cost_condition@bs{bs}"] = {"rule": "50 C_cheap + n C_high + overhead < 50 C_high (measured whole pipelines)",
                                                 "C_vs_FP50": g(f"C@bs{bs}") < g(f"FP50@bs{bs}"), "CB_vs_FP50": g(f"CB@bs{bs}") < g(f"FP50@bs{bs}"),
                                                 "ACB_vs_FP50": g(f"ACB@bs{bs}") < g(f"FP50@bs{bs}")}
        out["sana"] = "not run: no previously validated real-kernel SANA timing command is installed in this environment (Monday handoff)"
        out.update(run.record())
    jdump("timing.json", out)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "s1", "s2", "s3", "s4", "s5", "s6", "s7"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    launch_time()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if a.cmd != "checks" and not checks_ok():
        print("[abn1] checks have not passed: stage blocked", flush=True)
        jdump(f"{a.cmd}.json" if a.cmd not in ("s7",) else "timing.json", {"status": "blocked (checks not passed)"})
        sys.exit(4)
    rc = {"checks": checks, "s1": s1, "s2": s2, "s3": s3, "s4": s4, "s5": s5, "s6": s6, "s7": s7}[a.cmd](dev)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
