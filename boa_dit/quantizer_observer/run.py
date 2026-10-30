"""Quantizer-informed error observability (results/quantizer_observer/PROTOCOL.md). Stage tag qo1.
Bed: common smoothed DiT-XL/2 W4 + the frozen dev-selected E_L u100 carry, static single8 A8 with DCT-4 feedback, DDIM-50, cfg 1.5
(boa_dit.dit_joint.run). Measurement only: the deployed quantizer arithmetic is the parent JointQuant._fb, untouched; the instrument recomputes
its intermediates from the same inputs and checks the consumed value.
GPU stages : checks profile trace_fit trace_dev trace_conf dev_orig  [live_dev live_conf only if the observer gate passes]
CPU stages : inventory headroom accounting errstats observer_select observer_confirm cost report  (boa_dit/quantizer_observer/analysis.py)
Run: python -m boa_dit.quantizer_observer.run <stage>
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
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as J
from boa_dit.followup.carry_opt import Sched, guided_eps
from boa_dit.model import block_linear, labels_for_seeds, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample
from boa_dit.temporal import act as ACT

STAGE = "qo1"
OUT = os.path.join(RESULTS, "quantizer_observer")
CK = os.path.join(CKPT_DIR, "quantizer_observer")
T, CFG = J.T, J.CFG
CAP_GPU_H = 2.0
SPLITS = {"fit": list(range(90000, 90064)), "dev": list(range(90100, 90132)), "conf": list(range(90200, 90264)), "live": list(range(90300, 90364))}
PRIMARY = os.path.join(CKPT_DIR, "dit_joint", "carry_E_L_u100.pt")
PRIMARY_SHA = "c2b70026eee388586c7a6881128469be360b2d3c26061819cd996507849f3550"
KEYS = J.KEYS
TERMS = ("p", "-p_next", "c0", "g", "rho_u", "c1", "-(I-P)r", "-a", "eta")       # q - x = sum(TERMS) (float64)
EFEAT = ("E_c0", "E_g", "E_c1", "E_a", "E_discard", "E_eta", "rate_c0", "rate_g", "rate_c1", "rate_spare")
NPROJ = 8
BS = 32


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


class Ceiling(Exception):
    pass


def guard(label, est_h):
    s = spent()
    ok = s + est_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "log": []}
    b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h, "ok": ok})
    jdump("budget.json", b)
    print(f"[qo1 budget] {label}: spent {s:.3f} GPU-h, estimate {est_h:.2f} h -> {'OK' if ok else 'STOP (ceiling)'}", flush=True)
    if not ok:
        raise Ceiling(label)


def est(stage, default=0.3):
    p = jload("profile.json")
    return float(p["projected_h"].get(stage, default)) if p else default


# ------------------------------------------------------------------------------------------------ instrumented feedback quantizer
def chan_proj(dev):
    """fixed seeded channel projections [C, 8] per site (seed 1000 + site index), unit-norm columns."""
    tabs = torch.load(J.TABLES, map_location="cpu")
    out = {}
    for j, k in enumerate(KEYS):
        C = _site_dim(k)
        g = torch.Generator().manual_seed(1000 + j)
        P = torch.randn(C, NPROJ, generator=g, dtype=torch.float64)
        out[k] = (P / P.norm(dim=0, keepdim=True)).to(dev)
    return out


def _site_dim(k):
    return 4608 if k.endswith("mlp.fc2") else 1152


class InstrQuant(J.JointQuant):
    """JointQuant 'fb' whose consumed value comes from the parent _fb (deployed arithmetic). Afterwards it recomputes the identical fp32
    intermediates and records, per site and step, the guarded compressed-feedback accounting (float64) and the R / E / S features."""

    def __init__(self, model, dev, gram=False, feats=True):
        super().__init__(model, dev, "fb")
        self.gram, self.feats = gram, feats
        self.Pch = chan_proj(dev)
        self.Ud = self.U.double()
        self.acc = {}                                                         # (t, key) -> dict of float64 GPU tensors
        self.buf = {"R": {}, "S0": {}, "Sg": {}, "E": {}}
        self.mismatch = 0

    def _fb(self, key, t, xs, D, Z, qm, lo):
        U = self.U
        shp = (xs.shape[0], U.shape[1], xs.shape[2])
        if key not in self.state_in or tuple(self.state_in[key].shape) != shp or (self.auto and t == 0):
            self.state_in[key] = torch.zeros(shp, device=xs.device, dtype=torch.float16)
        s_in = self.state_in[key]
        consumed = super()._fb(key, t, xs, D, Z, qm, lo)                     # deployed arithmetic (writes state_out[key])
        self._instrument(key, t, xs, s_in, self.state_out[key], consumed, D, Z, qm, lo)
        return consumed

    @torch.no_grad()
    def _instrument(self, key, t, xs, s_in, s_out, consumed, D, Z, qm, lo):
        U, Ud = self.U, self.Ud
        p = torch.einsum("tk,rkc->rtc", U, s_in.float())                    # the same fp32 ops as JointQuant._fb
        hi_rep = (qm - Z[t]) * D[t]
        xcl = torch.minimum(torch.maximum(xs, lo), hi_rep)
        r_fed = torch.minimum(torch.maximum(p, lo - xcl), hi_rep - xcl)
        u = xcl + r_fed
        uc = torch.minimum(torch.maximum(u, lo), hi_rep)
        code = torch.clamp(torch.round(torch.clamp(uc, lo, hi_rep) / D[t]) + Z[t], 0, qm)
        Dd = float(D[t])
        rows = xs.shape[0]
        G = torch.zeros(len(TERMS), len(TERMS), dtype=torch.float64, device=xs.device) if self.gram else None
        qx = res = None
        q_re = ACT.q_static(uc, D[t], Z[t], torch.zeros((), device=xs.device), qm, hi_code=qm).to(torch.float16).float()
        self.mismatch += int((q_re != consumed).sum())
        E = torch.zeros(rows, len(EFEAT), dtype=torch.float64, device=xs.device)
        S0 = torch.zeros(rows, 4, xs.shape[2], dtype=torch.float64, device=xs.device)
        Sg = torch.zeros_like(S0)
        for a0 in range(0, rows, 16):
            sl = slice(a0, a0 + 16)
            x_, p_, xcl_, rf_, u_, uc_, q_ = (v[sl].double() for v in (xs, p, xcl, r_fed, u, uc, consumed))
            pn_ = torch.einsum("tk,rkc->rtc", Ud, s_out[sl].double())
            c0 = xcl_ - x_
            g = rf_ - p_
            rho = u_ - (xcl_ + rf_)
            c1 = uc_ - u_
            r_raw = uc_ - q_
            r = r_raw.clamp(-Dd / 2, Dd / 2)
            a = r_raw - r
            Pr = torch.einsum("tk,rkc->rtc", Ud, torch.einsum("tk,rtc->rkc", Ud, r))
            Ir = r - Pr
            eta = pn_ - Pr
            terms = (p_, -pn_, c0, g, rho, c1, -Ir, -a, eta)
            lhs = q_ - x_
            resid = lhs - sum(terms)
            if self.gram:
                M = torch.stack([v.reshape(-1) for v in terms])
                G += M @ M.T
            acc_q = lhs.pow(2).sum()
            acc_r = resid.pow(2).sum()
            qx = acc_q if qx is None else qx + acc_q
            res = acc_r if res is None else res + acc_r
            if self.feats:
                E[sl, 0] = c0.pow(2).sum((1, 2))
                E[sl, 1] = g.pow(2).sum((1, 2))
                E[sl, 2] = (uc[sl] - u[sl]).double().pow(2).sum((1, 2))
                E[sl, 3] = a.pow(2).sum((1, 2))
                E[sl, 4] = Ir.pow(2).sum((1, 2))
                E[sl, 5] = eta.pow(2).sum((1, 2))
                n = xs.shape[1] * xs.shape[2]
                E[sl, 6] = (xcl[sl] != xs[sl]).sum((1, 2)).double() / n
                E[sl, 7] = (r_fed[sl] != p[sl]).sum((1, 2)).double() / n
                E[sl, 8] = (u[sl] != uc[sl]).sum((1, 2)).double() / n
                E[sl, 9] = (code[sl] == qm).sum((1, 2)).double() / n
                S0[sl] = torch.einsum("tk,rtc->rkc", Ud, c0)
                Sg[sl] = torch.einsum("tk,rtc->rkc", Ud, g)
            del x_, p_, xcl_, rf_, u_, uc_, q_, pn_, c0, g, rho, c1, r_raw, r, a, Pr, Ir, eta, terms, lhs, resid
        rec = {"qx": qx, "res": res, "n": xs.numel(),
               "cnt": torch.stack([(xcl != xs).sum(), (r_fed != p).sum(), (u != uc).sum(), ((consumed - uc).abs() > Dd / 2).sum(), (code == qm).sum()]).double()}
        if self.gram:
            rec["G"], rec["qx_g"], rec["res_g"] = G, qx, res
        prev = self.acc.get((t, key))
        self.acc[(t, key)] = rec if prev is None else {k: (prev[k] + v) for k, v in rec.items()}
        if self.feats:
            P = self.Pch[key]
            self.buf["E"][key] = E.float()
            self.buf["S0"][key] = (S0 @ P).float()
            self.buf["Sg"][key] = (Sg @ P).float()
            self.buf["R"][key] = (s_out.double() @ P).float()

    def acc_cpu(self):
        out = {}
        for (t, k), rec in self.acc.items():
            out[(t, k)] = {n: (v.cpu() if torch.is_tensor(v) else v) for n, v in rec.items()}
        return out


def attach_capture(linear, V, delta, store, key, U):
    """deflate.carry.attach with the same op order and dtypes, additionally storing the 2x2 spatial DCT of the bottleneck z = x V."""
    V, delta = V.to(linear.weight.device), delta.to(linear.weight.device)

    def hook(mod, inp, out):
        x = inp[0]
        dt = out.dtype
        if dt == torch.float16:
            z = x.to(V.dtype) @ V
            y = out + (z @ delta.T).to(dt)
        else:
            z = x.float() @ V.float()
            y = out + z @ delta.float().T
        if store is not None:
            store[key] = torch.einsum("tk,rtc->rkc", U, z.float())
        return y
    return linear.register_forward_hook(hook)


def install_capture(model, carry, store, U):
    hs = []
    for key, c in carry.items():
        l, ln = key.split(".", 1)
        hs.append(attach_capture(block_linear(model.blocks[int(l[1:])], ln), c["V"], c["delta"], store, key, U))
    return hs


def load_primary():
    assert J.sha(PRIMARY) == PRIMARY_SHA, "primary carry hash differs from the frozen E_L u100"
    return torch.load(PRIMARY, map_location="cpu")


def model_W(dev, carry):
    """common W4 + carry, A16, feedback off (no quantizer hooks)."""
    m = J.common_base_nocarry(dev)
    hs = install(m, carry)
    return m


def model_WA(dev, carry, gram=False, feats=True):
    m = J.common_base_nocarry(dev)
    jq = InstrQuant(m, dev, gram=gram, feats=feats)
    store = {}
    install_capture(m, carry, store if feats else None, jq.U)
    return m, jq, store


def plain_sample(m, seeds, dev, bs=BS):
    d = make_diffusion(T)
    with torch.no_grad():
        return torch.cat([ddim_sample(m, d, noise_for_seeds(seeds[b:b + bs], dev), labels_for_seeds(seeds[b:b + bs]), CFG).cpu() for b in range(0, len(seeds), bs)])


# ------------------------------------------------------------------------------------------------ the trace pass
@torch.no_grad()
def trace(seeds, dev, carry, Wm, Fm, gram=False, feats=True, bs=BS, steps=None):
    """Actual WA trajectory (deployed ddim_sample, feedback state chained, reset at t = 0). Inside the sampler's eps_hook at step i (after the WA
    call), the separate W and FP models are evaluated at the SAME latent/timestep/labels; they have no quantizer, so WA's state advances once."""
    Wq, jq, store = model_WA(dev, carry, gram=gram, feats=feats)
    d = make_diffusion(T)
    sch = Sched(d, dev)
    out = {k: [] for k in ("eps_wa", "eW", "eAW", "x", "end", "C", "R", "S0", "Sg", "E")}
    nsteps = steps or T
    for b0 in range(0, len(seeds), bs):
        s = seeds[b0:b0 + bs]
        n = len(s)
        lab = torch.as_tensor(labels_for_seeds(s), device=dev)
        per = {k: [None] * nsteps for k in ("eps_wa", "eW", "eAW", "x", "C", "R", "S0", "Sg", "E")}

        def hook(i, xx, o):
            if i >= nsteps:
                return o
            fW = guided_eps(Wm, xx, lab, i, sch)
            fF = guided_eps(Fm, xx, lab, i, sch)
            ewa = o[:, :4]
            per["eps_wa"][i], per["x"][i] = ewa.cpu(), xx.cpu()
            per["eAW"][i], per["eW"][i] = (ewa - fW).cpu(), (fW - fF).cpu()
            if feats:
                pk = lambda dct: torch.stack([dct[k] for k in KEYS], 1)                      # [2n, 112, ...]
                two = lambda v: torch.stack([v[:n], v[n:]], 1)                              # [n, 2, 112, ...]
                per["C"][i] = two(pk(store)).half().cpu()
                for f in ("R", "S0", "Sg"):
                    per[f][i] = two(pk(jq.buf[f])).half().cpu()
                per["E"][i] = two(pk(jq.buf["E"])).cpu()
            return o
        jq.t, jq.auto = None, True
        jq.state_in, jq.state_out = {}, {}
        if steps:
            x = ddim_sample_partial(Wq, d, noise_for_seeds(s, dev), labels_for_seeds(s), hook, nsteps)
        else:
            x = ddim_sample(Wq, d, noise_for_seeds(s, dev), labels_for_seeds(s), CFG, eps_hook=hook)
        out["end"].append(x.cpu())
        for k, v in per.items():
            if v[0] is not None:
                out[k].append(torch.stack(v, 1))
        print(f"    trace batch {b0 // bs + 1}/{(len(seeds) + bs - 1) // bs} done", flush=True)
    res = {k: torch.cat(v) for k, v in out.items() if v}
    res["acc"] = jq.acc_cpu()
    res["calls"] = dict(jq.calls)
    res["mismatch"] = jq.mismatch
    jq.remove()
    return res


def ddim_sample_partial(m, d, noise, labels, hook, nsteps):
    """profiling helper: the first nsteps of ddim_sample (same per-step code path)."""
    class Stop(Exception):
        pass

    def h(i, xx, o):
        o = hook(i, xx, o)
        if i == nsteps - 1:
            raise Stop()
        return o
    try:
        ddim_sample(m, d, noise, labels, CFG, eps_hook=h)
    except Stop:
        pass
    return noise


# ------------------------------------------------------------------------------------------------ checks
@torch.no_grad()
def stage_checks(dev):
    res = {}
    with Run(STAGE, "qo1_checks", dev, {}) as run:
        guard("checks", 0.12)
        carry = load_primary()
        res["primary_carry"] = {"path": PRIMARY, "sha256": PRIMARY_SHA}
        base = json.load(open(os.path.join(RESULTS, "dit_joint", "base.json")))
        cal = json.load(open(os.path.join(RESULTS, "dit_joint", "calib_act.json")))
        res["hashes_ok"] = J.sha(J.BASE_STATE) == base["model_sha256"] and J.sha(J.BASE_CARRY) == base["carry_sha256"] and J.sha(J.TABLES) == cal["tables_sha256"]
        seeds = SPLITS["fit"][:4]
        nz = lambda s=seeds: noise_for_seeds(s, dev)
        d = make_diffusion(T)
        # 1. quantization off reproduces the common W4A16 reference (JointQuant 'off' + install == W path)
        Wm = model_W(dev, carry)
        xW = ddim_sample(Wm, d, nz(), labels_for_seeds(seeds), CFG).cpu()
        m2 = J.common_base_nocarry(dev)
        jo = J.JointQuant(m2, dev, "off")
        install(m2, carry)
        res["c1_quant_off_equals_W4A16_reference"] = bool(torch.equal(ddim_sample(m2, d, nz(), labels_for_seeds(seeds), CFG).cpu(), xW))
        # reference deployed WA (JointQuant fb + install, the joint evaluator's path)
        jo.remove()
        jq_ref = J.JointQuant(m2, dev, "fb")
        xWA_ref = J.sample_config(m2, jq_ref, seeds, dev, bs=4)
        state_rows = {k: tuple(v.shape) for k, v in list(jq_ref.state_out.items())[:1]}
        jq_ref.remove()
        del m2
        torch.cuda.empty_cache()
        Fm = J.fp_model(dev)
        # 2. instrumented trace (gram + features + aligned references) leaves feedback sampling bit-exact; repeat reproduces
        tr = trace(seeds, dev, carry, Wm, Fm, gram=True, feats=True, bs=4)
        res["c2_instrumented_trace_equals_deployed_WA"] = bool(torch.equal(tr["end"], xWA_ref))
        tr2 = trace(seeds, dev, carry, Wm, Fm, gram=False, feats=True, bs=4)
        res["c4_repeat_bitwise"] = bool(torch.equal(tr2["end"], tr["end"])) and bool(torch.equal(tr2["eW"], tr["eW"])) and bool(torch.equal(tr2["C"], tr["C"]))
        # 3. rows keep separate states: changing image 2's noise leaves image 1 (both CFG rows) bit-identical
        alt = [seeds[0], 99999991, seeds[2], seeds[3]]
        tr3 = trace(alt, dev, carry, Wm, Fm, gram=False, feats=False, bs=4)
        res["c3_row_independence_img0"] = bool(torch.equal(tr3["end"][0], tr["end"][0])) and bool(torch.equal(tr3["end"][2:], tr["end"][2:]))
        res["c3_state_rows (2n for n=4)"] = state_rows
        # 5. accounting identity closes (float64), stable denominator
        qx = np.array([float(v["qx"]) for v in tr["acc"].values()])
        rr = np.array([float(v["res"]) for v in tr["acc"].values()])
        Gtr = np.array([float(torch.trace(v["G"])) for v in tr["acc"].values()])
        Gsum = np.array([float(v["G"].sum()) for v in tr["acc"].values()])
        den = np.maximum(qx, 1e-12 * qx.mean())
        res["c5_identity_max_rel_resid_energy"] = float((rr / den).max())
        res["c5_identity_max_abs_resid_energy"] = float(rr.max())
        res["c5_gram_sum_vs_qx_max_rel"] = float((np.abs(Gsum - qx) / den).max())
        res["c5_consumed_recompute_mismatch"] = tr["mismatch"]
        # 6. aligned local errors close: e_total = eAW + eW against an independent FP evaluation along the stored WA states
        sch = Sched(d, dev)
        lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
        clo = []
        for i in (0, 25, 49):
            fF = guided_eps(Fm, tr["x"][:, i].to(dev), lab, i, sch).cpu()
            et = tr["eps_wa"][:, i] - fF
            clo.append(float((et - (tr["eAW"][:, i] + tr["eW"][:, i])).norm() / et.norm()))
        res["c6_aligned_closure_rel"] = clo
        # 7. calls per site per sample, CFG ordering
        res["c7_calls_per_site (4 images, 1 trace)"] = sorted(set(tr["calls"].values()))
        res["tolerances"] = {"c1, c2, c3, c4": "bitwise", "c5": "relative residual energy <= 1e-20 (float64 roundoff, squared)", "c6": "<= 1e-6 relative"}
        res["pass"] = all([res["hashes_ok"], res["c1_quant_off_equals_W4A16_reference"], res["c2_instrumented_trace_equals_deployed_WA"], res["c4_repeat_bitwise"],
                           res["c3_row_independence_img0"], res["c5_identity_max_rel_resid_energy"] <= 1e-20, tr["mismatch"] == 0, max(clo) <= 1e-6,
                           res["c7_calls_per_site (4 images, 1 trace)"] == [T]])
        res.update(run.record())
    jdump("checks.json", res)
    print(json.dumps(res, indent=1, default=str), flush=True)
    if not res["pass"]:
        raise SystemExit(6)


# ------------------------------------------------------------------------------------------------ profile
@torch.no_grad()
def stage_profile(dev):
    res = {"gpu": torch.cuda.get_device_name(0), "tf32": [torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32], "dtype": "fp16 autocast"}
    with Run(STAGE, "qo1_profile", dev, {}) as run:
        guard("profile", 0.08)
        carry = load_primary()
        Wm, Fm = model_W(dev, carry), J.fp_model(dev)
        seeds4, seeds32 = SPLITS["fit"][:4], SPLITS["fit"][:32]
        sync = torch.cuda.synchronize
        sync(); t0 = time.time()
        trace(seeds4, dev, carry, Wm, Fm, gram=True, feats=True, bs=4)
        sync(); res["trace_4img_bs4_gram_s"] = time.time() - t0
        for gram in (False, True):
            sync(); t0 = time.time()
            trace(seeds32, dev, carry, Wm, Fm, gram=gram, feats=True, bs=32, steps=3)
            sync(); res[f"trace_bs32_per_step_gram{int(gram)}_s"] = (time.time() - t0) / 3
        m = J.common_base_nocarry(dev)
        jq = J.JointQuant(m, dev, "fb")
        install(m, carry)
        sync(); t0 = time.time()
        J.sample_config(m, jq, seeds32, dev)
        sync(); res["plain_WA_bs32_s"] = time.time() - t0
        jq.remove()
        sync(); t0 = time.time()
        plain_sample(Wm, seeds32, dev)
        sync(); res["W_bs32_s"] = time.time() - t0
        res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        st0, st1 = res["trace_bs32_per_step_gram0_s"] * T, res["trace_bs32_per_step_gram1_s"] * T
        w = res["W_bs32_s"]
        res["projected_h"] = {"trace_fit": (st1 + st0 + 2 * 2 * w) / 3600, "trace_dev": (st0 + 2 * w) / 3600, "trace_conf": (2 * st0 + 2 * 2 * w) / 3600,
                              "dev_orig": (w + res["plain_WA_bs32_s"]) / 3600}
        res["gram_policy"] = "full cross-term Gram on the first fit batch (32 images); energies/features on every batch"
        res["spent_before_h"] = spent()
        res["projected_total_trace_h"] = sum(res["projected_h"].values())
        res.update(run.record())
    jdump("profile.json", res)
    print(json.dumps(res, indent=1), flush=True)


# ------------------------------------------------------------------------------------------------ traces + endpoints
def stage_trace(dev, split):
    f = os.path.join(CK, f"trace_{split}.pt")
    if os.path.exists(f):
        print(f"  trace_{split}: exists, skipped", flush=True)
        return
    with Run(STAGE, f"qo1_trace_{split}", dev, {"split": split}) as run:
        guard(f"trace_{split}", est(f"trace_{split}"))
        seeds = SPLITS[split]
        carry = load_primary()
        Wm, Fm = model_W(dev, carry), J.fp_model(dev)
        parts, accs = [], []
        for b0 in range(0, len(seeds), BS):
            gram = split == "fit" and b0 == 0
            tr = trace(seeds[b0:b0 + BS], dev, carry, Wm, Fm, gram=gram, feats=True)
            accs.append(tr.pop("acc"))
            tr.pop("calls")
            assert tr.pop("mismatch") == 0, "instrument recompute mismatch"
            parts.append(tr)
        tr = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
        acc = {}
        for a in accs:
            for key, rec in a.items():
                new = dict(acc.get(key, {}))
                for n_, v in rec.items():
                    new[n_] = new[n_] + v if n_ in new else v
                acc[key] = new
        tr["acc"] = acc
        tr["seeds"], tr["labels"] = seeds, labels_for_seeds(seeds).tolist()
        tr["W_end"] = plain_sample(Wm, seeds, dev)
        tr["F_end"] = plain_sample(Fm, seeds, dev)
        atomic_torch_save(tr, f)
        rec = {"file": f, "sha256": J.sha(f), "bytes": os.path.getsize(f), "seeds": [seeds[0], seeds[-1]], "gram_images": 32 if split == "fit" else 0, **run.record()}
    m = jload("manifest_traces.json") or {}
    m[split] = rec
    jdump("manifest_traces.json", m)
    print(rec, flush=True)


@torch.no_grad()
def stage_dev_orig(dev):
    f = os.path.join(CK, "dev_orig.pt")
    if os.path.exists(f):
        return
    with Run(STAGE, "qo1_dev_orig", dev, {}) as run:
        guard("dev_orig", est("dev_orig"))
        seeds = SPLITS["dev"]
        c0 = torch.load(J.BASE_CARRY, map_location="cpu")
        Wm = model_W(dev, c0)
        xW = plain_sample(Wm, seeds, dev)
        del Wm
        m = J.common_base_nocarry(dev)
        jq = J.JointQuant(m, dev, "fb")
        install(m, c0)
        xWA = J.sample_config(m, jq, seeds, dev)
        atomic_torch_save({"W_end": xW, "WA_end": xWA, "seeds": seeds, "carry": J.BASE_CARRY}, f)
        rec = {"file": f, "sha256": J.sha(f), **run.record()}
    m = jload("manifest_traces.json") or {}
    m["dev_orig"] = rec
    jdump("manifest_traces.json", m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    a = ap.parse_args()
    s = a.stage
    os.makedirs(CK, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)
    if s in ("inventory", "headroom", "accounting", "errstats", "observer_select", "observer_confirm", "cost", "report", "propagation"):
        from boa_dit.quantizer_observer import analysis
        getattr(analysis, f"stage_{s}")()
        return
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if s in ("checks", "profile") and jload(f"{s}.json"):
        print(f"  {s}: done, skipped", flush=True)
        return
    try:
        if s == "checks":
            stage_checks(dev)
        elif s == "profile":
            stage_profile(dev)
        elif s.startswith("trace_"):
            stage_trace(dev, s[6:])
        elif s == "dev_orig":
            stage_dev_orig(dev)
        elif s.startswith("live_"):
            from boa_dit.quantizer_observer import live
            getattr(live, f"stage_{s}")(dev)
        else:
            raise SystemExit(f"unknown stage {s}")
    except Ceiling as c:
        print(f"CEILING: {c} not run", flush=True)
        raise SystemExit(5)


if __name__ == "__main__":
    main()
