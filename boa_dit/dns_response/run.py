"""DNS response pilot (results/dns_response/PROTOCOL.md). Stage tag r1. A band-wise latent impulse x <- x + theta_j P_low(x) after DNS updates
9/19/29/39/48 on top of faithful DNS (boa_dit/dns/port.py, frozen calibration results/dns/stats.json), calibrated locally (L) or by measured
downstream responses (T). Stages: checks components fit select confirm fid_<arm> report. Each checks the 6 GPU-h / 6 h ceilings first.
Run: python -m boa_dit.dns_response.run <stage>
"""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.dns import port as P
from boa_dit.dns.run import heig, fp_model, load_stats
from boa_dit.model import NULL_CLASS, decode, labels_for_seeds, load_vae, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks
from boa_dit.sample import ddim_sample

STAGE = "r1"
OUT = os.path.join(RESULTS, "dns_response")
CK = os.path.join(CKPT_DIR, "dns_response")
FIT, DEV, CONF, GEN = list(range(72000, 72064)), list(range(72100, 72132)), list(range(72200, 72328)), list(range(74000, 76000))
INSERT = (9, 19, 29, 39, 48)
CKPTS = INSERT + (49,)
TMAX, TSUM, H1, H2 = 0.01, 0.02, 0.002, 0.004
CAP_GPU_H, CAP_ELAPSED_H = 6.0, 6.0
MASKS = [m.double() for m in band_masks(32)]
NB = [float(m.sum()) for m in MASKS]                                          # real degrees of freedom per band (per channel)


def jdump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    p = os.path.join(OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ budget
def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    if not os.path.exists(f):
        return 0.0
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE)


def guard(label, est_h):
    b = jload("budget.json") or {}
    now = datetime.now(timezone.utc)
    if "first_gpu_utc" not in b:
        b["first_gpu_utc"] = now.isoformat()
    el = (now - datetime.fromisoformat(b["first_gpu_utc"])).total_seconds() / 3600
    s = spent()
    ok = s + est_h <= CAP_GPU_H and el + est_h <= CAP_ELAPSED_H
    b.setdefault("log", []).append({"stage": label, "utc": now.isoformat(), "spent_gpu_h": s, "elapsed_h": el, "estimate_h": est_h, "ok": ok})
    b.update({"spent_gpu_h": s, "elapsed_h": el, "caps": {"gpu_h": CAP_GPU_H, "elapsed_h": CAP_ELAPSED_H}})
    jdump("budget.json", b)
    print(f"[r1 budget] {label}: spent {s:.3f} GPU-h, elapsed {el:.2f} h, estimate {est_h:.2f} h -> {'OK' if ok else 'STOP (ceiling)'}", flush=True)
    return ok


# ------------------------------------------------------------------------------------------------ sampler
def p_low(x):
    F = torch.fft.fft2(x.double())
    return torch.fft.ifft2(F * MASKS[0].to(x.device)).real.to(x.dtype)


def plain_coeffs(s):
    ab_t, ab_p = s["abar_t"], s["abar_prev"]
    ws = math.sqrt(ab_p / ab_t)
    we = math.sqrt(1 - ab_p) - math.sqrt(ab_p * (1 - ab_t) / ab_t)
    return ws, we


def variant(mode):
    """frozen DNS schedule with components removed (component-removal ablations, not re-optimized algorithms)."""
    tm = P.tmap(make_diffusion(P.T))
    base = P.schedule(load_stats(), tm)
    out = []
    for s in base:
        s = copy.deepcopy(s)
        if mode in ("plain", "affine"):                                       # ordinary DDIM coefficients
            s["ws"], s["we"] = plain_coeffs(s)
            s["C3sq"] = 1.0
        if mode in ("plain", "sched"):                                        # affine identity
            s["k"], s["b"] = [0.0] * 4, [0.0] * 4
        if mode != "dns":                                                     # DCNC off
            s["a"] = 0.0
        out.append(s)
    return out


@torch.no_grad()
def sample(model, seeds, sched, dev, theta=None, record=False, amp=True):
    """port.dns_sample's arithmetic (bitwise when theta is None/0) plus the impulse after the DNS update of each INSERT step, checkpoint states,
    and per-step state RMS. Batches of 32 (aligned with every other paired run)."""
    th = [0.0] * 5 if theta is None else [float(v) for v in theta]
    xs, cks, rms = [], [], []
    for b0 in range(0, len(seeds), 32):
        sd = seeds[b0:b0 + 32]
        x = noise_for_seeds(sd, dev)
        n = len(x)
        lab = torch.as_tensor(labels_for_seeds(sd), device=dev, dtype=torch.long)
        y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
        gens = [torch.Generator(device=dev).manual_seed(int(s) * 7919 + 7) for s in sd]
        ck, r = [], []
        for i, s in enumerate(sched):
            out = P.raw_out(model, x, s["t"], y, amp)
            if s["a"] != 0:
                a = s["a"]
                noise = torch.stack([(2 * a) * torch.rand((2,) + out.shape[1:], device=dev, generator=g) - a for g in gens])
                out = out + torch.cat([noise[:, 0], noise[:, 1]])
            e = P.cfg_combine(out, n)[:, :4]
            k = torch.tensor(s["k"], device=dev, dtype=torch.float32).view(1, 4, 1, 1)
            bb = torch.tensor(s["b"], device=dev, dtype=torch.float32).view(1, 4, 1, 1)
            e = (e - bb) / (1 + k)
            xh = s["ws"] * x + s["we"] * e
            x = xh / math.sqrt(s["C3sq"])
            if i in INSERT and th[INSERT.index(i)] != 0.0:
                x = x + th[INSERT.index(i)] * p_low(x)
            if not torch.isfinite(x).all():
                raise FloatingPointError(f"non-finite state after update {i}")
            r.append(float(x.pow(2).mean().sqrt()))
            if record and i in CKPTS:
                ck.append(x.cpu())
        xs.append(x.cpu())
        if record:
            cks.append(torch.stack(ck, 1))                                    # [n, 6, 4, 32, 32]
        rms.append(r)
    out = {"x": torch.cat(xs), "rms": np.mean(rms, 0).tolist()}
    if record:
        out["ck"] = torch.cat(cks)
    return out


@torch.no_grad()
def fp_traj(seeds, dev):
    """FP DDIM states at the checkpoint indices (after updates 9/19/29/39/48) and the final output; cached per seed set."""
    f = os.path.join(CK, f"fp_{seeds[0]}_{len(seeds)}.pt")
    if os.path.exists(f):
        return torch.load(f)
    fp = fp_model(dev)
    d = make_diffusion(P.T)
    xs, cks = [], []
    for b0 in range(0, len(seeds), 32):
        sd = seeds[b0:b0 + 32]
        x0, st = ddim_sample(fp, d, noise_for_seeds(sd, dev), labels_for_seeds(sd), P.CFG, record_states=True)
        cks.append(torch.stack([st[j + 1].cpu() for j in INSERT] + [x0.cpu()], 1))
        xs.append(x0.cpu())
    del fp
    torch.cuda.empty_cache()
    out = {"x": torch.cat(xs), "ck": torch.cat(cks)}
    atomic_torch_save(out, f)
    return out


# ------------------------------------------------------------------------------------------------ moments
def bands(x):
    F = torch.fft.fft2(x.double())
    return [torch.fft.ifft2(F * m).real for m in MASKS]


def moments(ck):
    """ck [S, 6, 4, 32, 32] -> DC mean [6, 4], DC sd [6, 4], band variance [6, 4 bands] (centered per element across samples, summed over the band's
    elements, divided by its real degrees of freedom, averaged over channels)."""
    ck = ck.double()
    dc = ck.mean((-1, -2))                                                    # [S, 6, 4]
    V = torch.zeros(ck.shape[1], 4, dtype=torch.float64)
    for b, comp in enumerate(bands(ck)):
        c = comp - comp.mean(0, keepdim=True)
        V[:, b] = (c.pow(2).mean(0).sum((-1, -2)) / NB[b]).mean(-1)
    return {"dc_mean": dc.mean(0), "dc_sd": dc.std(0, unbiased=False), "V": V}


def cross_identity(ckq, ckf):
    """Var(xq) - Var(xf) = Var(d) + 2 Cov(xf, d), same centering, per checkpoint and band."""
    d = ckq.double() - ckf.double()
    Bq, Bf, Bd = bands(ckq.double()), bands(ckf.double()), bands(d)
    res = []
    for b in range(4):
        cq, cf, cd = [t - t.mean(0, keepdim=True) for t in (Bq[b], Bf[b], Bd[b])]
        red = lambda u: (u.mean(0).sum((-1, -2)) / NB[b]).mean(-1)
        lhs = red(cq * cq) - red(cf * cf)
        vd, cov = red(cd * cd), red(cf * cd)
        res.append({"lhs": lhs.tolist(), "var_d": vd.tolist(), "two_cov": (2 * cov).tolist(), "closure_max_rel": float(((lhs - vd - 2 * cov).abs() / lhs.abs().clamp_min(1e-30)).max())})
    return res


def floors_from(fpm):
    return {"sd": 1e-3 * float(fpm["dc_sd"].median()), "V": 1e-6 * float(fpm["V"].median())}


def residual(qm, fpm, fl):
    """m: per checkpoint [4 DC-mean z-scores, 4 log band-variance ratios] -> [6, 8]; floor activity counted."""
    sd = fpm["dc_sd"].clamp_min(fl["sd"])
    mean_r = (qm["dc_mean"] - fpm["dc_mean"]) / sd
    var_r = torch.log(qm["V"].clamp_min(fl["V"]) / fpm["V"].clamp_min(fl["V"]))
    act = int((fpm["dc_sd"] < fl["sd"]).sum() + (fpm["V"] < fl["V"]).sum() + (qm["V"] < fl["V"]).sum())
    return torch.cat([mean_r, var_r], 1), act


W = 0.5                                                                       # 1/sqrt(4) per entry: equal group and checkpoint weights


def J_full(m):
    return float((W * m).pow(2).sum())


def J_local(m):
    return float((W * m[:5]).pow(2).sum())                                   # the five post-impulse checkpoints


# ------------------------------------------------------------------------------------------------ rollouts and responses
class Ctx:
    def __init__(self, dev):
        self.dev = dev
        self.m, _ = heig(dev)
        self.sched = variant("dns")
        self.cache = {}

    def roll(self, seeds, theta, fpt, fl, sched=None):
        key = (seeds[0], len(seeds), tuple(np.round(theta, 9)), id(sched))
        if key in self.cache:
            return self.cache[key]
        r = sample(self.m, seeds, sched or self.sched, self.dev, theta, record=True)
        qm = moments(r["ck"])
        m, act = residual(qm, moments(fpt["ck"]), fl)
        lat = ((r["x"] - fpt["x"]).flatten(1).pow(2).sum(1) / fpt["x"].flatten(1).pow(2).sum(1))
        out = {"m": m, "floor_active": act, "latent": float(lat.mean()), "latent_per": lat.tolist(), "rms": r["rms"], "ck": r["ck"], "x": r["x"],
               "dc_mean": qm["dc_mean"].tolist(), "V": qm["V"].tolist()}
        self.cache[key] = out
        return out


def response(ctx, theta0, fpt, fl, h):
    cols = []
    for j in range(5):
        tp, tm_ = list(theta0), list(theta0)
        tp[j] += h
        tm_[j] -= h
        cols.append(((ctx.roll(FIT, tp, fpt, fl)["m"] - ctx.roll(FIT, tm_, fpt, fl)["m"]) / (2 * h)).flatten())
    return torch.stack(cols, 1)                                               # [48, 5]


def solve(m0, G, theta_c, usable, local):
    """min ||W (m0 + G (theta - theta_c))||^2 (local: only each impulse's own checkpoint block and own column) s.t. |theta_j| <= TMAX,
    sum|theta_j| <= TSUM; unusable columns fixed at theta_c; 1e-9 ||theta||^2 minimum-norm tie-break. SLSQP on theta = p - n."""
    from scipy.optimize import minimize
    m0 = m0.flatten().numpy().astype(np.float64)
    Gn = G.numpy().astype(np.float64)
    if local:
        rows = [np.arange(8 * j, 8 * j + 8) for j in range(5)]
        def f(th):
            return sum(float(np.sum((W * (m0[r] + Gn[r, j] * (th[j] - theta_c[j]))) ** 2)) for j, r in enumerate(rows)) + 1e-9 * float(th @ th)
    else:
        def f(th):
            return float(np.sum((W * (m0 + Gn @ (th - np.asarray(theta_c)))) ** 2)) + 1e-9 * float(th @ th)
    fz = lambda z: f(z[:5] - z[5:])
    cons = [{"type": "ineq", "fun": lambda z: TSUM - z.sum()}]
    bnds = []
    for j in range(5):
        if usable[j]:
            bnds.append((0, TMAX))
        else:
            v = max(theta_c[j], 0.0)
            bnds.append((v, v))
    for j in range(5):
        if usable[j]:
            bnds.append((0, TMAX))
        else:
            v = max(-theta_c[j], 0.0)
            bnds.append((v, v))
    z0 = np.concatenate([np.clip(theta_c, 0, None), np.clip(-np.asarray(theta_c), 0, None)])
    r = minimize(fz, z0, method="SLSQP", bounds=bnds, constraints=cons, options={"ftol": 1e-14, "maxiter": 500})
    th = r.x[:5] - r.x[5:]
    return th.tolist(), {"success": bool(r.success), "pred_obj": float(f(th)), "binding_box": [abs(abs(v) - TMAX) < 1e-6 for v in th],
                         "binding_sum": abs(sum(abs(v) for v in th) - TSUM) < 1e-6}


# ------------------------------------------------------------------------------------------------ stages
def stage_checks(dev):
    if not guard("checks", 0.15):
        raise SystemExit(5)
    res = {}
    with Run(STAGE, "r1_checks", dev, {}) as run:
        m, _ = heig(dev)
        tm = P.tmap(make_diffusion(P.T))
        sched = variant("dns")
        res["c5_schedule_identical_to_faithful_dns"] = sched == P.schedule(load_stats(), tm)
        seeds = DEV[:4]
        lab = labels_for_seeds(seeds)
        ref = P.dns_sample(m, seeds, lab, P.schedule(load_stats(), tm), dev).cpu()
        z = sample(m, seeds, sched, dev, [0.0] * 5)["x"]
        res["c1_zero_equals_faithful_dns_bitwise"] = bool(torch.equal(ref, z))
        res["c6_repeat_bitwise"] = bool(torch.equal(z, sample(m, seeds, sched, dev, None)["x"]))
        cc = P.CallCounter(m)
        sample(m, seeds[:1], sched, dev, [0.005, 0, 0, 0, 0])
        res["c2_denoiser_calls_per_image"] = cc.n
        cc.remove()
        x = torch.randn(3, 4, 32, 32, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
        Fm = torch.fft.fft2(x) * MASKS[0]
        y = torch.fft.ifft2(Fm)
        px = y.real
        comp = sum(bands(x)[1:])
        res["c3_imag_rel"] = float(y.imag.norm() / y.real.norm())
        res["c3_idempotent_rel"] = float((p_low(px) - px).norm() / px.norm())
        res["c3_orthogonal_rel"] = float(abs((px * (x - px)).sum()) / (x.norm() ** 2))
        mk = MASKS[0]
        res["c3_mask_conjugate_symmetric"] = bool(torch.equal(mk, torch.roll(torch.flip(mk, (0, 1)), (1, 1), (0, 1))))
        res["c3_low_plus_complement_rel"] = float((px + comp - x).norm() / x.norm())
        # c4: the impulse changes the state after update 9 and leaves the prediction at step 9 unchanged
        tr0 = sample(m, seeds, sched, dev, None, record=True)
        tr1 = sample(m, seeds, sched, dev, [0.005, 0, 0, 0, 0], record=True)
        res["c4_state_after_9_changed"] = bool(not torch.equal(tr0["ck"][:, 0], tr1["ck"][:, 0]))
        res["c4_state_change_equals_impulse_rel"] = float(((tr1["ck"][:, 0] - tr0["ck"][:, 0]) - 0.005 * p_low(tr0["ck"][:, 0])).norm() / (0.005 * p_low(tr0["ck"][:, 0])).norm())
        fpt = fp_traj(DEV[:32], dev)
        q = sample(m, DEV[:32], sched, dev, None, record=True)
        ci = cross_identity(q["ck"], fpt["ck"])
        res["c7_moment_identity_max_rel"] = max(c["closure_max_rel"] for c in ci)
        res["c8_repeat_variation_latent"] = 0.0 if res["c6_repeat_bitwise"] else float((z - sample(m, seeds, sched, dev, None)["x"]).norm() / z.norm())
        res.update(run.record())
    res["pass"] = (res["c1_zero_equals_faithful_dns_bitwise"] and res["c2_denoiser_calls_per_image"] == 50 and res["c3_imag_rel"] < 1e-6
                   and res["c3_idempotent_rel"] < 1e-6 and res["c3_orthogonal_rel"] < 1e-6 and res["c3_mask_conjugate_symmetric"]
                   and res["c3_low_plus_complement_rel"] < 1e-6 and res["c4_state_after_9_changed"] and res["c4_state_change_equals_impulse_rel"] < 1e-6
                   and res["c5_schedule_identical_to_faithful_dns"] and res["c6_repeat_bitwise"] and res["c7_moment_identity_max_rel"] < 1e-6)
    jdump("correctness.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if not res["pass"]:
        raise SystemExit(6)


def stage_components(dev):
    if not guard("components", 0.3):
        raise SystemExit(5)
    out = {}
    with Run(STAGE, "r1_components", dev, {}) as run:
        m, _ = heig(dev)
        for setname, seeds in (("fit", FIT), ("dev", DEV)):
            fpt = fp_traj(seeds, dev)
            fpm = moments(fpt["ck"])
            fl = floors_from(fpm)
            out[setname] = {}
            for mode in ("plain", "affine", "sched", "affsched", "dns"):
                r = sample(m, seeds, variant(mode), dev, None, record=True)
                qm = moments(r["ck"])
                mm, act = residual(qm, fpm, fl)
                lat = ((r["x"] - fpt["x"]).flatten(1).pow(2).sum(1) / fpt["x"].flatten(1).pow(2).sum(1))
                out[setname][mode] = {"latent": float(lat.mean()), "J_full": J_full(mm), "rms": r["rms"], "dc_mean": qm["dc_mean"].tolist(), "V": qm["V"].tolist(),
                                      "cross": cross_identity(r["ck"], fpt["ck"])}
                print(f"  {setname} {mode}: latent {out[setname][mode]['latent']:.5f} J {out[setname][mode]['J_full']:.4f}", flush=True)
            out[setname]["FP"] = {"dc_mean": fpm["dc_mean"].tolist(), "dc_sd": fpm["dc_sd"].tolist(), "V": fpm["V"].tolist(), "floors": fl}
        out.update(run.record())
    out["note"] = "component removals of faithful DNS's frozen calibration (not independently optimized algorithms); context only"
    jdump("components.json", out)


def stage_fit(dev):
    """response matrices (±0.002, check ±0.004), L and T fits, acceptance with backtracking, one refresh each; all on FIT."""
    if not guard("fit", 1.2):
        raise SystemExit(5)
    rec = jload("fit.json") or {}
    with Run(STAGE, "r1_fit", dev, {}) as run:
        ctx = Ctx(dev)
        fpt = fp_traj(FIT, dev)
        fpm = moments(fpt["ck"])
        fl = floors_from(fpm)
        rec["floors"] = fl
        base = ctx.roll(FIT, [0.0] * 5, fpt, fl)
        rec["dns_fit"] = {"latent": base["latent"], "J_full": J_full(base["m"]), "J_local": J_local(base["m"]), "m": base["m"].tolist(), "floor_active": base["floor_active"]}
        rep = sample(ctx.m, FIT[:32], ctx.sched, dev, None)["x"]
        rec["repeat_variation"] = 0.0 if torch.equal(rep, base["x"][:32]) else float((rep - base["x"][:32]).norm() / base["x"][:32].norm())

        def measure(theta_c, tag):
            G1 = response(ctx, theta_c, fpt, fl, H1)
            G2 = response(ctx, theta_c, fpt, fl, H2)
            rel = [float((G1[:, j] - G2[:, j]).norm() / G2[:, j].norm().clamp_min(1e-30)) for j in range(5)]
            usable = [rel[j] <= 0.25 and float(G1[:, j].norm()) > 0 for j in range(5)]
            sv = torch.linalg.svdvals(G1).tolist()
            info = {"G": G1.tolist(), "G_check": G2.tolist(), "col_norm": G1.norm(dim=0).tolist(), "rel_diff_h1_h2": rel, "usable": usable,
                    "singular_values": sv, "condition": sv[0] / max(sv[-1], 1e-30),
                    "offtime_block_norm": [float(torch.cat([G1[8 * c:8 * c + 8, j] for c in range(6) if c != j]).norm()) for j in range(5)]}
            atomic_torch_save({"G": G1, "G_check": G2, "theta_c": theta_c}, os.path.join(CK, f"G_{tag}.pt"))
            return G1, usable, info

        G0, usable0, rec["G0"] = measure([0.0] * 5, "zero")
        arms = {}
        for arm, local in (("L", True), ("T", False)):
            Jf = J_local if local else J_full
            its = [{"theta": [0.0] * 5, "J_arm": Jf(base["m"]), "latent": base["latent"], "accepted": True, "it": 0}]
            theta_c, mc, Gc, usable = [0.0] * 5, base["m"], G0, usable0
            for it in (1, 2):
                if it == 2:
                    Gc, usable, info = measure(theta_c, f"{arm}_it1")
                    rec[f"G_{arm}_refresh"] = info
                    mc = ctx.roll(FIT, theta_c, fpt, fl)["m"]
                th, sol = solve(mc, Gc, theta_c, usable, local)
                prev = its[-1]
                attempts, acc = [], None
                for scale in (1.0, 0.5, 0.25):
                    cand = [theta_c[j] + scale * (th[j] - theta_c[j]) for j in range(5)]
                    try:
                        r = ctx.roll(FIT, cand, fpt, fl)
                        ok = Jf(r["m"]) < prev["J_arm"] and r["latent"] <= 1.25 * base["latent"] and max(r["rms"]) <= 2 * max(base["rms"])
                        attempts.append({"scale": scale, "theta": cand, "J_arm": Jf(r["m"]), "J_full": J_full(r["m"]), "latent": r["latent"], "accepted": ok})
                    except FloatingPointError as e:
                        ok = False
                        attempts.append({"scale": scale, "theta": cand, "error": str(e), "accepted": False})
                    if ok:
                        acc = attempts[-1]
                        break
                entry = {"it": it, "solve": sol, "theta_solved": th, "attempts": attempts}
                if acc:
                    theta_c = acc["theta"]
                    entry.update({"theta": theta_c, "J_arm": acc["J_arm"], "latent": acc["latent"], "accepted": True,
                                  "pred_J_arm": sol["pred_obj"], "actual_J_arm": acc["J_arm"]})
                else:
                    entry.update({"theta": theta_c, "J_arm": prev["J_arm"], "latent": prev["latent"], "accepted": False})
                its.append(entry)
                print(f"  {arm} it{it}: theta {np.round(theta_c, 5).tolist()} J_arm {entry['J_arm']:.4f} (prev {prev['J_arm']:.4f}) accepted {entry['accepted']}", flush=True)
                jdump("fit.json", {**rec, "arms": {**arms, arm: its}})
            arms[arm] = its
        rec["arms"] = arms
        rec["calibration_rollouts"] = len(ctx.cache)
        rec.update(run.record())
    jdump("fit.json", rec)


def stage_select(dev):
    """development: each arm picks among its zero / first / second iterates by the COMMON full objective, with improvement over zero and the
    1.25x fidelity guard; ties -> smaller norm."""
    if not guard("select", 0.25):
        raise SystemExit(5)
    fit = jload("fit.json")
    res = {"arms": {}}
    with Run(STAGE, "r1_select", dev, {}) as run:
        ctx = Ctx(dev)
        fpt = fp_traj(DEV, dev)
        fl = fit["floors"]
        z = ctx.roll(DEV, [0.0] * 5, fpt, fl)
        res["dns_dev"] = {"latent": z["latent"], "J_full": J_full(z["m"])}
        for arm, its in fit["arms"].items():
            cands = []
            seen = set()
            for e in its:
                t = tuple(np.round(e["theta"], 9))
                if t in seen:
                    continue
                seen.add(t)
                r = ctx.roll(DEV, list(t), fpt, fl)
                cands.append({"it": e["it"], "theta": list(t), "J_full_dev": J_full(r["m"]), "latent_dev": r["latent"], "norm": float(np.linalg.norm(t))})
            ok = [c for c in cands if c["it"] == 0 or (c["J_full_dev"] < res["dns_dev"]["J_full"] and c["latent_dev"] <= 1.25 * z["latent"])]
            best = min(ok, key=lambda c: (round(c["J_full_dev"], 10), c["norm"]))
            res["arms"][arm] = {"candidates": cands, "selected": best, "identity_selected": best["it"] == 0 or best["norm"] == 0}
            print(f"  {arm}: selected it{best['it']} theta {np.round(best['theta'], 5).tolist()} J_dev {best['J_full_dev']:.4f} (DNS {res['dns_dev']['J_full']:.4f})", flush=True)
        res.update(run.record())
    jdump("parameters.json", {"L": res["arms"]["L"]["selected"]["theta"], "T": res["arms"]["T"]["selected"]["theta"], "insert_after_updates": INSERT,
                              "projector": "objective.d2.band_masks(32)[0] (lowest radial band incl. DC), shared over channels", "frozen_utc": datetime.now(timezone.utc).isoformat()})
    jdump("select.json", res)


def stage_confirm(dev):
    if not guard("confirm", 0.4):
        raise SystemExit(5)
    import lpips
    prm = jload("parameters.json")
    arms = {"plain": ("plain", None), "dns": ("dns", None), "L": ("dns", prm["L"]), "T": ("dns", prm["T"])}
    if prm["L"] == prm["T"]:
        arms["T"] = ("same_as_L", None)
    res = {"seeds": [CONF[0], CONF[-1]], "arms": {}}
    with Run(STAGE, "r1_confirm", dev, {"arms": list(arms)}) as run:
        m, _ = heig(dev)
        fpt = fp_traj(CONF, dev)
        fpm = moments(fpt["ck"])
        fl = jload("fit.json")["floors"]
        vae = load_vae(dev)
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        imf = torch.cat([decode(vae, fpt["x"][b:b + 32].to(dev)) for b in range(0, 128, 32)])
        lats = {"fp": fpt["x"]}
        for name, (mode, th) in arms.items():
            if mode == "same_as_L":
                res["arms"][name] = {"same_as": "L"}
                continue
            torch.cuda.synchronize()
            t0 = time.time()
            r = sample(m, CONF, variant(mode), dev, th, record=True)
            torch.cuda.synchronize()
            sec = (time.time() - t0) / len(CONF)
            im = torch.cat([decode(vae, r["x"][b:b + 32].to(dev)) for b in range(0, 128, 32)])
            ps, lp = [], []
            for i in range(128):
                a, b = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
            qm = moments(r["ck"])
            mm, act = residual(qm, fpm, fl)
            lat = ((r["x"] - fpt["x"]).flatten(1).pow(2).sum(1) / fpt["x"].flatten(1).pow(2).sum(1))
            res["arms"][name] = {"rel_mse": lat.tolist(), "psnr": ps, "lpips": lp, "J_full": J_full(mm), "m": mm.tolist(), "rms": r["rms"],
                                 "sec_per_image_incl_record": sec, "theta": th, "cross": cross_identity(r["ck"], fpt["ck"])}
            lats[name] = r["x"]
            print(f"  {name}: latent {lat.mean():.5f} PSNR {np.mean(ps):.2f} LPIPS {np.mean(lp):.4f} J {J_full(mm):.4f}", flush=True)
        atomic_torch_save(lats, os.path.join(CK, "confirm_latents.pt"))
        res.update(run.record())
    res["costs"] = {"coefficient_bytes": 5 * 4, "denoiser_calls": 50, "projections_per_image": sum(1 for v in (prm["T"] or []) if v)}
    jdump("confirm.json", res)


@torch.no_grad()
def inception_feats(imgs_nhwc, dev, bs=200):
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    fs = []
    for i in range(0, len(imgs_nhwc), bs):
        x = torch.as_tensor(np.asarray(imgs_nhwc[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255
        fs.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(fs).numpy()


def ref_feats(dev):
    f = os.path.join(CK, "ref_feats_adm10k.npy")
    if not os.path.exists(f):
        arr = np.load(os.path.join(CKPT_DIR, "VIRTUAL_imagenet256_labeled.npz"))["arr_0"]
        np.save(f, inception_feats(arr, dev).astype(np.float32))
    return np.load(f).astype(np.float64)


def kid(a, b, n_sub=100, m=1000, seed=0):
    """unbiased MMD^2, polynomial kernel (x.y/d + 1)^3, mean over subsets (standard KID)."""
    rng = np.random.default_rng(seed)
    d = a.shape[1]
    vals = []
    for _ in range(n_sub):
        x = a[rng.choice(len(a), m, replace=False)]
        y = b[rng.choice(len(b), m, replace=False)]
        kxx, kyy, kxy = (x @ x.T / d + 1) ** 3, (y @ y.T / d + 1) ** 3, (x @ y.T / d + 1) ** 3
        vals.append((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (m * (m - 1)) - 2 * kxy.mean())
    return float(np.mean(vals)), float(np.std(vals))


def stage_fid(dev, arm):
    if not guard(f"fid_{arm}", 0.45):
        raise SystemExit(5)
    prm = jload("parameters.json")
    mode, th = {"dns": ("dns", None), "L": ("dns", prm["L"]), "T": ("dns", prm["T"]), "plain": ("plain", None)}[arm]
    dup = "dns" if (arm in ("L", "T") and not any(th)) else ("L" if arm == "T" and th == prm["L"] else None)
    if dup:                                                                   # identical candidates are deduplicated, not regenerated
        jdump(f"fid_{arm}.json", {"arm": arm, "same_as": dup, "theta": th, **{k: v for k, v in (jload(f"fid_{dup}.json") or {}).items() if k in ("fid2000", "kid", "kid_subset_sd", "n")}})
        print(f"  {arm}: identical to {dup}, not regenerated", flush=True)
        return
    os.makedirs(os.path.join(CK, "fid"), exist_ok=True)
    with Run(STAGE, f"r1_fid2000_{arm}", dev, {"arm": arm, "theta": th}) as run:
        m, _ = heig(dev)
        part = os.path.join(CK, "fid", f"lat_{arm}.partial")
        done = torch.load(part) if os.path.exists(part) else {}
        for b0 in range(0, len(GEN), 32):
            if b0 in done:
                continue
            done[b0] = sample(m, GEN[b0:b0 + 32], variant(mode), dev, th)["x"]
            if (b0 // 32) % 8 == 0:
                atomic_torch_save(done, part)
        x = torch.cat([done[b] for b in sorted(done)])
        atomic_torch_save(x, os.path.join(CK, "fid", f"latents_{arm}.pt"))
        del m
        torch.cuda.empty_cache()
        vae = load_vae(dev)
        imgs = torch.cat([decode(vae, x[b:b + 100].to(dev)) for b in range(0, len(x), 100)]).permute(0, 2, 3, 1).numpy()
        del vae
        F = inception_feats(imgs, dev)
        np.save(os.path.join(CK, "fid", f"feats_{arm}.npy"), F.astype(np.float32))
        from boa_dit.eval_fid import REF_STATS, fid_from_stats
        ref = np.load(REF_STATS)
        fid = fid_from_stats(F.mean(0), np.cov(F, rowvar=False), ref["mu"], ref["sigma"])
        kv, ks = kid(F, ref_feats(dev))
        rec = {"arm": arm, "n": len(F), "fid2000": fid, "kid": kv, "kid_subset_sd": ks, "theta": th, **run.record()}
    if os.path.exists(part):
        os.remove(part)
    jdump(f"fid_{arm}.json", rec)
    print(f"  FID-2k {arm}: {fid:.3f}  KID {kv:.5f} (subset sd {ks:.5f})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(CK, exist_ok=True)
    s = a.stage
    outs = {"checks": "correctness.json", "components": "components.json", "select": "parameters.json", "confirm": "confirm.json"}
    if s in outs and jload(outs[s]) is not None:
        print(f"  {s}: done, skipped", flush=True)
        return
    if s == "fit" and jload("fit.json") and "run" in jload("fit.json"):
        print("  fit: done, skipped", flush=True)
        return
    if s.startswith("fid_") and jload(f"{s}.json"):
        print(f"  {s}: done, skipped", flush=True)
        return
    if s == "report":
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "make_dns_response.py")], check=True)
        return
    {"checks": stage_checks, "components": stage_components, "fit": stage_fit, "select": stage_select, "confirm": stage_confirm}.get(
        s, lambda d: stage_fid(d, s[4:]))(dev)


if __name__ == "__main__":
    main()
