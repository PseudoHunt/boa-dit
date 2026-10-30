"""Anchor-drift calibration of the rank-8 branch (results/anchor_drift/PROTOCOL.md). Stage tag ad1.
Trains the 112 carries (dit_joint / followup.carry_opt machinery, unchanged) against the anchored error drift
    mean_t || [Q(x_t) - Q(x_a)] - [F(x_t) - F(x_a)] ||^2 / loc[a],  t in (a, a + 4), gradients through BOTH cheap calls,
on states from the current student's own anchored (K = 4, reuse) trajectory; evaluates with the aa1 sampler.
Stages: checks | train C_D | train E_D | dev | diag | conf
Run: python -m boa_dit.anchor_drift.run <stage> [arm]
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
from torch.utils.checkpoint import checkpoint

from boa_dit.activation_anchor.run import EXPECT
from boa_dit.activation_anchor.run import sample as aa_sample
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as DJ
from boa_dit.followup.carry_opt import Carry, Sched, ddim_step, guided_eps
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import ratio_ci
from boa_dit.weight_anchor.shared import boot_idx, ci, contrast, summary

STAGE = "ad1"
OUT = os.path.join(RESULTS, "anchor_drift")
CK = os.path.join(CKPT_DIR, "anchor_drift")
CAP_GPU_H = 2.0
CONF_RESERVE_H = 0.40
KA, T, UPDATES = 4, 50, 200
SNAPS = (0, 50, 100, 200)
DEV, DIAG, CONF = list(range(60208, 60224)), list(range(60208, 60216)), list(range(97100, 97164))
AA1DEV, G1 = list(range(60200, 60216)), list(range(61000, 61032))
BS = 32
TRAIN_ARMS = {"C_D": False, "E_D": True}                                       # arm -> feedback in training
CARRY = {("0", False): DJ.BASE_CARRY, ("0", True): DJ.BASE_CARRY,
         ("L", False): os.path.join(DJ.CK, "carry_C_L_u200.pt"), ("L", True): os.path.join(DJ.CK, "carry_E_L_u200.pt"),
         ("D", False): os.path.join(CK, "carry_C_D_u200.pt"), ("D", True): os.path.join(CK, "carry_E_D_u200.pt")}
ANCH = {"C0": ("0", False), "AC0": ("0", True), "C.L": ("L", False), "AC.L": ("L", True), "C.D": ("D", False), "AC.D": ("D", True)}
PLAIN = {"Q0": ("0", False), "A0": ("0", True), "Q.L": ("L", False), "A.L": ("L", True), "Q.D": ("D", False), "A.D": ("D", True)}
HASHES = {"base_model": (DJ.BASE_STATE, "f017af877a79015fb9670504da4fb9fb51df30e1d8e94f6b4a3deedd5aec7ced"),
          "base_carry": (DJ.BASE_CARRY, "b4a537b0570b97d7f50f8f7d9230f1f6e37c157b7277f589285dbd1ee41aedd5"),
          "tables": (DJ.TABLES, "b73030de42a83255338f4a4ab92dce52febf604f5b5b935779c1f2342298dd67"),
          "smoothing": (DJ.SMOOTH, "7ee15c0c72fc18569f2e72671089b20c1293d368dc193d1af49276c1a296db26"),
          "carry_C_L_u200": (CARRY[("L", False)], "ac20161034dbae00"), "carry_E_L_u200": (CARRY[("L", True)], "77b5d0ac43b9ed3e"),
          "norm": (os.path.join(DJ.CK, "norm.pt"), "656ddff9f7e58318")}


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
    print(f"[ad1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        sys.exit(5)


def provenance():
    out = {k: {"path": p, "sha256": DJ.sha(p), "expected": h} for k, (p, h) in HASHES.items()}
    assert all(v["sha256"].startswith(v["expected"]) for v in out.values()), out
    return out


def rl2(x, y):
    return float((x - y).double().norm() / y.double().norm().clamp_min(1e-30))


# ------------------------------------------------------------------------------------------------------------------- training path
def anchored_path(m, jq, fp, x, lab, i0, i1, sch, st, d, fb):
    """no-grad anchored trajectory on the training path (explicit feedback state) from step i0 (state st, held d) to i1 (exclusive)."""
    with torch.no_grad():
        for i in range(i0, i1):
            q = DJ.call_model(m, jq, x, lab, i, sch, *(st or ()))
            st = jq.snapshot() if fb else None
            if i % KA == 0:
                f = guided_eps(fp, x, lab, i, sch)
                d, eps = q - f, f
            else:
                eps = q - d
            x = ddim_step(x, eps, i, sch)
    return x, st, d


def collect(m, jq, fp, xa, lab, a, sch, sta, fb):
    """the anchored window from anchor a: recorded inputs (x, state, step) and FP predictions; no gradients."""
    rec = [(xa, sta, a)]
    with torch.no_grad():
        q = DJ.call_model(m, jq, xa, lab, a, sch, *(sta or ()))
        st = jq.snapshot() if fb else None
        fa = guided_eps(fp, xa, lab, a, sch)
        d, Fs = q - fa, [fa]
        x = ddim_step(xa, fa, a, sch)
        for t in range(a + 1, min(a + KA, T)):
            rec.append((x, st, t))
            q = DJ.call_model(m, jq, x, lab, t, sch, *(st or ()))
            st = jq.snapshot() if fb else None
            Fs.append(guided_eps(fp, x, lab, t, sch))
            x = ddim_step(x, q - d, t, sch)
    return rec, Fs


def drift_loss(m, jq, rec, Fs, lab, sch, nrm, detach_anchor=False, ckpt=True):
    """mean over window steps of ||(Q_t - Q_a) - (F_t - F_a)||^2 (sum over channels/pixels, mean over images) / loc[a]; gradients through Q_t and Q_a."""
    def call(x, i, st):
        if ckpt:
            return checkpoint(DJ.call_model, m, jq, x, lab, i, sch, *(st or ()), use_reentrant=False)
        return DJ.call_model(m, jq, x, lab, i, sch, *(st or ()))
    xa, sta, a = rec[0]
    Qa = call(xa, a, sta)
    if detach_anchor:
        Qa = Qa.detach()
    terms, loc = [], []
    for (x, st, t), Ft in zip(rec[1:], Fs[1:]):
        Qt = call(x, t, st)
        terms.append(((Qt - Qa) - (Ft - Fs[0])).pow(2).flatten(1).sum(1).mean())
        loc.append(float((Qt.detach() - Ft).pow(2).flatten(1).sum(1).mean()))
    loss = torch.stack(terms).mean() / nrm["loc"][a]
    diag = {"a": a, "window": len(terms), "drift_raw": float(torch.stack(terms).mean()), "local_window_raw": float(np.mean(loc)),
            "local_anchor_raw": float((Qa.detach() - Fs[0]).pow(2).flatten(1).sum(1).mean())}
    return loss, diag


def trainer(dev, fb):
    d = make_diffusion(T)
    sch = Sched(d, dev)
    nrm = DJ.norm_consts(dev)
    m = DJ.common_base_nocarry(dev)
    jq = DJ.JointQuant(m, dev, "fb" if fb else "static")
    jq.auto = False
    c = Carry(m, torch.load(DJ.BASE_CARRY, map_location="cpu"), dev)
    fp = DJ.fp_model(dev)
    return d, sch, nrm, m, jq, c, fp


def draw(plan, u, dev):
    seeds_i, i0 = plan[u]
    seeds = [DJ.TRAIN[j] for j in seeds_i]
    return noise_for_seeds(seeds, dev), torch.as_tensor(labels_for_seeds(seeds), device=dev), KA * (i0 // KA)


def train(dev, arm, profile_updates=None):
    fb = TRAIN_ARMS[arm]
    d, sch, nrm, m, jq, c, fp = trainer(dev, fb)
    opt = torch.optim.Adam(c.parameters(), lr=DJ.LR)
    plan = DJ.draws(UPDATES)
    dsha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    res_f = os.path.join(CK, f"resume_{arm}.pt")
    start, log = 0, []
    if profile_updates is None and os.path.exists(res_f):
        r = torch.load(res_f, map_location="cpu")
        for p, v in zip(c.parameters(), r["params"]):
            p.data.copy_(v.to(dev))
        opt.load_state_dict(r["opt"])
        start, log = r["update"], r["log"]
        assert r["draw_sha"] == dsha
        print(f"  resumed {arm} at update {start}", flush=True)
    if start == 0 and profile_updates is None:
        atomic_torch_save(c.export(), os.path.join(CK, f"carry_{arm}_u0.pt"))
    t0 = time.time()
    n_end = profile_updates if profile_updates else UPDATES
    for u in range(start, n_end):
        x0, lab, a = draw(plan, u, dev)
        xa, sta, _ = anchored_path(m, jq, fp, x0, lab, 0, a, sch, None, None, fb)
        rec, Fs = collect(m, jq, fp, xa, lab, a, sch, sta, fb)
        loss, dg = drift_loss(m, jq, rec, Fs, lab, sch, nrm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in c.parameters())))
        opt.step()
        log.append({"update": u + 1, "loss": float(loss), **dg, "local_window_norm": dg["local_window_raw"] / float(nrm["loc"][a]), "grad_norm": gn,
                    "t": time.time() - t0})
        if (u + 1) % 10 == 0:
            rec10 = log[-10:]
            print(f"    {arm}: update {u + 1}, drift loss {np.mean([r['loss'] for r in rec10]):.4f} (local {np.mean([r['local_window_norm'] for r in rec10]):.4f}), "
                  f"grad {gn:.3g}, {time.time() - t0:.0f} s", flush=True)
        if profile_updates is None and (u + 1) in SNAPS:
            atomic_torch_save(c.export(), os.path.join(CK, f"carry_{arm}_u{u + 1}.pt"))
            atomic_torch_save({"params": [p.detach().cpu() for p in c.parameters()], "opt": opt.state_dict(), "update": u + 1, "log": log,
                               "draw_sha": dsha}, res_f)
    sec = (time.time() - t0) / max(1, n_end - start)
    out = {"arm": arm, "feedback_in_training": fb, "updates": n_end, "sec_per_update": sec, "log": log,
           "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2 ** 30, "draw_sha": dsha}
    c.remove()
    jq.remove()
    return out


# ------------------------------------------------------------------------------------------------------------------- evaluation models
class Cheap:
    """one dit_joint.student per feedback mode; the carry is swapped by removing and re-installing its hooks."""

    def __init__(self, dev, fb):
        self.m, self.jq, self.hs = DJ.student(dev, None, "fb" if fb else "static")
        self.jq.t, self.jq.auto = None, True
        self.cur = DJ.BASE_CARRY

    def use(self, path):
        if path != self.cur:
            for h in self.hs:
                h.remove()
            self.hs = install(self.m, torch.load(path, map_location="cpu"))
            self.cur = path


def eval_models(dev):
    fp = load_dit(dev)
    fp.requires_grad_(False)
    return fp, {False: Cheap(dev, False), True: Cheap(dev, True)}


def run_arm(fp, cheap, d, nz, lab, name, **kw):
    spec = ANCH.get(name) or PLAIN.get(name)
    if name == "FP50":
        return aa_sample(fp, None, None, d, nz, lab, "fp", **kw)
    ch = cheap[spec[1]]
    ch.use(CARRY[spec])
    return aa_sample(fp, ch.m, ch.jq, d, nz, lab, "corr" if name in ANCH else "plain", k=KA, **kw)


def gen(fp, cheap, d, seeds, bs, arms, t0=None, base=0.0, est_batch_h=0.0, timing=None):
    lat, cnt = {a: [] for a in arms}, {}
    for b in range(0, len(seeds), bs):
        s = seeds[b:b + bs]
        if t0 is not None and base + (time.time() - t0) / 3600 + est_batch_h > CAP_GPU_H:
            print(f"[ad1 runtime stop] before batch {s[0]}-{s[-1]}", flush=True)
            return {a: torch.cat(v) for a, v in lat.items() if v}, cnt, False
        nz, lab = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            torch.cuda.synchronize()
            ts = time.perf_counter()
            x, c = run_arm(fp, cheap, d, nz, lab, a)
            torch.cuda.synchronize()
            if timing is not None:
                timing.setdefault(a, []).append(time.perf_counter() - ts)
            kind = "fp" if a == "FP50" else ("corr" if a in ANCH else "plain")
            assert torch.isfinite(x).all() and c == EXPECT[kind], (a, c)
            lat[a].append(x.cpu())
            cnt[a] = c
            print(f"  seeds {s[0]}-{s[-1]} {a}: {time.perf_counter() - ts:.1f} s", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt, True


# ------------------------------------------------------------------------------------------------------------------- checks
def checks(dev):
    guard("checks", 0.20)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "checks", dev, {}) as run:
        ch = {"provenance": provenance()}
        d, sch, nrm, m, jq, c, fp = trainer(dev, True)
        base = torch.load(DJ.BASE_CARRY, map_location="cpu")
        ex = c.export()
        ch["D5_u0_export_equals_base"] = all(torch.equal(ex[k]["V"], base[k]["V"]) and torch.equal(ex[k]["delta"], base[k]["delta"]) for k in base)
        plan = DJ.draws(UPDATES)
        x0, lab, _ = draw(plan, 0, dev)
        # D1 anchored path save / restore
        with torch.no_grad():
            xu, stu, du = anchored_path(m, jq, fp, x0, lab, 0, 24, sch, None, None, True)
            x12, st12, d12 = anchored_path(m, jq, fp, x0, lab, 0, 12, sch, None, None, True)
        p = os.path.join(CK, "d1_state.pt")
        atomic_torch_save({"x": x12.cpu(), "st": [s.cpu() for s in st12], "d": d12.cpu()}, p)
        r = torch.load(p, map_location="cpu")
        with torch.no_grad():
            xr, str_, dr = anchored_path(m, jq, fp, r["x"].to(dev), lab, 12, 24, sch, tuple(s.to(dev) for s in r["st"]), r["d"].to(dev), True)
        ch["D1_restore_bitwise"] = bool(torch.equal(xu, xr) and torch.equal(du, dr) and all(torch.equal(a_, b_) for a_, b_ in zip(stu, str_)))
        # D2 gradient through the anchor term (draw 0, a = 12 for a full window)
        xa, sta, _ = anchored_path(m, jq, fp, x0, lab, 0, 12, sch, None, None, True)
        rec, Fs = collect(m, jq, fp, xa, lab, 12, sch, sta, True)
        grads, losses = {}, {}
        for det in (False, True):
            c.zero_grad(set_to_none=True)
            jq.calls = {k: 0 for k in DJ.KEYS}
            loss, _ = drift_loss(m, jq, rec, Fs, lab, sch, nrm, detach_anchor=det)
            loss.backward()
            grads[det] = torch.cat([q.grad.flatten() for q in c.parameters()])
            losses[det] = float(loss)
            if not det:
                ch["D4_calls_per_site_fwd_plus_recompute"] = sorted(set(jq.calls.values()))
        ga = grads[False] - grads[True]
        ch["D2_loss_full_vs_detached_equal"] = losses[False] == losses[True]
        ch["D2_anchor_grad_rel_l2"] = float(ga.norm() / grads[False].norm())
        ch["D2_pass"] = bool(torch.isfinite(ga).all() and ga.norm() > 0 and ch["D2_anchor_grad_rel_l2"] >= 1e-3)
        ch["D4_pass"] = ch["D4_calls_per_site_fwd_plus_recompute"] == [2 * len(rec)]
        # D3 checkpoint on / off (2 images)
        rec2 = [(x[:2], None if st is None else tuple(torch.cat([s_[:2], s_[4:6]]) for s_ in st), t) for x, st, t in rec]   # rows: cond 0-3, uncond 4-7
        Fs2 = [f[:2] for f in Fs]
        g3, l3 = {}, {}
        for ck in (False, True):
            c.zero_grad(set_to_none=True)
            loss, _ = drift_loss(m, jq, rec2, Fs2, lab[:2], sch, nrm, ckpt=ck)
            loss.backward()
            g3[ck], l3[ck] = torch.cat([q.grad.flatten() for q in c.parameters()]), loss.detach()
        ch["D3_forward_bitwise"] = bool(torch.equal(l3[False], l3[True]))
        ch["D3_grad_rel_l2"] = float((g3[True] - g3[False]).norm() / g3[False].norm())
        ch["D3_pass"] = ch["D3_forward_bitwise"] and ch["D3_grad_rel_l2"] <= 1e-4
        c.zero_grad(set_to_none=True)
        # D6 training-path anchored endpoint (4 images, feedback on) - reported
        with torch.no_grad():
            xt, _, _ = anchored_path(m, jq, fp, x0, lab, 0, T, sch, None, None, True)
        c.remove()
        jq.remove()
        del m, jq, c
        torch.cuda.empty_cache()
        fpm, cheap = eval_models(dev)
        de = make_diffusion(T)
        seeds0 = [DJ.TRAIN[j] for j in plan[0][0]]
        xe, _ = run_arm(fpm, cheap, de, x0, labels_for_seeds(seeds0), "AC0")
        ch["D6_training_path_vs_eval_sampler_rel_l2_AC0_4img"] = rl2(xt.cpu(), xe.cpu())
        # D5 eval sampler: base carry reproduces aa1 C / AC dev latents; carry swap reproduces joint C_L / E_L plain latents
        ref = torch.load(os.path.join(CKPT_DIR, "activation_anchor", "checks_dev_latents.pt"), map_location="cpu")
        nz, lb = noise_for_seeds(AA1DEV, dev), labels_for_seeds(AA1DEV)
        ch["D5_aa1_bitwise"] = {}
        for name, refk in (("C0", "C"), ("AC0", "AC")):
            x, _ = run_arm(fpm, cheap, de, nz, lb, name)
            ch["D5_aa1_bitwise"][name] = bool(torch.equal(x.cpu(), ref[refk]))
        L = torch.load(os.path.join(DJ.CK, "eval_latents.pt"), map_location="cpu")
        nz, lb = noise_for_seeds(G1, dev), labels_for_seeds(G1)
        ch["D5_swap_bitwise_vs_joint"] = {}
        for name, refk in (("Q.L", "C_L"), ("A.L", "E_L"), ("Q0", "A")):
            x, _ = run_arm(fpm, cheap, de, nz, lb, name)
            ch["D5_swap_bitwise_vs_joint"][name] = bool(torch.equal(x.cpu(), L[refk][:32]))
            print(f"  D5 swap {name}: {ch['D5_swap_bitwise_vs_joint'][name]}", flush=True)
        del fpm, cheap
        torch.cuda.empty_cache()
        flat = [ch["D5_u0_export_equals_base"], ch["D1_restore_bitwise"], ch["D2_pass"], ch["D3_pass"], ch["D4_pass"], *ch["D5_aa1_bitwise"].values(),
                *ch["D5_swap_bitwise_vs_joint"].values()]
        ch["pass"] = bool(all(flat))
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1), flush=True)
    if not ch["pass"]:
        print("[ad1] checks FAILED: stop", flush=True)
        sys.exit(6)
    with Run(STAGE, "profile", dev, {}) as run:
        prof = {}
        for arm in TRAIN_ARMS:
            r = train(dev, arm, profile_updates=10)
            prof[arm] = {"sec_per_update": r["sec_per_update"], "peak_mem_GiB": r["peak_mem_GiB"]}
            print(f"  profile {arm}: {r['sec_per_update']:.2f} s/update, peak {r['peak_mem_GiB']:.1f} GiB", flush=True)
        prof.update(run.record())
    jdump("profile.json", prof)


def need_checks():
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        print("[ad1] checks must pass first", flush=True)
        sys.exit(6)


def stage_train(dev, arm):
    need_checks()
    prof = jload("profile.json")
    est = prof[arm]["sec_per_update"] * UPDATES / 3600 * 1.05
    r0 = torch.load(os.path.join(CK, f"resume_{arm}.pt"), map_location="cpu")["update"] if os.path.exists(os.path.join(CK, f"resume_{arm}.pt")) else 0
    guard(f"train {arm}", est * (UPDATES - r0) / UPDATES)
    with Run(STAGE, f"train_{arm}", dev, {"arm": arm, "updates": UPDATES, "lr": DJ.LR, "B": DJ.B, "K_anchor": KA}) as run:
        provenance()
        out = train(dev, arm)
        out.update(run.record())
    jdump(f"train_{arm}.json", out)


# ------------------------------------------------------------------------------------------------------------------- statistics
CO_PRIMARY = (("C.D", "C.L"), ("AC.D", "AC.L"))
SECONDARY = (("AC.D", "C.D"), ("AC.L", "C.L"), ("C.L", "C0"), ("AC.L", "AC0"), ("C.D", "C0"), ("AC.D", "AC0"), ("AC0", "C0"))
DEV_PLAIN = (("Q.D", "Q.L"), ("A.D", "A.L"), ("Q.L", "Q0"), ("A.L", "A0"), ("Q.D", "Q0"), ("A.D", "A0"))


def reading(c):
    m, lp = c["rel_mse"], c["lpips"]
    if m["ratio"] <= 0.95 and m["ratio_ci"][1] < 1 and not lp["diff_ci"][0] > 0:
        return "useful (>= 5% lower mean latent MSE, CI excludes no improvement, no clear LPIPS regression)"
    if m["ratio_ci"][1] < 1:
        return "smaller positive" + (" (LPIPS clearly worse)" if lp["diff_ci"][0] > 0 else " (below the 5% threshold)")
    if m["ratio_ci"][0] > 1:
        return "harmful"
    return "inconclusive"


def interaction(Mx):
    cl, cd, al, ad = (np.asarray(Mx[x]["rel_mse"]) for x in ("C.L", "C.D", "AC.L", "AC.D"))
    idx = boot_idx(len(cl))
    I = lambda i: (cl[i].mean() - cd[i].mean()) - (al[i].mean() - ad[i].mean())
    al_ = np.arange(len(cl))
    return {"I_abs": float(I(al_)), "I_abs_ci": ci([I(i) for i in idx]),
            "drift_gain_fb_off": float(cl.mean() - cd.mean()), "drift_gain_fb_off_ci": ci([cl[i].mean() - cd[i].mean() for i in idx]),
            "drift_gain_fb_on": float(al.mean() - ad.mean()), "drift_gain_fb_on_ci": ci([al[i].mean() - ad[i].mean() for i in idx]),
            "note": "I = [E(C.L) - E(C.D)] - [E(AC.L) - E(AC.D)]; I > 0: the drift objective helps more without feedback"}


def show(Mx, C):
    for a, v in summary(Mx).items():
        print(f"  {a}: rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f} PSNR {v['psnr']['mean']:.2f} LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in C.items():
        m = v["rel_mse"]
        print(f"  {k_}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] wins {m['wins_x']}/{m['n']} "
              f"dLPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['diff_ci'][0]:+.4f}, {v['lpips']['diff_ci'][1]:+.4f}]", flush=True)


def trained():
    for arm in TRAIN_ARMS:
        if not os.path.exists(CARRY[("D", TRAIN_ARMS[arm])]):
            print(f"[ad1] {arm} not trained", flush=True)
            sys.exit(6)


# ------------------------------------------------------------------------------------------------------------------- dev / diag
@torch.no_grad()
def dev_stage(dev):
    need_checks()
    trained()
    guard("dev", 0.15)
    with Run(STAGE, "dev", dev, {"seeds": [DEV[0], DEV[-1]], "batch": 16}) as run:
        provenance()
        fp, cheap = eval_models(dev)
        d = make_diffusion(T)
        arms = ["FP50"] + list(ANCH) + list(PLAIN)
        lat, cnt, _ = gen(fp, cheap, d, DEV, 16, arms)
        atomic_torch_save(lat, os.path.join(CK, "dev_latents.pt"))
        del fp, cheap
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in CO_PRIMARY + SECONDARY + DEV_PLAIN}
        e = lambda a: float(np.mean(Mx[a]["rel_mse"]))
        div = {"C.D_over_C.L": e("C.D") / e("C.L"), "AC.D_over_AC.L": e("AC.D") / e("AC.L")}
        div["guard_tripped"] = bool(div["C.D_over_C.L"] > 2 or div["AC.D_over_AC.L"] > 2)
        out = {"seeds": [DEV[0], DEV[-1]], "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C, "interaction": interaction(Mx),
               "divergence_guard": div, "note": "development (inspected seeds), exploratory; nothing selected", **run.record()}
    jdump("dev.json", out)
    show(Mx, C)
    print("  divergence guard:", div, flush=True)


@torch.no_grad()
def diag(dev):
    need_checks()
    trained()
    guard("diag", 0.08)
    with Run(STAGE, "diag", dev, {"seeds": [DIAG[0], DIAG[-1]]}) as run:
        fp, cheap = eval_models(dev)
        d = make_diffusion(T)
        nz, lab = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        sq = lambda X: X.flatten(2).pow(2).sum(2)
        out = {"seeds": [DIAG[0], DIAG[-1]], "arms": {},
               "note": "own anchored trajectories; raw e_t = f_cheap - f_FP (channels 0-3) against same-state FP calls (diagnostic only); "
                       "corrected = used prediction - f_FP; held-out local energy = sum ||e_t||^2, drift energy = corrected energy at non-anchor steps"}
        for name in ANCH:
            D = {"raw": [], "used": [], "anchor": []}
            x, c = run_arm(fp, cheap, d, nz, lab, name, diag=D)
            raw, used = torch.stack(D["raw"], 1).double(), torch.stack(D["used"], 1).double()
            na = ~torch.tensor(D["anchor"])
            t_idx = torch.arange(T)[na]
            a_idx = t_idx - t_idx % KA
            Et, Ea = sq(raw[:, t_idx]), sq(raw[:, a_idx])
            X = (raw[:, t_idx] * raw[:, a_idx]).flatten(2).sum(2)
            pred = sq(raw[:, t_idx] - raw[:, a_idx])
            r = {"local_energy_all_steps": float(sq(raw).sum()), "local_energy_nonanchor": float(Et.sum()), "anchor_energy_paired": float(Ea.sum()),
                 "drift_energy_nonanchor": float(sq(used[:, na]).sum()), "window_cross": float(X.sum()),
                 "window_uncentered_corr": float(X.sum() / (Et.sum() * Ea.sum()).sqrt()),
                 "drift_over_local_nonanchor": ratio_ci(sq(used[:, na]).sum(1).numpy(), Et.sum(1).numpy()),
                 "anchor_identity_rel_residual": float((sq(used[:, na]) - pred).abs().sum() / pred.sum()),
                 "per_step_local": sq(raw).mean(0).tolist(), "per_step_used": sq(used).mean(0).tolist()}
            out["arms"][name] = r
            print(f"  {name}: local {r['local_energy_nonanchor']:.4g} drift {r['drift_energy_nonanchor']:.4g} (ratio {r['drift_over_local_nonanchor']['v']:.3f}) "
                  f"rho {r['window_uncentered_corr']:.3f}", flush=True)
        out.update(run.record())
    jdump("diag.json", out)


# ------------------------------------------------------------------------------------------------------------------- conf
@torch.no_grad()
def conf(dev):
    fr = jload("frozen.json")
    if not (fr and jload("dev.json")) or fr.get("stop"):
        print("[ad1] committed frozen.json required (not stopped)", flush=True)
        sys.exit(6)
    guard("conf", 0.35, reserve_h=0.0)
    t0, base = time.time(), spent()
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "batch": BS}) as run:
        provenance()
        assert all(DJ.sha(CARRY[ANCH[a]]) == fr["carry_sha256"][a] for a in ANCH)
        fp, cheap = eval_models(dev)
        d = make_diffusion(T)
        timing = {}
        torch.cuda.reset_peak_memory_stats()
        lat, cnt, full = gen(fp, cheap, d, CONF, BS, ["FP50"] + list(ANCH), t0=t0, base=base, est_batch_h=0.15, timing=timing)
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt" if full else "conf_latents_partial.pt"))
        if not full:
            jdump("conf_partial.json", {"calls": cnt, **run.record()})
            sys.exit(5)
        del fp, cheap
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in CO_PRIMARY + SECONDARY}
        out = {"seeds": [CONF[0], CONF[-1]], "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C, "interaction": interaction(Mx),
               "reading_drift_vs_local_fb_off": reading(C["C.D_vs_C.L"]), "reading_drift_vs_local_fb_on": reading(C["AC.D_vs_AC.L"]),
               "reading_feedback_after_drift": reading(C["AC.D_vs_C.D"]), "reading_feedback_after_local": reading(C["AC.L_vs_C.L"]),
               "timing_s_per_batch32": {a: {"median": float(np.median(v)), "all": v} for a, v in timing.items()},
               "timing_note": "CUDA-synchronized wall per trajectory, FP + two cheap instances resident; simulation (fake-quant)", "peak_mem": peak, **run.record()}
    jdump("conf.json", out)
    show(Mx, C)
    print("  interaction:", json.dumps(out["interaction"]), flush=True)
    for k in [x for x in out if x.startswith("reading")]:
        print(f"  {k}: {out[k]}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "train", "dev", "diag", "conf"])
    ap.add_argument("arm", nargs="?", choices=list(TRAIN_ARMS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if a.cmd == "train":
        stage_train(dev, a.arm)
    else:
        {"checks": checks, "dev": dev_stage, "diag": diag, "conf": conf}[a.cmd](dev)


if __name__ == "__main__":
    main()
