"""dar1: anchor-aware refit of the DiT rank-8 branch (results/anchor_refit/PROTOCOL.md). Ledger tag dar1 (common handoff budget: scripts/h1031_budget.py).
Inference = abn1 CB configuration (W4A8 static, FP anchors K4, bridge lambda 1). Trains only the 112 carries (V, delta) with
  BA: two-K4-interval corrected window from cached B0 corrected-trajectory anchor states, current-student anchor measurements, teacher evaluated
      at the student's latent (frozen weights, latent derivative kept), loss ||x_end - x_end^FP||^2 / den[a];
  BL: per-step prediction reconstruction ||Q(x_i^FP) - F(x_i^FP)||^2 / loc[i] over the same window transitions and draws.
Stages: cache | norm | checks | profile | train_ba | train_bl | dev | conf | diag
Run: python -m boa_dit.anchor_refit.run <stage>
"""
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import h1031_budget as HB
from boa_dit.activation_bridge import run as AB
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as DJ
from boa_dit.followup.carry_opt import Carry, Sched, ddim_step, guided_eps
from boa_dit.model import NULL_CLASS, labels_for_seeds, make_diffusion, noise_for_seeds
from boa_dit.weight_anchor.run import guided, rel
from boa_dit.weight_anchor.shared import summary

TAG = "dar1"
OUT = os.path.join(RESULTS, "anchor_refit")
CK = os.path.join(CKPT_DIR, "anchor_refit")
T, KA, LAM, LR, B = 50, 4, 1.0, DJ.LR, DJ.B
TRAIN = DJ.TRAIN                                                             # 60100-60163
NORM_SEEDS = TRAIN[:16]
DEV, DIAG, CONF = list(range(60200, 60232)), list(range(60208, 60216)), list(range(124000, 124128))
OVN = list(range(110000, 110032))
STARTS = list(range(0, 49, KA))                                              # 13 window starts
NMAX = 4000
ARCH = os.path.join(DJ.CK, "carry_C_L_u200.pt")
ALLOC_H = 0.35
AMP = True


def jdump(n, o):
    atomic_write_json(os.path.join(OUT, n), o)


def jload(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def guard(name, est):
    ok, info = HB.ok(TAG, est)
    b = jload("budget.json") or {"cap_dar1_h": HB.PLAN["dar1"], "total_cap_h": HB.TOTAL_CAP, "log": []}
    b["log"].append({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": name, **info, "ok": ok})
    jdump("budget.json", b)
    print(f"[dar1 budget] {name}: {info} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        jdump(f"{name}.json", {"status": "skipped (budget)", **info})
        sys.exit(4)


def draws(n):
    """fixed draw list (prefix-consistent): (4 TRAIN seed indices, window start a)."""
    g = torch.Generator().manual_seed(0)
    return [(torch.randint(0, len(TRAIN), (B,), generator=g).tolist(), STARTS[int(torch.randint(0, len(STARTS), (1,), generator=g))]) for _ in range(n)]


# ------------------------------------------------------------------------------------------------------------------- differentiable sampler pieces
def stu(m, jq, x, lab, i, sch, ck):
    def f(x_):
        jq.t = i
        jq.load(None)
        jq.state_out = {}
        return guided_eps(m, x_, lab, i, sch, amp=AMP)
    return checkpoint(f, x, use_reentrant=False) if ck else f(x)


def tea(fp, x, lab, i, sch, ck, detach_input=False):
    xi = x.detach() if detach_input else x
    f = lambda x_: guided_eps(fp, x_, lab, i, sch, amp=AMP)
    return checkpoint(f, xi, use_reentrant=False) if (ck and xi.requires_grad) else f(xi)


def window(m, jq, fp, xa, lab, a, span, sch, Ac, bc, ck=True, detach_teacher=False, trace=None):
    """corrected anchored window from the pre-anchor-operation latent at anchor a: no bridge at a (history outside the window), bridges at internal
    anchors and at the right boundary anchor a + span if it is an anchor < T. Returns (endpoint, counts)."""
    end = min(a + span, T)
    x, dprev, aprev = xa, None, None
    cnt = {"cheap": 0, "fp": 0, "bridges": 0}
    for i in range(a, end):
        if i % KA == 0:
            q, f = stu(m, jq, x, lab, i, sch, ck), tea(fp, x, lab, i, sch, ck, detach_teacher)
            cnt["cheap"] += 1; cnt["fp"] += 1
            dn = q - f
            if dprev is not None:
                x = x - LAM * AB.bridge_coef(Ac, bc, aprev, i - aprev) * (dn - dprev)
                cnt["bridges"] += 1
            if trace is not None:
                trace[i] = x.detach().clone()
            dprev, aprev, eps = dn, i, f
        else:
            eps = stu(m, jq, x, lab, i, sch, ck) - dprev
            cnt["cheap"] += 1
        x = ddim_step(x, eps, i, sch)
    if end < T and end % KA == 0 and end == a + span:
        q, f = stu(m, jq, x, lab, end, sch, ck), tea(fp, x, lab, end, sch, ck, detach_teacher)
        cnt["cheap"] += 1; cnt["fp"] += 1
        x = x - LAM * AB.bridge_coef(Ac, bc, aprev, end - aprev) * ((q - f) - dprev)
        cnt["bridges"] += 1
    return x, cnt


@torch.no_grad()
def fp_continue(fp, xa, lab, a, span, sch):
    end = min(a + span, T)
    x, xs, Fs = xa, [], []
    for i in range(a, end):
        f = guided_eps(fp, x, lab, i, sch, amp=AMP)
        xs.append(x); Fs.append(f)
        x = ddim_step(x, f, i, sch)
    return x, xs, Fs


def loss_ba(m, jq, fp, xa, lab, a, sch, Ac, bc, den, xF=None, **kw):
    if xF is None:
        xF, _, _ = fp_continue(fp, xa, lab, a, 2 * KA, sch)
    xe, cnt = window(m, jq, fp, xa, lab, a, 2 * KA, sch, Ac, bc, **kw)
    return (xe - xF).pow(2).flatten(1).sum(1).mean() / den[a], cnt


def loss_bl(m, jq, fp, xa, lab, a, sch, nrm, cont=None):
    _, xs, Fs = cont if cont is not None else fp_continue(fp, xa, lab, a, 2 * KA, sch)
    terms = [(stu(m, jq, x, lab, a + j, sch, False) - f).pow(2).flatten(1).sum(1).mean() / nrm["loc"][a + j] for j, (x, f) in enumerate(zip(xs, Fs))]
    return torch.stack(terms).mean()


# ------------------------------------------------------------------------------------------------------------------- models / cache
def trainer(dev, carry_path=None):
    d = make_diffusion(T)
    sch = Sched(d, dev)
    Ac, bc = AB.coeffs(d)
    m = DJ.common_base_nocarry(dev)
    jq = DJ.JointQuant(m, dev, "static")
    jq.auto = False
    c = Carry(m, torch.load(carry_path or DJ.BASE_CARRY, map_location="cpu"), dev)
    fp = DJ.fp_model(dev)
    return d, sch, Ac, bc, m, jq, c, fp


def eval_models(dev):
    return AB.load_models(dev, ("fp", "static"))


def arm(carry_key):
    return AB.A("static", "fp", 4, True, carry_key)


def register_carry(key, path):
    AB.CARRY[key] = path
    AB.ARMS[key] = arm(key)


@torch.no_grad()
def anchor_states(M, d, seeds, bs=32):
    """B0 (abn1 CB) corrected trajectory: latents at anchors before the anchor operations (rec hook of abn1 sample). [n, 13, 4, 32, 32]."""
    out = []
    for b0 in range(0, len(seeds), bs):
        s = seeds[b0:b0 + bs]
        rec = []
        AB.sample(M, AB.ARMS["CB"], d, noise_for_seeds(s, "cuda:0"), labels_for_seeds(s), rec=rec)
        assert len(rec) == 13
        out.append(torch.stack([r[0].cpu() for r in rec], 1))
    return torch.cat(out)


def stage_cache(dev):
    guard("cache", 0.05)
    with Run(TAG, "cache", dev, {"seeds": "TRAIN 60100-60163, DIAG 60208-60215"}) as run:
        AB.provenance()
        M = eval_models(dev)
        d = make_diffusion(T)
        os.makedirs(CK, exist_ok=True)
        tr = anchor_states(M, d, TRAIN)
        dg = anchor_states(M, d, DIAG)
        atomic_torch_save({"train": tr, "diag": dg, "train_seeds": TRAIN, "diag_seeds": DIAG, "anchors": STARTS}, os.path.join(CK, "anchor_cache.pt"))
        rec = {"train_shape": list(tr.shape), "diag_shape": list(dg.shape), "sha256": DJ.sha(os.path.join(CK, "anchor_cache.pt")), **run.record()}
    jdump("cache.json", rec)
    print(rec, flush=True)


def load_cache():
    return torch.load(os.path.join(CK, "anchor_cache.pt"), map_location="cpu")


@torch.no_grad()
def stage_norm(dev):
    guard("norm", 0.05)
    with Run(TAG, "norm", dev, {"seeds": [NORM_SEEDS[0], NORM_SEEDS[-1]]}) as run:
        d, sch, Ac, bc, m, jq, c, fp = trainer(dev)
        C = load_cache()["train"]
        lab = torch.as_tensor(labels_for_seeds(NORM_SEEDS), device=dev)
        den = torch.zeros(T)
        for a in STARTS:
            xa = C[:16, a // KA].to(dev)
            xF, _, _ = fp_continue(fp, xa, lab, a, 2 * KA, sch)
            xe, _ = window(m, jq, fp, xa, lab, a, 2 * KA, sch, Ac, bc, ck=False)
            den[a] = (xe - xF).pow(2).flatten(1).sum(1).mean()
        floor = 1e-6 * float(den[STARTS].mean())
        den = den.clamp_min(floor)
        atomic_torch_save({"den": den, "floor": floor}, os.path.join(CK, "den.pt"))
        rec = {"den": {a: float(den[a]) for a in STARTS}, "floor": floor, "sha256": DJ.sha(os.path.join(CK, "den.pt")), **run.record()}
    jdump("norm.json", rec)
    print(rec, flush=True)


def load_den(dev):
    return torch.load(os.path.join(CK, "den.pt"))["den"]


# ------------------------------------------------------------------------------------------------------------------- checks
def flatgrad(c):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in c.parameters()])


def stage_checks(dev):
    global AMP
    guard("checks", 0.20)
    with Run(TAG, "checks", dev, {}) as run:
        ch = {"provenance": AB.provenance(), "tolerances": {"1": "endpoint rel L2 <= 0.05, step-1 latent rel <= 1e-4 (fixed before any result)",
                                                             "2": "bitwise", "3": "bitwise", "4": "bitwise", "6": "rel err <= 1e-3 at some h, neighbour <= 1e-2"}}
        d, sch, Ac, bc, m, jq, c, fp = trainer(dev)
        den, nrm = load_den(dev), DJ.norm_consts(dev)
        cache = load_cache()
        # 1. untrained differentiable inference vs validated B0 (abn1 sample), 4 TRAIN images
        s4 = TRAIN[:4]
        nz, lb = noise_for_seeds(s4, dev), labels_for_seeds(s4)
        lab = torch.as_tensor(lb, device=dev)
        trace = {}
        with torch.no_grad():
            xt, cnt_full = window(m, jq, fp, nz.float(), lab, 0, T, sch, Ac, bc, ck=False, trace=trace)
        c.remove(); jq.remove()
        M = eval_models(dev)
        D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
        xe, ce = AB.sample(M, AB.ARMS["CB"], d, nz, lb, diag=D)
        ch["1_endpoint_rel_l2"] = rel_l2 = float((xt - xe).double().norm() / xe.double().norm())
        x1t = ddim_step(nz.float(), guided_eps(fp, nz.float(), lab, 0, sch), 0, sch)
        ch["1_step1_rel"] = float((x1t - D["x"][1]).double().norm() / D["x"][1].double().norm())
        ch["1_full_counts"] = cnt_full
        ch["1_pass"] = rel_l2 <= 0.05 and ch["1_step1_rel"] <= 1e-4 and cnt_full == {"cheap": 50, "fp": 13, "bridges": 12}
        print(f"  1: endpoint rel {rel_l2:.4f}, step1 rel {ch['1_step1_rel']:.2e}, counts {cnt_full}", flush=True)
        # 2. export / install; abn1 CB and LCB bitwise via the eval harness
        nzo, lbo = noise_for_seeds(OVN, dev), labels_for_seeds(OVN)
        L1 = torch.load(os.path.join(AB.CK, "s1_latents.pt"), map_location="cpu")
        L4 = torch.load(os.path.join(AB.CK, "s4_latents.pt"), map_location="cpu")
        mt = DJ.common_base_nocarry(dev)
        jqt = DJ.JointQuant(mt, dev, "static"); jqt.auto = False
        ct = Carry(mt, torch.load(DJ.BASE_CARRY, map_location="cpu"), dev)
        u0 = ct.export()
        os.makedirs(CK, exist_ok=True)
        atomic_torch_save(u0, os.path.join(CK, "carry_u0_check.pt"))
        base = torch.load(DJ.BASE_CARRY, map_location="cpu")
        ch["3_u0_export_equals_original"] = all(torch.equal(u0[k]["V"], base[k]["V"]) and torch.equal(u0[k]["delta"], base[k]["delta"]) for k in base)
        xs_ = cache["train"][:4, 3].to(dev)
        with torch.no_grad():
            y_tr = stu(mt, jqt, xs_, lab, 12, sch, False)
            g = torch.Generator().manual_seed(5)
            for p in ct.parameters():
                p.data.add_(1e-2 * torch.randn(p.shape, generator=g).to(dev) * p.data.abs().mean())
            pert = ct.export()
            y_pt = stu(mt, jqt, xs_, lab, 12, sch, False)
        ct.remove()
        res2 = {}
        for name, cc, ytr in (("untrained", u0, y_tr), ("perturbed", pert, y_pt)):
            hs = install(mt, cc)
            with torch.no_grad():
                yi = stu(mt, jqt, xs_, lab, 12, sch, False)
            for h in hs:
                h.remove()
            res2[name] = bool(torch.equal(yi, ytr))
        ch["2_export_install_bitwise"] = res2
        jqt.remove(); del mt
        torch.cuda.empty_cache()
        register_carry("U0", os.path.join(CK, "carry_u0_check.pt"))
        register_carry("ARCH", ARCH)
        ch["2_eval_u0_equals_abn1_CB_bs32"] = bool(torch.equal(AB.sample(M, AB.ARMS["U0"], d, nzo, lbo)[0].cpu(), L1["CB"][:32]))
        ch["2_eval_arch_equals_abn1_LCB_bs32"] = bool(torch.equal(AB.sample(M, AB.ARMS["ARCH"], d, nzo, lbo)[0].cpu(), L4["LCB"][:32]))
        ch["3_eval_calls"] = AB.sample(M, AB.ARMS["U0"], d, nz, lb)[1]
        del M
        torch.cuda.empty_cache()
        # 3b/4/5 on a fresh trainer
        d, sch, Ac, bc, m, jq, c, fp = trainer(dev)
        tabs = DJ.sha(DJ.TABLES)
        w0 = m.blocks[0].attn.qkv.weight.detach().clone()
        opt = torch.optim.Adam(c.parameters(), lr=0.0)
        xa = cache["train"][:4, 0].to(dev)
        loss, _ = loss_ba(m, jq, fp, xa, lab, 0, sch, Ac, bc, den)
        loss.backward(); opt.step()
        ch["3_lr0_step_export_equals_original"] = all(torch.equal(c.export()[k]["V"], base[k]["V"]) and torch.equal(c.export()[k]["delta"], base[k]["delta"]) for k in base)
        ch["3_tables_and_W4_unchanged"] = bool(DJ.sha(DJ.TABLES) == tabs and torch.equal(w0, m.blocks[0].attn.qkv.weight))
        c.zero_grad(set_to_none=True)
        cnts = {}
        with torch.no_grad():
            for a in STARTS:
                _, cn = window(m, jq, fp, cache["train"][:4, a // KA].to(dev), lab, a, 2 * KA, sch, Ac, bc, ck=False)
                cnts[a] = cn
        exp = lambda a: {"cheap": 8, "fp": 3, "bridges": 2} if a <= 40 else ({"cheap": 6, "fp": 2, "bridges": 1} if a == 44 else {"cheap": 2, "fp": 1, "bridges": 0})
        ch["4_window_counts"] = {a: v for a, v in cnts.items()}
        ch["4_window_counts_pass"] = all(cnts[a] == exp(a) for a in STARTS)
        with torch.no_grad():
            tr0 = {}
            window(m, jq, fp, nz.float(), lab, 0, T, sch, Ac, bc, ck=False, trace=tr0)
            w8, _ = window(m, jq, fp, nz.float(), lab, 0, 8, sch, Ac, bc, ck=False)
        ch["4_window_a0_equals_full_rollout_corrected_x8_bitwise"] = bool(torch.equal(w8, tr0[8]))
        # 5. gradients
        keys = c.keys
        i0 = next(i for i, k in enumerate(keys) if k.startswith("b0."))
        i27 = next(i for i, k in enumerate(keys) if k.startswith("b27."))
        loss, _ = loss_ba(m, jq, fp, xa, lab, 0, sch, Ac, bc, den)
        loss.backward()
        g_full = flatgrad(c).clone()
        ch["5_grad_b0_nonzero"] = bool(c.V[i0].grad.abs().sum() > 0 and c.D[i0].grad.abs().sum() > 0)
        ch["5_grad_b27_nonzero"] = bool(c.V[i27].grad.abs().sum() > 0 and c.D[i27].grad.abs().sum() > 0)
        ch["5_teacher_params_no_grad"] = all(p.grad is None for p in fp.parameters())
        c.zero_grad(set_to_none=True)
        loss2, _ = loss_ba(m, jq, fp, xa, lab, 0, sch, Ac, bc, den, detach_teacher=True)
        loss2.backward()
        g_det = flatgrad(c)
        ch["5_teacher_latent_derivative_rel_effect"] = float((g_full - g_det).norm() / g_full.norm())
        ch["5_loss_equal_with_detached_teacher_input"] = bool(float(loss) == float(loss2))
        c.zero_grad(set_to_none=True)
        lbl = loss_bl(m, jq, fp, xa, lab, 0, sch, nrm)
        lbl.backward()
        ch["5_bl_grad_finite_nonzero"] = bool(torch.isfinite(flatgrad(c)).all() and flatgrad(c).abs().sum() > 0)
        c.zero_grad(set_to_none=True)
        # 4b. training resume (2 + 2 vs 4 updates, BA)
        ch["4_resume_bitwise"] = resume_check(dev)
        # 6. finite differences: quantizer off, fp32, batch 1, window a = 0
        AMP = False
        jq.mode = "off"
        x1, l1 = cache["train"][:1, 0].to(dev), lab[:1]
        xF1, _, _ = fp_continue(fp, x1, l1, 0, 2 * KA, sch)
        L = lambda xa_=x1: (window(m, jq, fp, xa_, l1, 0, 2 * KA, sch, Ac, bc, ck=False)[0] - xF1).pow(2).sum()
        g = torch.Generator().manual_seed(9)
        dirs = [torch.randn(p.shape, generator=g).to(dev) for p in c.parameters()]
        th = math.sqrt(sum(float(p.detach().pow(2).sum()) for p in c.parameters()))
        dn = math.sqrt(sum(float(v.pow(2).sum()) for v in dirs))
        dirs = [v * th / dn for v in dirs]
        c.zero_grad(set_to_none=True)
        l0 = L(); l0.backward()
        ad = sum(float((p.grad * v).sum()) for p, v in zip(c.parameters(), dirs))
        fd = {}
        with torch.no_grad():
            for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
                for sgn in (1, -1):
                    for p, v in zip(c.parameters(), dirs):
                        p.add_(sgn * h * v)
                    if sgn == 1:
                        lp = float(L())
                    else:
                        lm = float(L())
                    for p, v in zip(c.parameters(), dirs):
                        p.sub_(sgn * h * v)
                fd[h] = {"fd": (lp - lm) / (2 * h), "rel_err": abs((lp - lm) / (2 * h) - ad) / abs(ad)}
        ch["6_param_direction"] = {"autograd": ad, "fd": fd}
        c.zero_grad(set_to_none=True)
        xv = x1.clone().requires_grad_(True)
        gx = torch.autograd.grad(L(xv), xv)[0]
        dx = torch.randn(x1.shape, generator=torch.Generator().manual_seed(10)).to(dev) * float(x1.norm()) / math.sqrt(x1.numel())
        adx = float((gx * dx).sum())
        fdx = {}
        with torch.no_grad():
            for h in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
                fdx[h] = {"fd": (float(L(x1 + h * dx)) - float(L(x1 - h * dx))) / (2 * h)}
                fdx[h]["rel_err"] = abs(fdx[h]["fd"] - adx) / abs(adx)
        ch["6_latent_direction_through_student_teacher_bridge"] = {"autograd": adx, "fd": fdx}

        def resolved(r):
            hs = sorted(r, reverse=True)
            return any(r[h]["rel_err"] <= 1e-3 and ((i > 0 and r[hs[i - 1]]["rel_err"] <= 1e-2) or (i + 1 < len(hs) and r[hs[i + 1]]["rel_err"] <= 1e-2))
                       for i, h in enumerate(hs))
        ch["6_param_resolved"] = resolved(fd)
        ch["6_latent_resolved"] = resolved(fdx)
        # STE-quantized case, reported only
        AMP = True
        jq.mode = "static"
        c.zero_grad(set_to_none=True)
        l0 = L(); l0.backward()
        ad_q = sum(float((p.grad * v).sum()) for p, v in zip(c.parameters(), dirs))
        fdq = {}
        with torch.no_grad():
            for h in (1e-1, 1e-2, 1e-3):
                for p, v in zip(c.parameters(), dirs):
                    p.add_(h * v)
                lp = float(L())
                for p, v in zip(c.parameters(), dirs):
                    p.sub_(2 * h * v)
                lm = float(L())
                for p, v in zip(c.parameters(), dirs):
                    p.add_(h * v)
                fdq[h] = {"fd": (lp - lm) / (2 * h), "rel_vs_ste": abs((lp - lm) / (2 * h) - ad_q) / abs(ad_q)}
        ch["6_ste_quantized_fp16_reported_only"] = {"ste_autograd": ad_q, "fd": fdq, "note": "STE gradient through the discrete static quantizer; not exact"}
        c.zero_grad(set_to_none=True)
        flat = [ch["1_pass"], *res2.values(), ch["2_eval_u0_equals_abn1_CB_bs32"], ch["2_eval_arch_equals_abn1_LCB_bs32"], ch["3_u0_export_equals_original"],
                ch["3_eval_calls"] == {"cheap": 50, "high": 13, "corrections": 12}, ch["3_lr0_step_export_equals_original"], ch["3_tables_and_W4_unchanged"],
                ch["4_window_counts_pass"], ch["4_window_a0_equals_full_rollout_corrected_x8_bitwise"], ch["5_grad_b0_nonzero"], ch["5_grad_b27_nonzero"],
                ch["5_teacher_params_no_grad"], ch["5_teacher_latent_derivative_rel_effect"] > 0, ch["5_loss_equal_with_detached_teacher_input"],
                ch["5_bl_grad_finite_nonzero"], ch["6_param_resolved"], ch["6_latent_resolved"]]
        ch["4_resume_note"] = "reported; bitwise expected"
        ch["pass"] = bool(all(flat))
        ch["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1, default=str), flush=True)
    return 0 if ch["pass"] else 6


def resume_check(dev):
    p1 = train(dev, "BA", n_updates=4, save=False, return_params=True)
    p2a = train(dev, "BA", n_updates=2, save=False, return_params=True, keep_state=True)
    p2 = train(dev, "BA", n_updates=4, save=False, return_params=True, resume_state=p2a["state"])
    return bool(all(torch.equal(a, b) for a, b in zip(p1["params"], p2["params"])))


# ------------------------------------------------------------------------------------------------------------------- training
def train(dev, obj, n_updates, ckpt_at=(), save=True, time_cap_s=None, return_params=False, keep_state=False, resume_state=None, tag=None):
    """obj BA | BL. Updates u = start..n_updates-1 on the fixed draw list; BL with time_cap_s continues past n_updates until its training wall time
    reaches time_cap_s (BL-tm). Checkpoints exported at ckpt_at (and BL-tm)."""
    d, sch, Ac, bc, m, jq, c, fp = trainer(dev)
    den, nrm = load_den(dev), DJ.norm_consts(dev)
    C = load_cache()["train"]
    opt = torch.optim.Adam(c.parameters(), lr=LR)
    plan = draws(NMAX)
    dsha = hashlib.sha256(json.dumps(plan).encode()).hexdigest()
    tag = tag or obj
    res_f = os.path.join(CK, f"resume_{tag}.pt")
    start, log, ttrain = 0, [], 0.0
    st = resume_state
    if st is None and save and os.path.exists(res_f):
        st = torch.load(res_f, map_location="cpu")
    if st is not None:
        for p, v in zip(c.parameters(), st["params"]):
            p.data.copy_(v.to(dev))
        opt.load_state_dict(st["opt"])
        start, log, ttrain = st["update"], st["log"], st["ttrain"]
        assert st["draw_sha"] == dsha
        print(f"  resumed {tag} at update {start}", flush=True)
    if save and start == 0:
        atomic_torch_save(c.export(), os.path.join(CK, f"carry_{tag}_u0.pt"))
    u = start

    def snap():
        return {"params": [p.detach().cpu().clone() for p in c.parameters()], "opt": opt.state_dict(), "update": u, "log": log, "ttrain": ttrain,
                "draw_sha": dsha}
    while True:
        if time_cap_s is None and u >= n_updates:
            break
        if time_cap_s is not None and u >= n_updates and ttrain >= time_cap_s:
            break
        if u >= NMAX:
            break
        idx, a = plan[u]
        xa = C[idx, a // KA].to(dev)
        lab = torch.as_tensor(labels_for_seeds([TRAIN[j] for j in idx]), device=dev)
        torch.cuda.synchronize(); t0 = time.time()
        if obj == "BA":
            loss, _ = loss_ba(m, jq, fp, xa, lab, a, sch, Ac, bc, den)
        else:
            loss = loss_bl(m, jq, fp, xa, lab, a, sch, nrm)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in c.parameters())))
        opt.step()
        torch.cuda.synchronize(); dt = time.time() - t0
        ttrain += dt
        u += 1
        log.append({"update": u, "a": a, "loss": float(loss), "grad_norm": gn, "sec": dt, "ttrain": ttrain})
        if u % 10 == 0:
            print(f"    {tag}: update {u}, loss {np.mean([r['loss'] for r in log[-10:]]):.4f}, grad {gn:.3g}, train {ttrain:.0f} s", flush=True)
        if save and u in ckpt_at:
            atomic_torch_save(c.export(), os.path.join(CK, f"carry_{tag}_u{u}.pt"))
            atomic_torch_save(snap(), res_f)
    if save and time_cap_s is not None:
        atomic_torch_save(c.export(), os.path.join(CK, f"carry_{tag}_tm.pt"))
        atomic_torch_save(snap(), res_f)
    out = {"obj": obj, "updates": u, "examples": u * B, "ttrain_s": ttrain, "sec_per_update": ttrain / max(1, u - start), "log": log, "draw_sha": dsha,
           "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2 ** 30}
    if return_params:
        out["params"] = [p.detach().cpu().clone() for p in c.parameters()]
    if keep_state:
        out["state"] = snap()
    c.remove(); jq.remove()
    del m, fp
    torch.cuda.empty_cache()
    return out


def need(name):
    r = jload(name)
    if not (r and (r.get("pass") or r.get("status") == "complete")):
        print(f"[dar1] {name} required first", flush=True)
        sys.exit(6)
    return r


def stage_profile(dev):
    need("checks.json")
    guard("profile", 0.06)
    with Run(TAG, "profile", dev, {}) as run:
        res = {}
        for obj in ("BA", "BL"):
            torch.cuda.reset_peak_memory_stats()
            r = train(dev, obj, n_updates=10, save=False)
            res[obj] = {"sec_per_update": r["sec_per_update"], "peak_mem_GiB": r["peak_mem_GiB"],
                        "sec_per_update_excl_first": float(np.mean([x["sec"] for x in r["log"][1:]]))}
            print(f"  {obj}: {res[obj]}", flush=True)
        s = res["BA"]["sec_per_update_excl_first"]
        N = int(0.95 * ALLOC_H * 3600 / s)
        res["N"] = N
        res["ckpts"] = sorted({round(N / 4), round(N / 2), N})
        res["budget_limited"] = N < 100
        res.update(run.record())
    jdump("profile.json", res)
    print(res, flush=True)


def stage_train(dev, obj):
    prof = need("profile.json")
    N, ck = prof["N"], prof["ckpts"]
    if obj == "BA":
        guard("train_ba", ALLOC_H * 1.05)
        with Run(TAG, "train_BA", dev, {"N": N, "ckpts": ck, "lr": LR, "B": B}) as run:
            out = train(dev, "BA", N, ck)
            out.update(run.record())
        jdump("train_ba.json", {k: v for k, v in out.items()})
    else:
        ba = need("train_ba.json")
        cap = min(ba["ttrain_s"], ALLOC_H * 3600)
        guard("train_bl", ALLOC_H * 1.05)
        with Run(TAG, "train_BL", dev, {"N": N, "ckpts": ck, "time_cap_s": cap}) as run:
            out = train(dev, "BL", N, ck, time_cap_s=cap)
            out.update(run.record())
        out["time_matched_to_BA_train_s"] = cap
        jdump("train_bl.json", out)
    print({k: v for k, v in out.items() if k != "log"}, flush=True)


# ------------------------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def gen(M, d, seeds, arms, bs=32):
    lat, cnt = {a: [] for a in arms}, {}
    for b0 in range(0, len(seeds), bs):
        s = seeds[b0:b0 + bs]
        nz, lb = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            sp = AB.ARMS[a]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x, c = AB.sample(M, sp, d, nz, lb)
            assert torch.isfinite(x).all() and c == AB.expect(sp), (a, c)
            lat[a].append(x.cpu()); cnt[a] = c
            print(f"  seeds {s[0]}-{s[-1]} {a}: {time.perf_counter() - t0:.1f} s", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt


def register_all():
    prof = jload("profile.json")
    names = {"B0": None, "BL-arch": ARCH}
    for u in prof["ckpts"]:
        names[f"BA@{u}"] = os.path.join(CK, f"carry_BA_u{u}.pt")
        names[f"BL@{u}"] = os.path.join(CK, f"carry_BL_u{u}.pt")
    names["BL-tm"] = os.path.join(CK, "carry_BL_tm.pt")
    for k, p in names.items():
        if k == "B0":
            AB.ARMS["B0"] = AB.ARMS["CB"]
        else:
            register_carry(k, p)
    return names


def stage_dev(dev):
    need("train_bl.json")
    names = register_all()
    arms = ["FP50"] + list(names)
    guard("dev", 0.22)
    with Run(TAG, "dev", dev, {"seeds": [DEV[0], DEV[-1]], "arms": arms}) as run:
        M = eval_models(dev)
        d = make_diffusion(T)
        lat, cnt = gen(M, d, DEV, arms)
        del M
        torch.cuda.empty_cache()
        atomic_torch_save(lat, os.path.join(CK, "dev_latents.pt"))
        prof = jload("profile.json")
        Mx, C = AB.evaluate(lat, dev, [])
        S = summary(Mx)
        mean = {a: S[a]["rel_mse"]["mean"] for a in S}
        def pick_tie(pre):
            us = sorted(prof["ckpts"])
            best = min(mean[f"{pre}@{u}"] for u in us)
            return next(u for u in us if mean[f"{pre}@{u}"] <= best * (1 + 1e-6))
        ba_u, bl_u = pick_tie("BA"), pick_tie("BL")
        comps = {"BL-sel": f"BL@{bl_u}", "BL-tm": "BL-tm", "BL-arch": "BL-arch"}
        strongest = min(comps, key=lambda k: mean[comps[k]])
        ba = f"BA@{ba_u}"
        sc = comps[strongest]
        pairs = [(ba, sc), (ba, "B0"), (ba, f"BL@{bl_u}"), (ba, "BL-tm"), (ba, "BL-arch"), (f"BL@{bl_u}", "B0"), ("BL-arch", "B0"), ("BL-tm", "B0")]
        from boa_dit.weight_anchor.shared import contrast
        Cx = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in dict.fromkeys(pairs)}
        promo = {"BA_sel": ba, "strongest_comparator": strongest, "strongest_arm": sc, "ratio": mean[ba] / mean[sc],
                 "dlpips": S[ba]["lpips"]["mean"] - S[sc]["lpips"]["mean"]}
        promo["pass"] = bool(promo["ratio"] <= 0.95 and promo["dlpips"] <= 0)
        out = {"status": "complete", "seeds": [DEV[0], DEV[-1]], "n": len(DEV), "arms": Mx, "summary": S, "calls": cnt, "selected": {"BA": ba, "BL": f"BL@{bl_u}"},
               "comparators": comps, "contrasts": Cx, "promotion": promo,
               "rule": "BA-sel mean <= 0.95 x strongest of {BL-sel, BL-tm, BL-arch} and mean LPIPS not worse", **run.record()}
    jdump("dev.json", out)
    for a in S:
        print(f"  {a}: rel MSE {S[a]['rel_mse']['mean']:.5f} LPIPS {S[a]['lpips']['mean']:.4f}", flush=True)
    print(f"  promotion: {promo}", flush=True)


def stage_conf(dev):
    dv = need("dev.json")
    if not dv["promotion"]["pass"]:
        jdump("conf.json", {"status": "skipped (development promotion rule failed; 124000-124127 not opened)", "promotion": dv["promotion"]})
        return 4
    register_all()
    sel = dv["selected"]
    arms = ["FP50", "B0", sel["BA"], sel["BL"], "BL-tm", "BL-arch"]
    arms = list(dict.fromkeys(arms))
    guard("conf", 0.40)
    from boa_dit.weight_anchor.shared import contrast
    with Run(TAG, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "arms": arms}) as run:
        M = eval_models(dev)
        d = make_diffusion(T)
        lat, cnt = gen(M, d, CONF, arms)
        del M
        torch.cuda.empty_cache()
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt"))
        Mx, _ = AB.evaluate(lat, dev, [])
        sc = dv["promotion"]["strongest_arm"]
        pairs = [(sel["BA"], sc), (sel["BA"], "B0"), (sel["BA"], sel["BL"]), (sel["BA"], "BL-tm"), (sel["BA"], "BL-arch"), (sc, "B0")]
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in dict.fromkeys(pairs) if x != y}
        prim = C[f"{sel['BA']}_vs_{sc}"]
        m_, lp = prim["rel_mse"], prim["lpips"]
        reading = ("confirmed fidelity improvement" if m_["ratio_ci"][1] < 1 and not lp["diff_ci"][0] > 0 else
                   "harmful" if m_["ratio_ci"][0] > 1 else "inconclusive")
        out = {"status": "complete", "seeds": [CONF[0], CONF[-1]], "n": len(CONF), "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C,
               "primary": f"{sel['BA']}_vs_{sc}", "reading_primary": reading, **run.record()}
    jdump("conf.json", out)
    AB.show(Mx, C)


@torch.no_grad()
def stage_diag(dev):
    dv = need("dev.json")
    register_all()
    sel = dv["selected"]
    arms = {"B0": None, "BL-sel": AB.CARRY[sel["BL"]], "BA-sel": AB.CARRY[sel["BA"]], "BL-arch": ARCH}
    guard("diag", 0.10)
    with Run(TAG, "diag", dev, {"seeds": [DIAG[0], DIAG[-1]]}) as run:
        out = {"status": "complete", "seeds": [DIAG[0], DIAG[-1]], "window_heldout": {}, "teacher_forced": {}, "live": {}}
        cache = load_cache()["diag"]
        lab = torch.as_tensor(labels_for_seeds(DIAG), device=dev)
        for name, path in arms.items():
            d, sch, Ac, bc, m, jq, c, fp = trainer(dev, path)
            den, nrm = load_den(dev), DJ.norm_consts(dev)
            wl = {}
            for a in STARTS:
                xa = cache[:, a // KA].to(dev)
                lba, _ = loss_ba(m, jq, fp, xa, lab, a, sch, Ac, bc, den, ck=False)
                lbl = loss_bl(m, jq, fp, xa, lab, a, sch, nrm)
                wl[a] = {"BA_loss": float(lba), "BL_loss": float(lbl)}
            out["window_heldout"][name] = {"per_start": wl, "BA_mean": float(np.mean([v["BA_loss"] for v in wl.values()])),
                                           "BL_mean": float(np.mean([v["BL_loss"] for v in wl.values()]))}
            # teacher-forced: FP50 trajectory states
            x = noise_for_seeds(DIAG, dev).float()
            e = {}
            for i in range(T):
                f = guided_eps(fp, x, lab, i, sch)
                e[i] = (stu(m, jq, x, lab, i, sch, False) - f).double().cpu()
                x = ddim_step(x, f, i, sch)
            out["teacher_forced"][name] = interval_stats(e, Ac, bc)
            c.remove(); jq.remove()
            del m, fp
            torch.cuda.empty_cache()
        M = eval_models(dev)
        d = make_diffusion(T)
        Ac, bc = AB.coeffs(d)
        nz, lb = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        for name, path in arms.items():
            key = "CB" if path is None else f"diag_{name}"
            if path is not None:
                register_carry(key, path)
            D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": M["fp"]}
            AB.sample(M, AB.ARMS[key], d, nz, lb, diag=D)
            r = interval_stats(D["e"], Ac, bc)
            r["jump_norms"] = D["jump"]
            out["live"][name] = r
        out["note"] = "teacher-forced = FP50 trajectory states (matched across arms); live = each arm's own trajectory (not matched)"
        out.update(run.record())
    jdump("diag.json", out)
    for k in ("teacher_forced", "live"):
        for n, r in out[k].items():
            print(f"  {k} {n}: raw {r['raw']:.4g} HH {r['HH']:.4g} resid/HH {r['res'] / r['HH']:.3f} lag1 {r['lag_energy'][1]:.4g}", flush=True)


def interval_stats(e, Ac, bc):
    S = {"raw": sum(float(v.pow(2).sum()) for v in e.values()), "HH": 0.0, "vv": 0.0, "Hv": 0.0, "res": 0.0}
    for a in range(0, T - KA, KA):
        cc = a + KA
        if cc > 48:
            break
        w = AB.weights(Ac, bc, a, KA)
        H = sum(wj * (e[a + j] - e[a]) for j, wj in zip(range(1, KA), w))
        v = AB.bridge_coef(Ac, bc, a, KA) * (e[cc] - e[a])
        S["HH"] += float(H.pow(2).sum()); S["vv"] += float(v.pow(2).sum()); S["Hv"] += float((H * v).sum()); S["res"] += float((H - v).pow(2).sum())
    S["cos_H_v"] = S["Hv"] / math.sqrt(max(S["HH"] * S["vv"], 1e-300))
    S["lag_energy"] = {L: sum(float((e[i + L] - e[i]).pow(2).sum()) for i in range(T - L) if i in e and i + L in e) for L in (1, 2, 4, 8)}
    return S


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    st = sys.argv[1]
    f = {"cache": stage_cache, "norm": stage_norm, "checks": stage_checks, "profile": stage_profile, "train_ba": lambda d: stage_train(d, "BA"),
         "train_bl": lambda d: stage_train(d, "BL"), "dev": stage_dev, "conf": stage_conf, "diag": stage_diag}[st]
    sys.exit(f(dev) or 0)


if __name__ == "__main__":
    main()
