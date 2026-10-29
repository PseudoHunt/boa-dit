"""DiT activation-feedback + weight-correction pilot (results/dit_joint/PROTOCOL.md). Stage tag j1.
Stages (resumable; each skips itself when its output exists): base reparam_check calib_act checks profile normalizers train_C_L train_E_L train_C_R
train_E_R select eval report.
Sources: carry training from followup/carry_opt.py (e3-confirm 21f0c55); static A8 / DCT-4 feedback arithmetic from temporal/dit_act.py
(temporal-oracles 216e4e6), re-implemented here as a state-in/state-out quantizer (checkpoint-safe) and checked bitwise against the original.
Run: python -m boa_dit.dit_joint.run <stage>
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.followup.carry_opt import Carry, Sched, ddim_step, guided_eps
from boa_dit.model import LINEARS, block_linear, decode, labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample
from boa_dit.temporal import act as ACT
from boa_dit.temporal import dit_act as DA

STAGE = "j1"
OUT = os.path.join(RESULTS, "dit_joint")
CK = os.path.join(CKPT_DIR, "dit_joint")
T, CFG, K, LAM, LR, B, UPDATES = 50, 1.5, 5, 0.5, 1e-4, 4, 200
SNAPS = (0, 25, 50, 100, 200)
CAL, TRAIN, DEV, SEALED = list(range(60000, 60032)), list(range(60100, 60164)), list(range(60200, 60232)), list(range(61000, 61128))
DIAG = SEALED[:32]
SMOOTH = os.path.join(CKPT_DIR, "temporal", "dit_smooth_s.pt")
ORIG = ("deflate_heig_r8_w4", os.path.join(CKPT_DIR, "ckpt_deflate_heig_r8_w4", "model.pt"),
        os.path.join(CKPT_DIR, "temporal", "rebuild_quant", "deflate_heig_r8_w4", "carry.pt"))
BASE_NAME = "deflate_heig_r8_w4_s"
BASE_STATE = os.path.join(CKPT_DIR, f"ckpt_{BASE_NAME}", "model.pt")
BASE_CARRY = os.path.join(RESULTS, "deflate", "quant", BASE_NAME, "carry.pt")
TABLES = os.path.join(CK, "static_single8.pt")
KEYS = [f"b{l}.{ln}" for l in range(28) for ln in LINEARS]
ARMS = {"C_L": ("local", False), "E_L": ("local", True), "C_R": ("roll", False), "E_R": ("roll", True)}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def jdump(name, obj):
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, name), obj)


def done(name):
    return os.path.exists(os.path.join(OUT, name))


# ------------------------------------------------------------------------------------------------ smoothed models
def smoothing():
    return torch.load(SMOOTH, map_location="cpu")


def apply_smoothing(m, sm):
    """Reparameterize fc1/fc2: W~ = W diag(s); a permanent forward PRE-hook feeds x~ = x/s (registered first, so every later pre-hook,
    the Gram hooks of the builder, the A8 quantizer and the carry all see x~)."""
    dev = next(m.parameters()).device
    for key, s in sm.items():
        l, ln = key.split(".", 1)
        lin = block_linear(m.blocks[int(l[1:])], ln)
        s = s.to(dev).float()
        lin.weight.data = (lin.weight.data.float() * s[None, :]).to(lin.weight.dtype)

        def pre(mod, inp, s=s):
            x = inp[0]
            return ((x.float() / s).to(x.dtype),)
        lin.register_forward_pre_hook(pre)
    return m


def load_dit_smoothed(device, dtype=torch.float32):
    return apply_smoothing(load_dit(device, dtype), smoothing())


def common_base(dev, carry=True):
    m = load_dit_smoothed(dev)
    load_quant_state(m, BASE_STATE)
    m.requires_grad_(False)
    hs = install(m, torch.load(BASE_CARRY, map_location="cpu")) if carry else []
    return m, hs


def common_base_nocarry(dev):
    m = load_dit_smoothed(dev)
    load_quant_state(m, BASE_STATE)
    m.requires_grad_(False)
    return m


def orig_base(dev):
    m = load_dit(dev)
    load_quant_state(m, ORIG[1])
    m.requires_grad_(False)
    install(m, torch.load(ORIG[2], map_location="cpu"))
    return m


def fp_model(dev):
    m = load_dit(dev)
    m.requires_grad_(False)
    return m


# ------------------------------------------------------------------------------------------------ quantizer
class _Ident(torch.autograd.Function):
    """forward: the deployed quantizer's value (bit-identical); backward: identity (STE)."""

    @staticmethod
    def forward(ctx, x, val):
        return val

    @staticmethod
    def backward(ctx, g):
        return g, None


class JointQuant:
    """Static per-tensor asymmetric A8 (single8) on the 112 block-linear inputs, optional DCT-4 temporal feedback. Arithmetic copied from
    temporal/dit_act.Quantize (no smoothing here: it is folded into the base), guard on, residual clamp ±Δ/2, fp16 state of 4 coefficients per
    channel per site per row. State is explicit: state_in (read) and state_out (written) per call, so a checkpointed recomputation reads the same
    state and never advances it twice. mode: 'off' (pass-through), 'static', 'fb'."""

    def __init__(self, model, dev, mode="static"):
        tabs = torch.load(TABLES, map_location="cpu")
        self.p = {k: (tb["D"].float().to(dev), tb["Z"].float().to(dev), tb["qmax"]) for k, tb in tabs.items()}
        self.U = DA.token_basis("dct4", dev)
        self.mode, self.t, self.dev = mode, None, dev
        self.state_in, self.state_out = {}, {}
        self.auto = True                                                     # eval: state_out -> state_in automatically, reset at t = 0
        self.calls = {k: 0 for k in KEYS}
        self.clip = torch.zeros(T, 2, dtype=torch.float64)
        self.hs = DA.input_hooks(model, self._fn)

    def step_t(self):
        return CTX.step if self.t is None else self.t

    def _fn(self, key, x, mod):
        if self.mode == "off":
            return None
        t = self.step_t()
        self.calls[key] += 1
        xs = x.float()
        D, Z, qm = self.p[key]
        lo, hi = -Z[t] * D[t], (qm - 1 - Z[t]) * D[t]
        with torch.no_grad():
            xd = xs.detach()
            if self.mode == "static":
                val = ACT.q_static(xd, D[t], Z[t], torch.zeros((), device=xd.device), qm)
                self.clip[t] += torch.tensor([float(((xd < lo) | (xd > hi)).sum()), float(xd.numel())], dtype=torch.float64)
            else:
                val = self._fb(key, t, xd, D, Z, qm, lo)
        return _Ident.apply(xs, val).to(x.dtype)

    def _fb(self, key, t, xs, D, Z, qm, lo):
        U = self.U
        shp = (xs.shape[0], U.shape[1], xs.shape[2])
        if key not in self.state_in or tuple(self.state_in[key].shape) != shp or (self.auto and t == 0):
            self.state_in[key] = torch.zeros(shp, device=xs.device, dtype=torch.float16)
        s_prev = torch.einsum("tk,rkc->rtc", U, self.state_in[key].float())
        hi_rep = (qm - Z[t]) * D[t]
        xcl = torch.minimum(torch.maximum(xs, lo), hi_rep)
        r_fed = torch.minimum(torch.maximum(s_prev, lo - xcl), hi_rep - xcl)
        u = xcl + r_fed
        uc = torch.minimum(torch.maximum(u, lo), hi_rep)
        q = ACT.q_static(uc, D[t], Z[t], torch.zeros((), device=xs.device), qm, hi_code=qm)
        consumed = q.to(torch.float16).float()
        r = (uc - consumed).clamp(-D[t] / 2, D[t] / 2)
        s = torch.einsum("tk,rtc->rkc", U, r).to(torch.float16)
        self.state_out[key] = s
        if self.auto:
            self.state_in[key] = s
        self.clip[t] += torch.tensor([float((u != uc).sum()), float(u.numel())], dtype=torch.float64)
        return consumed

    def snapshot(self):
        return tuple(self.state_out[k].detach().clone() for k in KEYS) if self.state_out else None

    def load(self, st):
        self.state_in = {} if st is None else {k: v for k, v in zip(KEYS, st)}

    def state_bytes(self):
        return int(sum(v.numel() * v.element_size() for v in self.state_out.values()))

    def remove(self):
        for h in self.hs:
            h.remove()


# ------------------------------------------------------------------------------------------------ stages: base, reparam, calibration
def stage_base(dev):
    """common smoothed rank-8 W4 base: the heig recipe (quantize_dc --mode heig --r 8 --bits 4) on the reparameterized model."""
    if os.path.exists(BASE_STATE) and os.path.exists(BASE_CARRY):
        print("  base exists", flush=True)
    else:
        from boa_dit.deflate import quantize_dc as Q
        Q.load_dit = lambda device, dtype=torch.float32: load_dit_smoothed(device, dtype)
        Q.arm_name = lambda a: BASE_NAME
        sys.argv = ["quantize_dc", "--mode", "heig", "--r", "8", "--bits", "4", "--device", dev, "--resume"]
        try:
            Q.main()
        except FileNotFoundError as e:                                        # post-hoc uniform-drift diagnostic only (after model + carry are saved)
            print(f"  post-hoc diagnostic skipped: {e}", flush=True)
        assert os.path.exists(BASE_STATE) and os.path.exists(BASE_CARRY), "base build incomplete"
    rec = {"name": BASE_NAME, "recipe": "deflate.quantize_dc --mode heig --r 8 --bits 4 (calibration seeds 1000-1031) on W~ = W diag(s) at mlp.fc1/fc2",
           "smoothing_file": SMOOTH, "smoothing_sha256": sha(SMOOTH), "model_sha256": sha(BASE_STATE), "carry_sha256": sha(BASE_CARRY),
           "smoothed_sites": sorted(smoothing())}
    jdump("base.json", rec)
    print(rec, flush=True)


@torch.no_grad()
def stage_reparam_check(dev):
    """FP reparameterization (folded s, no quantization) reproduces FP outputs: fp32 forward, 4 seeds x 3 steps, rel <= 1e-5."""
    a, b = load_dit(dev), load_dit_smoothed(dev)
    d = make_diffusion(T)
    sch = Sched(d, dev)
    seeds = CAL[:4]
    lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
    _, states = ddim_sample(a, d, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), CFG, record_states=True, amp=False)
    errs = {}
    for i in (0, 25, 49):
        ea = guided_eps(a, states[i], lab, i, sch, amp=False)
        eb = guided_eps(b, states[i], lab, i, sch, amp=False)
        errs[i] = float((ea - eb).norm() / ea.norm())
    res = {"rel_diff_fp32": errs, "tolerance": 1e-5, "pass": max(errs.values()) <= 1e-5}
    jdump("reparam_check.json", res)
    print(res, flush=True)
    assert res["pass"]


@torch.no_grad()
def stage_calib_act(dev):
    """single8 tables on the common base + carry, feedback off, trajectories of seeds 60000-60031 (both CFG halves, all tokens)."""
    m, _ = common_base(dev)
    d = make_diffusion(T)
    cal = ACT.Calib(dev)
    with Run(STAGE, "j1_calib_act", dev, {"seeds": [CAL[0], CAL[-1]]}) as run:
        for pas in ("minmax", "hist"):
            fn = (lambda key, x, mod: cal.add_minmax(key, CTX.step, x.float())) if pas == "minmax" else (lambda key, x, mod: cal.add_hist(key, CTX.step, x.float()))
            hs = DA.input_hooks(m, lambda key, x, mod, fn=fn: (fn(key, x, mod), None)[1])
            ddim_sample(m, d, noise_for_seeds(CAL, dev), labels_for_seeds(CAL), CFG)
            for h in hs:
                h.remove()
        tabs = cal.tables(T, lo_min_keys=DA.GELU_TYPES)
        os.makedirs(CK, exist_ok=True)
        atomic_torch_save(tabs["single8"], TABLES)
        rec = {"tables_sha256": sha(TABLES), "clip_rate_by_type": ACT.clip_summary({"single8": tabs["single8"]}), **run.record()}
    jdump("calib_act.json", rec)
    print(rec["clip_rate_by_type"], flush=True)


# ------------------------------------------------------------------------------------------------ sampling with a configuration
def student(dev, carry_path=None, mode="static"):
    """common base (+ carry file or its original carry) with a JointQuant in `mode`; returns (model, quant, hooks)."""
    m = common_base_nocarry(dev)
    jq = JointQuant(m, dev, mode)                                            # registered after the smoothing pre-hooks
    hs = install(m, torch.load(carry_path or BASE_CARRY, map_location="cpu"))
    return m, jq, hs


@torch.no_grad()
def sample_config(m, jq, seeds, dev, bs=32):
    d = make_diffusion(T)
    out = []
    jq.t, jq.auto = None, True                                               # eval path: step from CTX, state chained and reset at t = 0
    for b in range(0, len(seeds), bs):
        s = seeds[b:b + bs]
        jq.state_in, jq.state_out = {}, {}
        out.append(ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), CFG).cpu())
    return torch.cat(out)


# ------------------------------------------------------------------------------------------------ training-path rollout
def call_model(m, jq, x, lab, i, sch, *state):
    """one denoiser call with explicit feedback state (checkpoint-safe): sets the step and the incoming state, returns the guided eps."""
    jq.t = i
    jq.load(state if state else None)
    jq.state_out = {}
    return guided_eps(m, x, lab, i, sch)


def prefix(m, jq, seeds, i0, sch, dev):
    """the CURRENT student's own trajectory from the original noise to step i0, no gradients, with its actual feedback state."""
    x = noise_for_seeds(seeds, dev)
    lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
    st = None
    with torch.no_grad():
        for i in range(i0):
            eps = call_model(m, jq, x, lab, i, sch, *(st or ()))
            st = jq.snapshot() if jq.mode == "fb" else None
            x = ddim_step(x, eps, i, sch)
    return x.detach(), lab, st


def window(m, jq, x, lab, i0, k, sch, grad=True, ckpt=True):
    """k student calls from (x, state) with gradients through the latent; state detached at every temporal update."""
    st, eps0 = getattr(window, "st", None), None
    for i in range(i0, i0 + k):
        args = st or ()
        eps = checkpoint(call_model, m, jq, x, lab, i, sch, *args, use_reentrant=False) if (grad and ckpt) else call_model(m, jq, x, lab, i, sch, *args)
        st = jq.snapshot() if jq.mode == "fb" else None
        eps0 = eps if eps0 is None else eps0
        x = ddim_step(x, eps, i, sch)
    window.last_st = st
    return eps0, x


def fp_rollout(fp, x, lab, i0, k, sch):
    with torch.no_grad():
        e0 = None
        for i in range(i0, i0 + k):
            e = guided_eps(fp, x, lab, i, sch)
            e0 = e if e0 is None else e0
            x = ddim_step(x, e, i, sch)
    return e0, x


# ------------------------------------------------------------------------------------------------ checks, profile, normalizers
def stage_checks(dev):
    res = {}
    d = make_diffusion(T)
    sch = Sched(d, dev)
    seeds = DEV[:4]
    lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
    nz = lambda: noise_for_seeds(seeds, dev)
    with torch.no_grad():
        # 1. A8 + feedback disabled == the W4+carry reference (same numerical path); original heig-8 reproduces the archived E3C C0 latents
        m, hs = common_base(dev)
        ref = ddim_sample(m, d, nz(), labels_for_seeds(seeds), CFG).cpu()
        jq_off = JointQuant(m, dev, "off")
        res["c1_off_equals_reference"] = bool(torch.equal(ddim_sample(m, d, nz(), labels_for_seeds(seeds), CFG).cpu(), ref))
        jq_off.remove()
        e3c = os.path.join(CKPT_DIR, "e3c", "eval_latents.pt")
        if os.path.exists(e3c):
            o = orig_base(dev)
            s32 = list(range(20000, 20032))
            res["c1_orig_heig8_equals_e3c_C0_batch32"] = bool(torch.equal(ddim_sample(o, d, noise_for_seeds(s32, dev), labels_for_seeds(s32), CFG).cpu(),
                                                                          torch.load(e3c)["C0"][:32]))
            del o
        # 2. JointQuant static / DCT-4 == temporal Quantize (tables swapped in, smoothing off) over full sampling, bitwise
        DA.bpath = lambda kind: TABLES
        DA.load_smooth = lambda dev=None: {}
        for mode, kw in (("static", {}), ("fb", {"feedback": True, "compress": "dct4"})):
            jq = JointQuant(m, dev, mode)
            mine = ddim_sample(m, d, nz(), labels_for_seeds(seeds), CFG).cpu()
            jq.remove()
            qo = DA.Quantize(m, "single8", "none", dev, **kw)
            orig = ddim_sample(m, d, nz(), labels_for_seeds(seeds), CFG).cpu()
            qo.remove()
            res[f"c2_{mode}_equals_temporal_Quantize"] = bool(torch.equal(mine, orig))
        for h in hs:
            h.remove()
    # 3. trainable carry at init == installed carry, export bitwise
    m = common_base_nocarry(dev)
    jq = JointQuant(m, dev, "fb")
    base_c = torch.load(BASE_CARRY, map_location="cpu")
    with torch.no_grad():
        hs = install(m, base_c)
        a = sample_config(m, jq, seeds, dev)
        for h in hs:
            h.remove()
        c = Carry(m, base_c, dev)
        b = sample_config(m, jq, seeds, dev)
    exp = c.export()
    res["c3_trainable_init_equals_installed"] = bool(torch.equal(a, b))
    res["c3_export_bitwise"] = all(torch.equal(exp[k]["V"], base_c[k]["V"]) and torch.equal(exp[k]["delta"], base_c[k]["delta"]) for k in base_c)
    # 4. prefix + restored state == uninterrupted training-path rollout (bitwise), and vs ddim_sample (tolerance)
    jq.auto = False
    c.remove()                                                                # the trainable carry is re-attached below for 5-7
    c = Carry(m, base_c, dev)
    with torch.no_grad():
        xa, _, _ = prefix(m, jq, seeds, T, sch, dev)
        x30, lab30, st30 = prefix(m, jq, seeds, 30, sch, dev)
        window.st = st30
        _, xb = window(m, jq, x30, lab30, 30, T - 30, sch, grad=False)
        window.st = None
    res["c4_prefix_restore_bitwise"] = bool(torch.equal(xa, xb))
    res["c4_training_path_vs_ddim_sample_rel"] = float((xa.cpu() - b).norm() / b.norm())
    # 5. checkpointing on/off: forward bitwise, gradients within 1e-4 relative, one state update per call per site
    x20, lab20, st20 = prefix(m, jq, seeds, 20, sch, dev)
    grads, outs = {}, {}
    for ck in (False, True):
        c.zero_grad(set_to_none=True)
        jq.calls = {k: 0 for k in KEYS}
        window.st = st20
        e0, xe = window(m, jq, x20.clone(), lab20, 20, K, sch, grad=True, ckpt=ck)
        window.st = None
        (xe.pow(2).mean() + e0.pow(2).mean()).backward()
        grads[ck] = torch.cat([p.grad.flatten() for p in c.parameters()])
        outs[ck] = (xe.detach(), window.last_st)
        res[f"c5_calls_per_site_ckpt{int(ck)}"] = jq.calls[KEYS[0]]
    res["c5_forward_bitwise"] = bool(torch.equal(outs[False][0], outs[True][0]))
    res["c5_end_state_equal (no double advance)"] = bool(all(torch.equal(a, b) for a, b in zip(outs[False][1], outs[True][1])))
    res["c5_grad_rel"] = float((grads[False] - grads[True]).norm() / grads[False].norm())
    # 6. finite nonzero gradients at early and late sites; frozen weights unchanged by an optimizer step
    g0, g1 = c.V[0].grad, c.V[len(c.keys) - 1].grad
    res["c6_grad_first_last_nonzero"] = bool(g0 is not None and g1 is not None and g0.abs().sum() > 0 and g1.abs().sum() > 0)
    res["c6_grads_finite"] = bool(torch.isfinite(grads[True]).all())
    w_before = block_linear(m.blocks[0], "attn.qkv").weight.detach().clone()
    torch.optim.Adam(c.parameters(), lr=LR).step()
    res["c6_frozen_unchanged"] = bool(torch.equal(w_before, block_linear(m.blocks[0], "attn.qkv").weight))
    # 7. export -> install reproduces the trainable path; resume restores optimizer/RNG/draws (see train(): asserted on resume)
    exp2 = c.export()
    with torch.no_grad():
        jq.auto = True
        c_out = sample_config(m, jq, seeds, dev)
        c.remove()
        hs = install(m, exp2)
        i_out = sample_config(m, jq, seeds, dev)
        for h in hs:
            h.remove()
    res["c7_export_install_bitwise"] = bool(torch.equal(c_out, i_out))
    # 8. shapes / tables / 50 calls
    res["c8_tables_sha256"] = sha(TABLES)
    res["c8_calls_per_site_per_sample"] = None
    jq.calls = {k: 0 for k in KEYS}
    with torch.no_grad():
        hs = install(m, base_c)
        sample_config(m, jq, seeds[:1], dev)
        for h in hs:
            h.remove()
    res["c8_calls_per_site_per_sample"] = jq.calls[KEYS[0]]
    jq.remove()
    res["tolerances"] = {"c1": "bitwise", "c2": "bitwise", "c3": "bitwise", "c4": "bitwise (training path); ddim_sample reported", "c5": "forward bitwise, grad <= 1e-4", "c7": "bitwise"}
    res["pass"] = all([res["c1_off_equals_reference"], res.get("c1_orig_heig8_equals_e3c_C0_batch32", True), res["c2_static_equals_temporal_Quantize"],
                       res["c2_fb_equals_temporal_Quantize"], res["c3_trainable_init_equals_installed"], res["c3_export_bitwise"],
                       res["c4_prefix_restore_bitwise"], res["c5_forward_bitwise"], res["c5_end_state_equal (no double advance)"], res["c5_grad_rel"] <= 1e-4, res["c5_calls_per_site_ckpt0"] == K,
                       res["c5_calls_per_site_ckpt1"] == 2 * K, res["c6_grad_first_last_nonzero"], res["c6_grads_finite"], res["c6_frozen_unchanged"],
                       res["c7_export_install_bitwise"], res["c8_calls_per_site_per_sample"] == T])
    jdump("checks.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if not res["pass"]:
        raise SystemExit(6)


def norm_consts(dev):
    f = os.path.join(CK, "norm.pt")
    if os.path.exists(f):
        return torch.load(f)
    d = make_diffusion(T)
    sch = Sched(d, dev)
    m, jq, hs = student(dev, mode="static")
    fp = fp_model(dev)
    seeds = TRAIN[:16]
    n_loc, n_roll = torch.zeros(T), torch.zeros(T)
    with torch.no_grad():
        for i in range(T):
            x, lab, _ = prefix(m, jq, seeds, i, sch, dev)
            k = min(K, T - i)
            es, xs = window(m, jq, x, lab, i, k, sch, grad=False)
            et, xt = fp_rollout(fp, x, lab, i, k, sch)
            n_loc[i] = (es - et).pow(2).flatten(1).sum(1).mean()
            n_roll[i] = (xs - xt).pow(2).flatten(1).sum(1).mean()
    floor_l, floor_r = 1e-6 * float(n_loc.mean()), 1e-6 * float(n_roll.mean())
    out = {"loc": n_loc.clamp_min(floor_l), "roll": n_roll.clamp_min(floor_r), "floor": [floor_l, floor_r]}
    atomic_torch_save(out, f)
    return out


def stage_normalizers(dev):
    with Run(STAGE, "j1_normalizers", dev, {"seeds": [TRAIN[0], TRAIN[15]]}) as run:
        n = norm_consts(dev)
        rec = {"n_loc": n["loc"].tolist(), "n_roll": n["roll"].tolist(), "floor": n["floor"], **run.record()}
    jdump("normalizers.json", rec)
    print({"n_loc[0,25,49]": [rec["n_loc"][i] for i in (0, 25, 49)], "n_roll[0,25,49]": [rec["n_roll"][i] for i in (0, 25, 49)]}, flush=True)


# ------------------------------------------------------------------------------------------------ training
def draws(n):
    g = torch.Generator().manual_seed(0)
    return [(torch.randint(0, len(TRAIN), (B,), generator=g).tolist(), int(torch.randint(0, T, (1,), generator=g))) for _ in range(n)]


def train(dev, arm, profile_updates=None):
    obj, fb = ARMS[arm]
    d = make_diffusion(T)
    sch = Sched(d, dev)
    nrm = norm_consts(dev)
    m = common_base_nocarry(dev)
    jq = JointQuant(m, dev, "fb" if fb else "static")
    jq.auto = False
    c = Carry(m, torch.load(BASE_CARRY, map_location="cpu"), dev)
    opt = torch.optim.Adam(c.parameters(), lr=LR)
    fp = fp_model(dev)
    plan = draws(UPDATES)
    res_f = os.path.join(CK, f"resume_{arm}.pt")
    start, log = 0, []
    if profile_updates is None and os.path.exists(res_f):                  # resume: masters, optimizer, update count (draws are a fixed list)
        r = torch.load(res_f, map_location="cpu")
        for p, v in zip(c.parameters(), r["params"]):
            p.data.copy_(v.to(dev))
        opt.load_state_dict(r["opt"])
        start, log = r["update"], r["log"]
        assert r["draw_sha"] == hashlib.sha256(json.dumps(plan).encode()).hexdigest()
        print(f"  resumed {arm} at update {start}", flush=True)
    if start == 0 and profile_updates is None:
        atomic_torch_save(c.export(), os.path.join(CK, f"carry_{arm}_u0.pt"))
    t0 = time.time()
    n_end = profile_updates if profile_updates else UPDATES
    for u in range(start, n_end):
        seeds_i, i0 = plan[u]
        seeds = [TRAIN[j] for j in seeds_i]
        k = min(K, T - i0)
        x, lab, st = prefix(m, jq, seeds, i0, sch, dev)
        et, xt = fp_rollout(fp, x, lab, i0, k if obj == "roll" else 1, sch)
        window.st = st
        if obj == "local":
            es, _ = window(m, jq, x, lab, i0, 1, sch, grad=True, ckpt=False)
            ll = (es - et).pow(2).flatten(1).sum(1).mean() / nrm["loc"][i0]
            loss, lr_ = ll, None
        else:
            es, xs = window(m, jq, x, lab, i0, k, sch, grad=True, ckpt=True)
            ll = (es - et).pow(2).flatten(1).sum(1).mean() / nrm["loc"][i0]
            lr_ = (xs - xt).pow(2).flatten(1).sum(1).mean() / nrm["roll"][i0]
            loss = ll + LAM * lr_
        window.st = None
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum(p.grad.pow(2).sum() for p in c.parameters())))
        opt.step()
        log.append({"update": u + 1, "i0": i0, "loss": float(loss), "local": float(ll), "roll": None if lr_ is None else float(lr_), "grad_norm": gn,
                    "t": time.time() - t0})
        if (u + 1) % 10 == 0:
            print(f"    {arm}: update {u + 1}, loss {float(loss):.4f} (local {float(ll):.4f}" + ("" if lr_ is None else f", roll {float(lr_):.4f}") +
                  f"), grad {gn:.3g}, {time.time() - t0:.0f} s", flush=True)
        if profile_updates is None and (u + 1) in SNAPS:
            atomic_torch_save(c.export(), os.path.join(CK, f"carry_{arm}_u{u + 1}.pt"))
            atomic_torch_save({"params": [p.detach().cpu() for p in c.parameters()], "opt": opt.state_dict(), "update": u + 1, "log": log,
                               "draw_sha": hashlib.sha256(json.dumps(plan).encode()).hexdigest()}, res_f)
    sec = (time.time() - t0) / max(1, n_end - start)
    c.remove()
    jq.remove()
    return {"arm": arm, "objective": obj, "feedback_in_training": fb, "updates": n_end, "sec_per_update": sec, "log": log,
            "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2 ** 30}


def stage_profile(dev):
    res = {"gpu": torch.cuda.get_device_name(0)}
    with Run(STAGE, "j1_profile", dev, {}) as run:
        for arm in ARMS:
            torch.cuda.reset_peak_memory_stats()
            r = train(dev, arm, profile_updates=3)
            res[arm] = {"sec_per_update": r["sec_per_update"], "peak_mem_GiB": r["peak_mem_GiB"]}
            torch.cuda.empty_cache()
        res["projected_training_h"] = sum(res[a]["sec_per_update"] for a in ARMS) * UPDATES / 3600
        res.update(run.record())
    jdump("profile.json", res)
    print(res, flush=True)


def stage_train(dev, arm):
    with Run(STAGE, f"j1_train_{arm}", dev, {"arm": arm, "updates": UPDATES, "lr": LR, "K": K, "lam": LAM, "B": B}) as run:
        torch.cuda.reset_peak_memory_stats()
        r = train(dev, arm)
        r.update(run.record())
    jdump(f"train_{arm}.json", r)


# ------------------------------------------------------------------------------------------------ evaluation
def metrics(x, xf, vae, net, dev):
    rel = ((x - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1)).tolist()
    ps, lp = [], []
    imf, im = decode(vae, xf.to(dev)), decode(vae, x.to(dev))
    for i in range(len(x)):
        a, b = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
        ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
        lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
    return {"rel_mse": rel, "psnr": ps, "lpips": lp}


def fp_latents(seeds, dev, tag):
    f = os.path.join(CK, f"fp_{tag}.pt")
    if os.path.exists(f):
        return torch.load(f)
    fp = fp_model(dev)
    d = make_diffusion(T)
    with torch.no_grad():
        x = torch.cat([ddim_sample(fp, d, noise_for_seeds(seeds[b:b + 32], dev), labels_for_seeds(seeds[b:b + 32]), CFG).cpu() for b in range(0, len(seeds), 32)])
    atomic_torch_save(x, f)
    return x


def eval_one(dev, carry_path, mode, seeds):
    m, jq, hs = student(dev, carry_path, mode)
    t0 = time.time()
    x = sample_config(m, jq, seeds, dev)
    sec = (time.time() - t0) / len(seeds)
    sb, clip = jq.state_bytes(), jq.clip.clone()
    jq.remove()
    return x, {"sec_per_image": sec, "state_bytes_last_batch": sb, "clip_rate": float(clip[:, 0].sum() / clip[:, 1].sum().clamp_min(1))}


def stage_select(dev):
    """development seeds 60200-60231: C arms pick among snapshots with feedback OFF (D reuses C's pick); E arms pick with feedback ON."""
    import lpips
    xf = fp_latents(DEV, dev, "dev")
    vae = load_vae(dev)
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    res = {"seeds": [DEV[0], DEV[-1]], "arms": {}}
    with Run(STAGE, "j1_select", dev, {}) as run:
        for arm, (obj, fb) in ARMS.items():
            mode = "fb" if fb else "static"
            r = {}
            for u in SNAPS:
                x, _ = eval_one(dev, os.path.join(CK, f"carry_{arm}_u{u}.pt"), mode, DEV)
                r[f"u{u}"] = float(np.mean(((x - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1)).numpy()))
                print(f"  dev {arm} u{u} ({mode}): latent {r[f'u{u}']:.5f}", flush=True)
            best = min(r, key=r.get)
            res["arms"][arm] = {"dev_latent": r, "selected": best, "label": "untrained" if best == "u0" else best}
        res.update(run.record())
    jdump("select.json", res)


def stage_eval(dev):
    """sealed seeds 61000-61127 (opened once, after select.json is frozen): every configuration at fixed 200 updates and at the selected snapshot."""
    import lpips
    sel = json.load(open(os.path.join(OUT, "select.json")))["arms"]
    xf = fp_latents(SEALED, dev, "sealed")
    vae = load_vae(dev)
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    f = os.path.join(OUT, "eval.json")
    res = json.load(open(f)) if os.path.exists(f) else {"seeds": [SEALED[0], SEALED[-1]], "configs": {}}
    cp = lambda arm, u: os.path.join(CK, f"carry_{arm}_u{u}.pt")
    configs = {"A": (BASE_CARRY, "static"), "B": (BASE_CARRY, "fb"),
               "C_L": (cp("C_L", 200), "static"), "D_L": (cp("C_L", 200), "fb"), "E_L": (cp("E_L", 200), "fb"),
               "C_R": (cp("C_R", 200), "static"), "D_R": (cp("C_R", 200), "fb"), "E_R": (cp("E_R", 200), "fb"),
               "W4A16_common": (BASE_CARRY, "off")}
    for arm in ARMS:
        s = sel[arm]["selected"]
        if arm.startswith("C"):
            configs[f"{arm}*"] = (cp(arm, s[1:]), "static")
            configs[f"D_{arm[2]}*"] = (cp(arm, s[1:]), "fb")
        else:
            configs[f"{arm}*"] = (cp(arm, s[1:]), "fb")
    with Run(STAGE, f"j1_eval_{len(res['configs'])}", dev, {"configs": list(configs)}) as run:
        lat_f = os.path.join(CK, "eval_latents.pt")
        lats = torch.load(lat_f) if os.path.exists(lat_f) else {"fp": xf}
        if "W4A16_orig" not in res["configs"]:
            o = orig_base(dev)
            d = make_diffusion(T)
            with torch.no_grad():
                x = torch.cat([ddim_sample(o, d, noise_for_seeds(SEALED[b:b + 32], dev), labels_for_seeds(SEALED[b:b + 32]), CFG).cpu() for b in range(0, 128, 32)])
            del o
            res["configs"]["W4A16_orig"] = {**metrics(x, xf, vae, net, dev), "path": "original heig-8 W4 + carry, no A8"}
            lats["W4A16_orig"] = x
        for name, (cpath, mode) in configs.items():
            if name in res["configs"]:
                continue
            x, info = eval_one(dev, cpath, mode, SEALED)
            res["configs"][name] = {**metrics(x, xf, vae, net, dev), **info, "carry": os.path.basename(cpath), "mode": mode}
            lats[name] = x
            print(f"  {name}: latent {np.mean(res['configs'][name]['rel_mse']):.5f} PSNR {np.mean(res['configs'][name]['psnr']):.2f} "
                  f"LPIPS {np.mean(res['configs'][name]['lpips']):.4f}", flush=True)
            atomic_torch_save(lats, lat_f)
            atomic_write_json(f, res)
        if "diag" not in res:
            res["diag"] = diag_ci(dev, {k: configs[k] for k in ("A", "B", "D_R", "E_R", "D_L", "E_L", "C_R", "C_L")})
        res.setdefault("runs", []).append(run.record())
    atomic_write_json(f, res)


@torch.no_grad()
def diag_ci(dev, configs):
    """C = T·||mean_t e_t||², I = mean_t ||e_t||², e_t = eps_config(x^F_t) - eps_FP(x^F_t) on the FP trajectory states of seeds 61000-61031;
    the configuration's feedback state evolves along those inputs (reset at t = 0)."""
    d = make_diffusion(T)
    fp = fp_model(dev)
    lab = labels_for_seeds(DIAG)
    xf, st = ddim_sample(fp, d, noise_for_seeds(DIAG, dev), lab, CFG, record_states=True)
    cap = {}
    ddim_sample(fp, d, noise_for_seeds(DIAG, dev), lab, CFG, eps_hook=lambda i, xx, o: (cap.__setitem__(i, o[:, :4].clone()), o)[1])
    del fp
    out = {}
    for name, (cpath, mode) in configs.items():
        m, jq, hs = student(dev, cpath, mode)
        ep = {}
        for i in range(T):
            CTX.step = i
            jq.t = i
            ep[i] = guided_eps(m, st[i], torch.as_tensor(lab, device=dev), i, Sched(d, dev))
        jq.remove()
        E = torch.stack([(ep[i] - cap[i]).double() for i in range(T)], 1)
        Cc = T * E.mean(1).pow(2).flatten(1).sum(1)
        I = E.pow(2).flatten(2).sum(2).mean(1)
        out[name] = {"C": Cc.tolist(), "I": I.tolist(), "C_over_I": float(Cc.sum() / I.sum())}
        print(f"  diag {name}: C {Cc.mean():.4g} I {I.mean():.4g} C/I {out[name]['C_over_I']:.3f}", flush=True)
    CTX.step = None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(CK, exist_ok=True)
    s = a.stage
    outputs = {"base": "base.json", "reparam_check": "reparam_check.json", "calib_act": "calib_act.json", "checks": "checks.json",
               "profile": "profile.json", "normalizers": "normalizers.json", "select": "select.json"}
    if s in outputs and done(outputs[s]):
        print(f"  {s}: done ({outputs[s]}), skipped", flush=True)
        return
    if s.startswith("train_") and done(f"{s}.json"):
        print(f"  {s}: done, skipped", flush=True)
        return
    if s == "report":
        import subprocess
        subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "scripts", "make_dit_joint.py")], check=True)
        return
    fn = {"base": stage_base, "reparam_check": stage_reparam_check, "calib_act": stage_calib_act, "checks": stage_checks, "profile": stage_profile,
          "normalizers": stage_normalizers, "select": stage_select, "eval": stage_eval}
    if s.startswith("train_"):
        stage_train(dev, s[6:])
    else:
        fn[s](dev)


if __name__ == "__main__":
    main()
