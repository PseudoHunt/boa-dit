"""Local weight-error refresh with temporal activation feedback (results/local_error_refresh/PROTOCOL.md). Stage tag ler1.
Cheap model: common smoothed rank-8 W4 base + original carry + static single8 tables (dit_joint, unchanged), JointQuant subclass LocalQuant that
captures h (JointQuant input) and q (its output) and can force a zero previous state (G). Refresh: at selected linears, b = W h - M h (fp32, autocast
off) at i mod K = 0, stored fp16, added after the complete cheap linear (carry included) at every step.
Stages: checks | screen | e1 | e2 | diag | conf | cost
Run: python -m boa_dit.local_refresh.run <stage>
"""
import argparse
import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as DJ
from boa_dit.model import LINEARS, NULL_CLASS, block_linear, decode, labels_for_seeds, load_dit, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import guided, ratio_ci
from boa_dit.weight_anchor.shared import boot_idx, ci, contrast, summary

STAGE = "ler1"
OUT = os.path.join(RESULTS, "local_error_refresh")
CK = os.path.join(CKPT_DIR, "local_error_refresh")
CAP_GPU_H = 2.0
CONF_RESERVE_H = 0.55
POOL = [f"b{l}.{ln}" for l in (3, 13, 24) for ln in LINEARS]                 # LINEARS = attn.qkv, attn.proj, mlp.fc1, mlp.fc2
SEL, DEV, DIAG = list(range(60200, 60208)), list(range(60208, 60224)), list(range(60208, 60216))
CONF, G1, CALIB = list(range(97100, 97164)), list(range(61000, 61032)), list(range(60000, 60032))
SCREEN_STEPS = (5, 25, 45)
BS = 32
C5_TOL, C4_TOL, C2_TOL, C2_TOL_FP16, C3_TOL = 5e-3, 1e-6, 1e-6, 1e-3, 1e-10
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
    print(f"[ler1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        sys.exit(5)


def provenance():
    out = {k: {"path": p, "sha256": DJ.sha(p), "expected": h} for k, (p, h) in HASHES.items()}
    assert all(v["sha256"] == v["expected"] for v in out.values()), out
    return out


def lin_of(m, key):
    l, ln = key.split(".", 1)
    return block_linear(m.blocks[int(l[1:])], ln)


# ------------------------------------------------------------------------------------------------------------------- quantizer
class LocalQuant(DJ.JointQuant):
    """JointQuant with unchanged arithmetic; records h (input, after smoothing) and q (output) at cap_keys; zero_state forces the previous
    feedback state to zero at every call (G); acct (dict) accumulates the quantizer-observer guarded-feedback accounting (energies only)."""

    def __init__(self, model, dev, mode):
        super().__init__(model, dev, mode)
        self.zero_state = False
        self.cap_keys, self.h, self.q = set(), {}, {}
        self.acct = None

    def _fn(self, key, x, mod):
        if self.zero_state and self.mode == "fb":
            self.state_in.pop(key, None)                                     # parent creates a zero state, exactly as at t = 0
        y = super()._fn(key, x, mod)
        if key in self.cap_keys:
            self.h[key], self.q[key] = x, y
        return y

    def _fb(self, key, t, xs, D, Z, qm, lo):
        if self.acct is None:
            return super()._fb(key, t, xs, D, Z, qm, lo)
        U = self.U
        shp = (xs.shape[0], U.shape[1], xs.shape[2])
        if key not in self.state_in or tuple(self.state_in[key].shape) != shp or (self.auto and t == 0):
            self.state_in[key] = torch.zeros(shp, device=xs.device, dtype=torch.float16)
        s_in = self.state_in[key]
        consumed = super()._fb(key, t, xs, D, Z, qm, lo)
        self._account(key, t, xs, s_in, self.state_out[key], consumed, D, Z, qm, lo)
        return consumed

    @torch.no_grad()
    def _account(self, key, t, xs, s_in, s_out, consumed, D, Z, qm, lo):
        """quantizer-observer-pilot InstrQuant accounting (same fp32 recomputation; energies and rates only):
        q - x = p - p_next + c0 + g + rho_u + c1 - (I-P)r - a + eta."""
        U = self.U
        p = torch.einsum("tk,rkc->rtc", U, s_in.float())
        hi_rep = (qm - Z[t]) * D[t]
        xcl = torch.minimum(torch.maximum(xs, lo), hi_rep)
        r_fed = torch.minimum(torch.maximum(p, lo - xcl), hi_rep - xcl)
        u = xcl + r_fed
        uc = torch.minimum(torch.maximum(u, lo), hi_rep)
        code = torch.clamp(torch.round(torch.clamp(uc, lo, hi_rep) / D[t]) + Z[t], 0, qm)
        Dd = float(D[t])
        pn = torch.einsum("tk,rkc->rtc", U, s_out.float())
        c0, g, rho, c1 = xcl - xs, r_fed - p, u - (xcl + r_fed), uc - u
        r_raw = uc - consumed
        r = r_raw.clamp(-Dd / 2, Dd / 2)
        a = r_raw - r
        Pr = torch.einsum("tk,rkc->rtc", U, torch.einsum("tk,rtc->rkc", U, r))
        terms = {"p": p, "-p_next": -pn, "c0_input_clip": c0, "g_state_guard": g, "rho_u": rho, "c1_second_clip": c1,
                 "-(I-P)r_discarded": -(r - Pr), "-a_residual_clamp": -a, "eta_storage": pn - Pr}
        lhs = consumed - xs
        res = lhs - sum(terms.values())
        n = float(xs.numel())
        rec = {f"E[{k}]": float(v.double().pow(2).sum()) for k, v in terms.items()}
        rec.update({"E[q-x]": float(lhs.double().pow(2).sum()), "E[identity_residual]": float(res.double().pow(2).sum()),
                    "n": n, "rate_input_clip": float((xcl != xs).sum()), "rate_state_guard": float((r_fed != p).sum()),
                    "rate_second_clip_old_counter": float((u != uc).sum()), "rate_spare_code": float((code == qm).sum())})
        A = self.acct.setdefault(key, {})
        for k, v in rec.items():
            A[k] = A.get(k, 0.0) + v


# ------------------------------------------------------------------------------------------------------------------- refresh
class Refresh:
    """Local held weight corrections at `sites` of a cheap model (forward hooks registered after the carry). Hook-free measurement helpers."""

    def __init__(self, m, jq, carry, wref, sites):
        self.m, self.jq, self.sites = m, jq, list(sites)
        dev = next(m.parameters()).device
        self.W = {k: wref[k] for k in self.sites}                              # plain fp32 tensors, no module
        self.Q = {k: lin_of(m, k).weight for k in self.sites}
        self.bias = {k: lin_of(m, k).bias for k in self.sites}
        self.V = {k: carry[k]["V"].to(dev) for k in self.sites}
        self.dl = {k: carry[k]["delta"].to(dev) for k in self.sites}
        self.ref_self = False                                                  # C5: reference := Q with the same carry (M = W)
        self.K, self.active, self.scale, self.perm = None, set(), 1.0, None
        self.trace, self.diag, self.debug = None, None, None
        self.reset()
        self.hs = [lin_of(m, k).register_forward_hook(self._hook(k)) for k in self.sites]

    def reset(self):
        self.b, self.bx, self.last = {}, {}, {}
        self.n_refresh = {k: 0 for k in self.sites}

    def configure(self, K=None, active=(), scale=1.0, perm=None):
        self.reset()
        self.K, self.active, self.scale, self.perm = K, set(active), scale, perm
        self.jq.cap_keys = set(self.active) | (set(self.trace["sites"]) if self.trace else set())

    def mh(self, k, x32):
        return F.linear(x32, self.Q[k].float()) + (x32 @ self.V[k].float()) @ self.dl[k].float().T

    def wh(self, k, x32):
        return self.mh(k, x32) if self.ref_self else F.linear(x32, self.W[k])

    def measure(self, k, h):
        """b = W h - M h, fp32, autocast off, no module call (no hook, no quantizer)."""
        with torch.autocast("cuda", enabled=False):
            h32 = h.float()
            return self.wh(k, h32) - self.mh(k, h32)

    def _hook(self, k):
        def hook(mod, inp, out):
            if self.trace is not None and k in self.trace["sites"]:
                self._trace(k)
            if k not in self.active or self.K is None:
                return None
            i = CTX.step
            if i % self.K == 0:
                bx = self.measure(k, self.jq.h[k])
                self.b[k], self.last[k] = bx.half(), i
                self.n_refresh[k] += 1
                if self.diag is not None:
                    self.bx[k] = bx
            b = self.b[k] if self.perm is None else self.b[k][self.perm]
            y = (out.float() + self.scale * b.float()).to(out.dtype)
            if self.diag is not None:
                self._diag(k, i, out, y)
            if self.debug is not None and i in self.debug["steps"]:
                self.debug["rec"].append({"k": k, "i": i, "y": y.float(), "h": self.jq.h[k].float(), "q": self.jq.q[k].float()})
            return y
        return hook

    # ------------------------------------------------------------------------------------------- E1 trace: eW_t = (M - W) h_t, nothing applied
    def _trace(self, k):
        T = self.trace
        i = CTX.step
        eW = -self.measure(k, self.jq.h[k])
        n = eW.shape[0] // 2
        img = lambda X: (lambda s: s[:n] + s[n:])(X.double().pow(2).sum((1, 2)))
        dot = lambda X, Y: (lambda s: s[:n] + s[n:])((X.double() * Y.double()).sum((1, 2)))
        S = T["stats"].setdefault(k, {})
        acc = lambda name, v: S.__setitem__(name, S.get(name, 0) + v.cpu())
        acc("E_tot", img(eW))
        ring = T["ring"].setdefault(k, deque(maxlen=8))
        for L in (1, 4, 8):
            if len(ring) >= L:
                ea = ring[-L]
                acc(f"L{L}_diff", img(eW - ea)); acc(f"L{L}_Et", img(eW)); acc(f"L{L}_Ea", img(ea)); acc(f"L{L}_cross", dot(eW, ea))
        for K in (4, 8):
            if i % K == 0:
                T["anchor"][(k, K)] = eW
            acc(f"K{K}_res", img(eW - T["anchor"][(k, K)]))
        ring.append(eW)

    # ------------------------------------------------------------------------------------------- section 7 accounting
    def _diag(self, k, i, out, y):
        D = self.diag.setdefault(k, {})
        h, q = self.jq.h[k], self.jq.q[k]
        with torch.autocast("cuda", enabled=False):
            h32, q32 = h.float(), q.float()
            Mh, Mq, Wh = self.mh(k, h32), self.mh(k, q32), self.wh(k, h32)
            eW = Mh - Wh
            a = eW + self.bx[k]                                                # (M - W)(h_t - h_a), since b_exact_a = -(M - W) h_a
            v = Mq - Mh
            bs = self.b[k].float() if self.perm is None else self.b[k][self.perm].float()
            db = bs - self.bx[k]
            lhs = Mq + bs - Wh                                                 # y_concept - (W h + bias), y_concept = M q + bias + b_stored
            res = lhs - (a + v + db)
            ycon = Mq + self.bias[k].float() + bs
            live = y.float() - ycon
        dd = lambda X, Y: float((X.double() * Y.double()).sum())
        rec = {"E[a]": dd(a, a), "E[v]": dd(v, v), "E[db]": dd(db, db), "2<a,v>": 2 * dd(a, v), "2<a,db>": 2 * dd(a, db), "2<v,db>": 2 * dd(v, db),
               "E[lhs]": dd(lhs, lhs), "E[identity_residual]": dd(res, res), "E[eW]": dd(eW, eW), "E[live_defect]": dd(live, live),
               "E[y_concept]": dd(ycon, ycon)}
        for kk, vv in rec.items():
            D[kk] = D.get(kk, 0.0) + vv
        ps = D.setdefault("per_step", {})
        ps.setdefault("E[a]", [0.0] * 50)[i] += rec["E[a]"]
        ps.setdefault("E[v]", [0.0] * 50)[i] += rec["E[v]"]
        ps.setdefault("E[lhs]", [0.0] * 50)[i] += rec["E[lhs]"]

    def remove(self):
        for h in self.hs:
            h.remove()


# ------------------------------------------------------------------------------------------------------------------- models
def ref_weights(fp, keys, dev):
    sm = DJ.smoothing()
    out = {}
    for k in keys:
        W = lin_of(fp, k).weight.detach().float().clone()
        if k in sm:
            W = W * sm[k].to(dev).float()[None, :]                              # same expression as dit_joint.apply_smoothing
        out[k] = W.contiguous()
    return out


def build(dev, mode, wref, sites, mp_sites=()):
    """cheap model (dit_joint.student construction with LocalQuant) + Refresh over `sites`; mp_sites: weights := W, carry hook removed."""
    m = DJ.common_base_nocarry(dev)
    jq = LocalQuant(m, dev, mode)                                              # registered after the smoothing pre-hooks
    jq.t, jq.auto = None, True
    carry = torch.load(DJ.BASE_CARRY, map_location="cpu")
    hs = dict(zip(carry.keys(), install(m, carry)))
    for k in mp_sites:
        lin_of(m, k).weight.data.copy_(wref[k])
        hs[k].remove()
    R = Refresh(m, jq, carry, wref, sites)
    return {"m": m, "jq": jq, "R": R, "carry": carry, "carry_hooks": hs, "mp_sites": list(mp_sites)}


def load_fp(dev):
    fp = load_dit(dev)
    fp.requires_grad_(False)
    return fp


# ------------------------------------------------------------------------------------------------------------------- sampler
ARMDEF = {"Q": dict(model="static"), "G": dict(model="fb", G=True), "A": dict(model="fb"), "FP50": dict(model="fp")}


def arm_spec(name, sites, k=None):
    """arm name -> spec. O1:<site>, O3, OA3: K = 1 oracles; L<K>, LA<K>: held corrections; MP, MPA; C<K>: output anchors on G."""
    if name in ARMDEF:
        return dict(ARMDEF[name])
    if name.startswith("O1:"):
        return dict(model="fb", G=True, K=1, sites=[name[3:]])
    if name == "O3":
        return dict(model="fb", G=True, K=1, sites=sites)
    if name == "OA3":
        return dict(model="fb", K=1, sites=sites)
    if name.startswith("LA"):
        kk = name[2:].split("_")[0]
        return dict(model="fb", K=int(kk), sites=sites, shuffle=name.endswith("_shuf"))
    if name.startswith("L"):
        return dict(model="fb", G=True, K=int(name[1:]), sites=sites)
    if name == "MP":
        return dict(model="mp", G=True)
    if name == "MPA":
        return dict(model="mp")
    if name.startswith("C"):
        return dict(model="fb", G=True, anchor=int(name[1:]))
    raise KeyError(name)


def derangement(n, dev):
    j = (torch.arange(n) + 1) % n
    return torch.cat([j, j + n]).to(dev)


@torch.no_grad()
def sample(spec, models, d, noise, labels, rec_x=None, rec_in=None, stop_at=None, resume=None, scale=1.0):
    """One trajectory. models: {'fp': FP model or None, 'fb'/'static'/'mp': build() dicts}. Returns x, counts, refresh counts."""
    n = len(noise)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    kind = spec["model"]
    fp = models.get("fp")
    B = models.get(kind) if kind != "fp" else None
    jq, R = (B["jq"], B["R"]) if B else (None, None)
    ka = spec.get("anchor")
    if ka is not None or kind == "fp":
        assert fp is not None
    if resume is None:
        x, i0 = noise.float().clone(), 0
        st = {"d": None}
        cnt = {"cheap": 0, "fp": 0}
        if jq is not None:
            jq.state_in, jq.state_out = {}, {}
            jq.zero_state = bool(spec.get("G", False))
            R.configure(spec.get("K"), spec.get("sites", ()), scale, derangement(n, noise.device) if spec.get("shuffle") else None)
    else:
        x, i0 = resume["x"].to(noise.device), resume["i"]
        st = {"d": None if resume["d"] is None else resume["d"].to(noise.device)}
        cnt = dict(resume["cnt"])
        torch.set_rng_state(resume["rng_cpu"]); torch.cuda.set_rng_state(resume["rng_cuda"])
        jq.zero_state = bool(spec.get("G", False))
        R.configure(spec.get("K"), spec.get("sites", ()), scale, derangement(n, noise.device) if spec.get("shuffle") else None)
        jq.state_in = {kk: v.to(noise.device) for kk, v in resume["fb_state"].items()}
        jq.state_out = {}
        R.b = {kk: v.to(noise.device) for kk, v in resume["b"].items()}
        R.last, R.n_refresh = dict(resume["last"]), dict(resume["n_refresh"])
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        if i < i0:
            continue
        if stop_at is not None and i == stop_at:
            return {"x": x.cpu(), "i": i, "d": None if st["d"] is None else st["d"].cpu(), "cnt": dict(cnt),
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "fb_state": {kk: v.detach().clone().cpu() for kk, v in jq.state_in.items()},
                    "b": {kk: v.detach().clone().cpu() for kk, v in R.b.items()}, "last": dict(R.last), "n_refresh": dict(R.n_refresh)}
        if rec_x is not None and i in rec_x:
            rec_x[i] = x.detach().clone()
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            if rec_in is not None and _i in rec_in:
                rec_in[_i] = (xx.detach().clone(), ts.detach().clone())    # the model's actual (respaced) inputs
            if kind == "fp":
                cnt["fp"] += 1
                return guided(fp, xx, ts, y, n)
            cnt["cheap"] += 1
            rc = guided(B["m"], xx, ts, y, n)
            if ka is not None and _i % ka == 0:
                cnt["fp"] += 1
                rf = guided(fp, xx, ts, y, n)
                st["d"] = rc[:, :4] - rf[:, :4]
                return rf
            if ka is not None and st["d"] is not None:
                res = rc.clone()
                res[:, :4] -= st["d"]
                return res
            return rc
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt, (dict(R.n_refresh) if R else {})


def expect(spec):
    if spec["model"] == "fp":
        return {"cheap": 0, "fp": 50}
    ka = spec.get("anchor")
    return {"cheap": 50, "fp": 0 if ka is None else len(range(0, 50, ka))}


def gen(models, d, seeds, bs, arms, sites, t0=None, base=0.0, est_batch_h=0.0, timing=None, tag="ler1"):
    lat, cnt, refr = {a: [] for a in arms}, {}, {}
    for b in range(0, len(seeds), bs):
        s = seeds[b:b + bs]
        if t0 is not None and base + (time.time() - t0) / 3600 + est_batch_h > CAP_GPU_H:
            print(f"[{tag} runtime stop] before batch {s[0]}-{s[-1]}", flush=True)
            return {a: torch.cat(v) for a, v in lat.items() if v}, cnt, refr, False
        nz, lab = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            spec = arm_spec(a, sites)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            x, c, r = sample(spec, models, d, nz, lab)
            torch.cuda.synchronize()
            if timing is not None:
                timing.setdefault(a, []).append(time.perf_counter() - ts)
            assert torch.isfinite(x).all() and c == expect(spec), (a, c)
            if spec.get("K"):
                assert set(r[k] for k in spec["sites"]) == {len(range(0, 50, spec["K"]))}, (a, r)
            lat[a].append(x.cpu())
            cnt[a], refr[a] = c, {k: v for k, v in r.items() if k in spec.get("sites", ())}
            print(f"  seeds {s[0]}-{s[-1]} {a}: {time.perf_counter() - ts:.1f} s", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt, refr, True


def rel(x, y):
    return float((x - y).double().pow(2).sum() / y.double().pow(2).sum().clamp_min(1e-30))


def rl2(x, y):
    return rel(x, y) ** 0.5


# ------------------------------------------------------------------------------------------------------------------- C3 synthetic
def synthetic():
    """fp64 identity tests: y - (W h + bias) = a + v + db, with q from (i) ideal full-state feedback, (ii) compressed state, (iii) input
    clipping; carry nonzero and nonorthogonal; bias nonzero; stored b rounded (fp16)."""
    g = torch.Generator().manual_seed(0)
    out = {}
    for name, (din, dout, T, tok) in {"scalar": (1, 1, 12, 1), "matrix": (6, 5, 12, 8)}.items():
        W = torch.randn(dout, din, generator=g, dtype=torch.float64)
        Qw = (W * 4).round() / 4
        V = torch.randn(din, 2, generator=g, dtype=torch.float64)              # nonorthogonal columns
        V[:, 1] = V[:, 1] + 0.7 * V[:, 0]
        dl = torch.randn(dout, 2, generator=g, dtype=torch.float64) * 0.1
        M = Qw + dl @ V.T
        bias = torch.randn(dout, generator=g, dtype=torch.float64)
        h = torch.cumsum(torch.randn(T, tok, din, generator=g, dtype=torch.float64) * 0.3, 0)
        D, lo, hi = 0.25, -1.0, 1.0
        Pm = torch.zeros(tok, tok, dtype=torch.float64)
        Pm[: tok // 2, : tok // 2] = torch.eye(tok // 2, dtype=torch.float64)    # orthogonal token projection (scalar: state discarded)
        for qname in ("full_state", "compressed_state", "input_clip"):
            r = torch.zeros(tok, din, dtype=torch.float64)
            K, b_st, b_ex, ha = 3, None, None, None
            S = {k_: 0.0 for k_ in ("E[a]", "E[v]", "E[db]", "2<a,v>", "2<a,db>", "2<v,db>", "res", "lhs", "fs_identity_res", "fs_identity_den")}
            for t in range(T):
                x = h[t] * (3.0 if qname == "input_clip" else 1.0)
                xc = x.clamp(lo, hi) if qname == "input_clip" else x
                u = xc + r
                qv = torch.round(u / D) * D
                r_new = u - qv
                if qname == "compressed_state":
                    r_new = Pm @ r_new
                S["fs_identity_res"] += float((qv - x - (r - r_new)).pow(2).sum())
                S["fs_identity_den"] += float((qv - x).pow(2).sum())
                r = r_new
                if t % K == 0:
                    b_ex = x @ W.T - x @ M.T
                    b_st = b_ex.to(torch.float16).double()
                    ha = x
                yv = qv @ M.T + bias + b_st
                lhs = yv - (x @ W.T + bias)
                a = (x - ha) @ (M - W).T
                v = (qv - x) @ M.T
                db = b_st - b_ex
                res = lhs - (a + v + db)
                dd = lambda X, Y: float((X * Y).sum())
                for k_, val in (("E[a]", dd(a, a)), ("E[v]", dd(v, v)), ("E[db]", dd(db, db)), ("2<a,v>", 2 * dd(a, v)), ("2<a,db>", 2 * dd(a, db)),
                                ("2<v,db>", 2 * dd(v, db)), ("res", dd(res, res)), ("lhs", dd(lhs, lhs))):
                    S[k_] += val
            S["identity_rel_residual"] = (S["res"] / S["lhs"]) ** 0.5
            S["full_state_identity_rel_residual"] = (S["fs_identity_res"] / S["fs_identity_den"]) ** 0.5
            S["pass"] = S["identity_rel_residual"] <= C3_TOL and (qname != "full_state" or S["full_state_identity_rel_residual"] <= C3_TOL)
            out[f"{name}/{qname}"] = S
    return out


# ------------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def checks(dev):
    guard("checks", 0.30)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "checks", dev, {"pool": POOL, "G1": [G1[0], G1[-1]]}) as run:
        ch = {"provenance": provenance(), "C3_synthetic_fp64": synthetic()}
        ch["C3_pass"] = all(v["pass"] for v in ch["C3_synthetic_fp64"].values())
        print("  C3", ch["C3_pass"], flush=True)
        fp = load_fp(dev)
        wref = ref_weights(fp, POOL, dev)
        ms = {"fp": fp, "static": build(dev, "static", wref, POOL), "fb": build(dev, "fb", wref, POOL), "mp": build(dev, "fb", wref, POOL, mp_sites=POOL)}
        fbm, R, jq = ms["fb"]["m"], ms["fb"]["R"], ms["fb"]["jq"]
        ch["bias_unchanged"] = all(torch.equal(lin_of(fbm, k).bias, lin_of(fp, k).bias) for k in POOL)
        ch["ref_tensors_not_module_params"] = all(all(p.data_ptr() != wref[k].data_ptr() for p in list(fbm.parameters()) + list(fp.parameters()))
                                                  for k in POOL)
        ch["W_vs_Q_rel"] = {k: rel(lin_of(fbm, k).weight, wref[k]) for k in POOL}
        d = make_diffusion(50)
        # C1 parent reproduction (refresh hooks installed, off)
        L = torch.load(os.path.join(DJ.CK, "eval_latents.pt"), map_location="cpu")
        nz, lab = noise_for_seeds(G1, dev), labels_for_seeds(G1)
        c1 = {}
        for arm, ref in (("Q", L["A"][:32]), ("A", L["B"][:32])):
            x, c, _ = sample(arm_spec(arm, POOL), ms, d, nz, lab)
            c1[arm] = bool(torch.equal(x.cpu(), ref))
            print(f"  C1 {arm}: {c1[arm]}", flush=True)
        ch["C1_parent_bitwise"] = c1
        # small batch for the remaining mechanics
        S4 = SEL[:4]
        nz, lab = noise_for_seeds(S4, dev), labels_for_seeds(S4)
        labt = torch.as_tensor(lab, device=dev)
        yy = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        n = len(S4)
        refl = {}
        for arm in ("G", "A", "L4", "LA4"):
            refl[arm], c, r = sample(arm_spec(arm, POOL), ms, d, nz, lab)
            refl[arm] = refl[arm].cpu()
        # C1b: zero-state A == G for one forward from a fixed nonzero-step state; G vs A at step 0
        xs = torch.randn(n, 4, 32, 32, generator=torch.Generator().manual_seed(1), device="cpu").to(dev)
        ts = torch.full((n,), 500, device=dev, dtype=torch.long)
        CTX.step, CTX.n = 20, n
        R.configure(None)
        jq.zero_state = False
        jq.state_in = {k: torch.zeros(2 * n, jq.U.shape[1], (4608 if k.endswith("fc2") else 1152), device=dev, dtype=torch.float16) for k in DJ.KEYS}
        oa = guided(fbm, xs, ts, yy, n)
        jq.zero_state, jq.state_in = True, {k: torch.randn_like(v) for k, v in jq.state_in.items()}   # nonzero junk: must be ignored
        og = guided(fbm, xs, ts, yy, n)
        ch["C1_zero_state_A_equals_G"] = bool(torch.equal(oa, og))
        rec_g, rec_a = {0: None, 1: None}, {0: None, 1: None}
        sample(arm_spec("G", POOL), ms, d, nz, lab, rec_x=rec_g)
        sample(arm_spec("A", POOL), ms, d, nz, lab, rec_x=rec_a)
        ch["C1_G_A_equal_after_step0"] = bool(torch.equal(rec_g[1], rec_a[1]))
        # C2 capture point
        probe = {}
        k2 = [k for k in POOL if "mlp" in k]
        sm = DJ.smoothing()
        hp = [lin_of(fbm, k).register_forward_pre_hook((lambda kk: lambda mod, inp: probe.__setitem__(("raw", kk), inp[0].detach().clone()))(k), prepend=True)
              for k in k2]
        hc = [lin_of(fbm, k).register_forward_hook((lambda kk: lambda mod, inp, out: probe.__setitem__(("gemm_in", kk), inp[0].detach().clone()))(k), prepend=True)
              for k in POOL]
        R.configure(None)
        jq.cap_keys = set(POOL)
        jq.zero_state = True
        CTX.step = 20
        guided(fbm, xs, ts, yy, n)
        for hh in hp + hc:
            hh.remove()
        ch["C2_h_times_s_equals_raw_rel_l2"] = {k: {"v": rl2(jq.h[k].float() * sm[k].to(dev).float(), probe[("raw", k)].float()), "dtype": str(jq.h[k].dtype),
                                                    "tol": C2_TOL if jq.h[k].dtype == torch.float32 else C2_TOL_FP16} for k in k2}
        ch["C2_q_equals_gemm_input"] = all(torch.equal(jq.q[k], probe[("gemm_in", k)]) for k in POOL)
        ch["C2_pass"] = all(v["v"] <= v["tol"] for v in ch["C2_h_times_s_equals_raw_rel_l2"].values()) and ch["C2_q_equals_gemm_input"]
        # C4 helper vs module computation
        c4, c4live = {}, {}
        for k in POOL:
            h = jq.h[k]
            lin = lin_of(fbm, k)
            hook = ms["fb"]["carry_hooks"][k]
            fn = lin._forward_hooks[hook.id]
            with torch.autocast("cuda", enabled=False):
                mod = fn(lin, (h.float(),), F.linear(h.float(), lin.weight))         # bias-free GEMM (amendment: fused-bias addmm differs by ~1e-6)
                helper = R.mh(k, h.float())
            c4[k] = rl2(helper, mod)
            with torch.autocast("cuda", dtype=torch.float16):
                hh_ = h.half()
                live = fn(lin, (hh_,), lin.forward(hh_)).float() - lin.bias.float()
            c4live[k] = rl2(live, helper)
        ch["C4_helper_vs_module_fp32_rel"], ch["C4_live_autocast_defect_rel"] = c4, c4live
        ch["C4_pass"] = all(v <= C4_TOL for v in c4.values())
        # C7a: measurement helper touches no counter
        before = dict(jq.calls)
        for k in POOL:
            R.measure(k, jq.h[k])
        ch["C7_measure_no_quantizer_call"] = jq.calls == before
        # C5 K = 1: live corrected output vs fp32 W h + M(q - h) + bias
        R.debug = {"steps": {0, 25, 49}, "rec": []}
        sample(arm_spec("O3", POOL), ms, d, nz[:2], lab[:2])
        c5 = {}
        for r_ in R.debug["rec"]:
            k = r_["k"]
            with torch.autocast("cuda", enabled=False):
                tgt = R.wh(k, r_["h"]) + (R.mh(k, r_["q"]) - R.mh(k, r_["h"])) + R.bias[k].float()
            c5[f"{k}@{r_['i']}"] = rl2(r_["y"], tgt)
        R.debug = None
        ch["C5_K1_rel_l2"] = c5
        ch["C5_K1_max_rel_l2"] = max(c5.values())
        ch["C5_K1_pass"] = ch["C5_K1_max_rel_l2"] <= C5_TOL and len(c5) == 3 * len(POOL)
        R.ref_self = True
        bz = [bool(torch.count_nonzero(R.measure(k, jq.h[k])) == 0) for k in POOL]
        xg, _, _ = sample(arm_spec("L4", POOL), ms, d, nz, lab)
        R.ref_self = False
        ch["C5_M_equals_W_zero_b"] = all(bz)
        ch["C5_M_equals_W_trajectory_equals_G"] = bool(torch.equal(xg.cpu(), refl["G"]))
        # C6 scale-0 correction == G / A; MP structure and numerics
        c6 = {}
        for arm, pl in (("L4", "G"), ("LA4", "A")):
            x, _, _ = sample(arm_spec(arm, POOL), ms, d, nz, lab, scale=0.0)
            c6[arm] = bool(torch.equal(x.cpu(), refl[pl]))
        ch["C6_zero_b_bitwise"] = c6
        mpm = ms["mp"]["m"]
        carry_ids = {k: ms["mp"]["carry_hooks"][k].id for k in DJ.KEYS}
        ch["C6_MP_weights_equal_W"] = all(torch.equal(lin_of(mpm, k).weight, wref[k]) for k in POOL)
        ch["C6_MP_carry_removed_selected"] = all(carry_ids[k] not in lin_of(mpm, k)._forward_hooks for k in POOL)
        ch["C6_MP_carry_present_elsewhere"] = all(carry_ids[k] in lin_of(mpm, k)._forward_hooks for k in DJ.KEYS if k not in POOL)
        mo = {}
        hmo = [lin_of(mpm, k).register_forward_hook((lambda kk: lambda mod, inp, out: mo.__setitem__(kk, (inp[0].detach().clone(), out.detach().clone())))(k))
               for k in POOL]
        ms["mp"]["jq"].zero_state = True
        ms["mp"]["R"].configure(None)
        CTX.step = 20
        guided(mpm, xs, ts, yy, n)
        for hh in hmo:
            hh.remove()
        with torch.autocast("cuda", dtype=torch.float16):
            ch["C6_MP_output_equals_Wq_bitwise"] = all(torch.equal(F.linear(mo[k][0], wref[k], lin_of(mpm, k).bias), mo[k][1]) for k in POOL)
        # C7 quantizer updates and refresh counts; C9 FP calls
        c7 = {}
        for arm in ("G", "A", "L4", "LA4", "L8"):
            jq.calls = {k: 0 for k in DJ.KEYS}
            x, c, r = sample(arm_spec(arm, POOL), ms, d, nz, lab)
            c7[arm] = {"calls_per_site": sorted(set(jq.calls.values())), "counts": c, "refresh": sorted(set(r[k] for k in POOL)) if arm[0] == "L" else None}
            if arm in ("L4", "LA4"):
                c7[arm]["bitwise_rerun"] = bool(torch.equal(x.cpu(), refl[arm]))
        ch["C7_C9"] = c7
        ch["C7_C9_pass"] = all(v["calls_per_site"] == [50] and v["counts"] == {"cheap": 50, "fp": 0} for v in c7.values()) and \
            c7["L4"]["refresh"] == [13] and c7["LA4"]["refresh"] == [13] and c7["L8"]["refresh"] == [7] and c7["L4"]["bitwise_rerun"] and c7["LA4"]["bitwise_rerun"]
        # C8 save / resume (LA4)
        s30 = sample(arm_spec("LA4", POOL), ms, d, nz, lab, stop_at=30)
        p = os.path.join(CK, "resume_test.pt")
        atomic_torch_save(s30, p)
        loaded = torch.load(p, map_location="cpu")
        x, c, r = sample(arm_spec("LA4", POOL), ms, d, nz, lab, resume=loaded)
        ch["C8_save_resume_bitwise"] = bool(torch.equal(x.cpu(), refl["LA4"]))
        ch["C8_saved_fields"] = sorted(loaded.keys())
        ch["C8_refresh_after_resume"] = sorted(set(r[k] for k in POOL))
        # C10 profile (batch 16, dev seeds, diagnostics off)
        nz, lab = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        prof = {}
        for arm in ("FP50", "G", "A", "L4", "O3", "C4", "MP"):
            torch.cuda.synchronize()
            t_ = time.perf_counter()
            sample(arm_spec(arm, POOL), ms, d, nz, lab)
            torch.cuda.synchronize()
            prof[arm] = time.perf_counter() - t_
            print(f"  profile {arm}: {prof[arm]:.1f} s / 16 images", flush=True)
        ch["C10_profile_s_per_16_images_12_sites"] = prof
        ch["peak_mem"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        flat = [ch["bias_unchanged"], ch["ref_tensors_not_module_params"], ch["C3_pass"], *c1.values(), ch["C1_zero_state_A_equals_G"], ch["C1_G_A_equal_after_step0"],
                ch["C2_pass"], ch["C4_pass"], ch["C7_measure_no_quantizer_call"], ch["C5_K1_pass"], ch["C5_M_equals_W_zero_b"], ch["C5_M_equals_W_trajectory_equals_G"],
                *c6.values(), ch["C6_MP_weights_equal_W"], ch["C6_MP_carry_removed_selected"], ch["C6_MP_carry_present_elsewhere"], ch["C6_MP_output_equals_Wq_bitwise"],
                ch["C7_C9_pass"], ch["C8_save_resume_bitwise"]]
        ch["pass"] = bool(all(flat))
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k_: v for k_, v in ch.items() if k_ not in ("config", "provenance", "C3_synthetic_fp64", "C5_K1_rel_l2")}, indent=1, default=str), flush=True)
    if not ch["pass"]:
        print("[ler1] checks FAILED: stop", flush=True)
        sys.exit(6)


def need_checks():
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        print("[ler1] checks must pass first", flush=True)
        sys.exit(6)


# ------------------------------------------------------------------------------------------------------------------- E0 screen
@torch.no_grad()
def screen(dev):
    need_checks()
    guard("screen", 0.05)
    with Run(STAGE, "screen", dev, {"seeds": [SEL[0], SEL[-1]], "steps": SCREEN_STEPS, "pool": POOL}) as run:
        provenance()
        fp = load_fp(dev)
        wref = ref_weights(fp, POOL, dev)
        ms = {"fp": fp, "fb": build(dev, "fb", wref, POOL)}
        B = ms["fb"]
        d = make_diffusion(50)
        nz, lab = noise_for_seeds(SEL, dev), labels_for_seeds(SEL)
        n = len(SEL)
        labt = torch.as_tensor(lab, device=dev)
        yy = torch.cat([labt, torch.full_like(labt, NULL_CLASS)])
        recin = {s: None for s in SCREEN_STEPS}
        xg, _, _ = sample(arm_spec("G", POOL), ms, d, nz, lab, rec_in=recin)
        sse = lambda a, b: float((a[:, :4] - b[:, :4]).double().pow(2).sum())
        base, scores, per = 0.0, {k: 0.0 for k in POOL}, {}
        B["jq"].zero_state = True
        for i in SCREEN_STEPS:
            x, ts = recin[i]
            CTX.step, CTX.n = i, n
            ref = guided(fp, x, ts, yy, n)
            B["R"].configure(None)
            g = guided(B["m"], x, ts, yy, n)
            e0 = sse(g, ref)
            base += e0
            per[i] = {"G_sse": e0}
            for k in POOL:
                B["R"].configure(1, [k])
                gk = guided(B["m"], x, ts, yy, n)
                assert B["R"].n_refresh[k] == 1
                red = e0 - sse(gk, ref)
                scores[k] += red
                per[i][k] = red
        CTX.step = None
        order = sorted(POOL, key=lambda k: (-scores[k], POOL.index(k)))
        sel, blocks = [], set()
        for k in order:
            if k.split(".")[0] not in blocks:
                sel.append(k)
                blocks.add(k.split(".")[0])
            if len(sel) == 3:
                break
        out = {"pool": POOL, "seeds": [SEL[0], SEL[-1]], "steps": SCREEN_STEPS, "G_sse_total": base, "score_abs_reduction": scores,
               "score_rel_reduction": {k: v / base for k, v in scores.items()}, "per_step": per, "ranking": order, "selected": sel,
               "rule": "three highest absolute reductions of guided-prediction SSE (channels 0-3), at most one per block, ties by pool order; top three within this pool",
               "note": "fresh K = 1 correction at one site, same latent, no-feedback G, FP reference computed once per state; intervention ranking, not an endpoint oracle",
               **run.record()}
    jdump("selection.json", out)
    for k in order:
        print(f"  {k}: {scores[k]:+.4g} ({scores[k] / base:+.2%})", flush=True)
    print("  selected:", sel, flush=True)


def sites_sel():
    s = jload("selection.json")
    if not s:
        print("[ler1] selection.json required", flush=True)
        sys.exit(6)
    return s["selected"]


# ------------------------------------------------------------------------------------------------------------------- statistics
def interaction(Mx, l="L", la="LA", g="G", a="A"):
    L, LA, G, A = (np.asarray(Mx[x]["rel_mse"]) for x in (l, la, g, a))
    idx = boot_idx(len(L))
    I = lambda i: (L[i].mean() - LA[i].mean()) - (G[i].mean() - A[i].mean())
    al = np.arange(len(L))
    return {"I_abs": float(I(al)), "I_abs_ci": ci([I(i) for i in idx]), "feedback_gain_with_local": float(L.mean() - LA.mean()),
            "feedback_gain_with_local_ci": ci([L[i].mean() - LA[i].mean() for i in idx]), "feedback_gain_without": float(G.mean() - A.mean()),
            "feedback_gain_without_ci": ci([G[i].mean() - A[i].mean() for i in idx]),
            "note": "I = [E(L) - E(LA)] - [E(G) - E(A)]; I > 0: feedback's absolute benefit grows with local correction"}


def reading(c):
    m, lp = c["rel_mse"], c["lpips"]
    if m["ratio"] <= 0.95 and m["ratio_ci"][1] < 1 and not lp["diff_ci"][0] > 0:
        return "useful (>= 5% lower mean latent MSE, CI excludes no improvement, no clear LPIPS regression)"
    if m["ratio_ci"][1] < 1:
        return "smaller positive" + (" (LPIPS clearly worse)" if lp["diff_ci"][0] > 0 else " (below the 5% threshold)")
    if m["ratio_ci"][0] > 1:
        return "harmful"
    return "inconclusive"


def show(Mx, C):
    for a, v in summary(Mx).items():
        print(f"  {a}: rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f} PSNR {v['psnr']['mean']:.2f} LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in C.items():
        m = v["rel_mse"]
        print(f"  {k_}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] wins {m['wins_x']}/{m['n']} "
              f"dLPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['diff_ci'][0]:+.4f}, {v['lpips']['diff_ci'][1]:+.4f}]", flush=True)


def mean_rel(Mx, a):
    return float(np.mean(Mx[a]["rel_mse"]))


# ------------------------------------------------------------------------------------------------------------------- E1
@torch.no_grad()
def e1(dev):
    need_checks()
    sites = sites_sel()
    guard("e1", 0.25)
    with Run(STAGE, "e1", dev, {"sites": sites, "trace_seeds": [DIAG[0], DIAG[-1]], "live_seeds": [DEV[0], DEV[-1]]}) as run:
        provenance()
        fp = load_fp(dev)
        wref = ref_weights(fp, sites, dev)
        ms = {"fp": fp, "fb": build(dev, "fb", wref, sites), "static": build(dev, "static", wref, sites)}
        d = make_diffusion(50)
        # traces
        R = ms["fb"]["R"]
        nz, lab = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        tr = {}
        for arm in ("G", "A"):
            R.trace = {"sites": list(sites), "stats": {}, "ring": {}, "anchor": {}}
            x, c, _ = sample(arm_spec(arm, sites), ms, d, nz, lab)
            S = R.trace["stats"]
            R.trace = None
            res = {}
            for k in list(sites) + ["pooled"]:
                st = {n_: sum(S[s][n_] for s in sites) for n_ in S[sites[0]]} if k == "pooled" else S[k]
                st = {n_: v.numpy() for n_, v in st.items()}
                r = {}
                for L in (1, 4, 8):
                    Et, Ea, X = st[f"L{L}_Et"].sum(), st[f"L{L}_Ea"].sum(), st[f"L{L}_cross"].sum()
                    r[f"lag{L}"] = {"diff_over_Et": ratio_ci(st[f"L{L}_diff"], st[f"L{L}_Et"]), "Et_over_Ea": float(Et / Ea), "cross": float(X),
                                    "uncentered_corr": float(X / np.sqrt(Et * Ea)), "E_t": float(Et), "E_a": float(Ea),
                                    "identity_residual_rel": float(abs(st[f"L{L}_diff"].sum() - (Et + Ea - 2 * X)) / st[f"L{L}_diff"].sum())}
                for K in (4, 8):
                    r[f"periodic_K{K}_residual_over_total"] = ratio_ci(st[f"K{K}_res"], st["E_tot"])
                r["E_total"] = float(st["E_tot"].sum())
                res[k] = r
            tr[arm] = res
            print(f"  trace {arm}: pooled lag1 {res['pooled']['lag1']['diff_over_Et']['v']:.3f} lag4 {res['pooled']['lag4']['diff_over_Et']['v']:.3f} "
                  f"lag8 {res['pooled']['lag8']['diff_over_Et']['v']:.3f}; K4 {res['pooled']['periodic_K4_residual_over_total']['v']:.3f} "
                  f"K8 {res['pooled']['periodic_K8_residual_over_total']['v']:.3f}", flush=True)
        jdump("trace.json", {"sites": sites, "seeds": [DIAG[0], DIAG[-1]], "eW": "(M - W) h_t at the selected sites, nothing applied, own G / A trajectories",
                             "pooling": "sums over all pairs, rows (cond+uncond), tokens, channels and images; per-image sums for the 2,000-draw bootstrap",
                             "arms": tr})
        # live
        nz, lab = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        arms = ["FP50", "G", "A", "Q"] + [f"O1:{k}" for k in sites] + ["O3", "OA3"]
        lat, cnt, refr, _ = gen(ms, d, DEV, 16, arms, sites)
        atomic_torch_save(lat, os.path.join(CK, "dev_e1_latents.pt"))
        del ms, fp
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        pairs = [("A", "G"), ("G", "Q"), ("O3", "G"), ("OA3", "A"), ("OA3", "O3")] + [(f"O1:{k}", "G") for k in sites]
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in pairs}
        r_o3, r_oa3 = mean_rel(Mx, "O3") / mean_rel(Mx, "G"), mean_rel(Mx, "OA3") / mean_rel(Mx, "A")
        go = r_o3 <= 0.95 or r_oa3 <= 0.95
        out = {"sites": sites, "seeds": [DEV[0], DEV[-1]], "arms": Mx, "summary": summary(Mx), "calls": cnt, "refresh": refr, "contrasts": C,
               "headroom": {"O3_over_G": r_o3, "OA3_over_A": r_oa3, "proceed": go,
                            "rule": "proceed iff E(O3)/E(G) <= 0.95 or E(OA3)/E(A) <= 0.95 (mean endpoint latent rel. MSE); practical screen, not a test"},
               **run.record()}
    jdump("dev_e1.json", out)
    show(Mx, C)
    print(f"  headroom: O3/G {r_o3:.3f}, OA3/A {r_oa3:.3f} -> {'PROCEED' if go else 'STOP'}", flush=True)


# ------------------------------------------------------------------------------------------------------------------- E2
@torch.no_grad()
def e2(dev):
    need_checks()
    sites = sites_sel()
    E1 = jload("dev_e1.json")
    if not (E1 and E1["headroom"]["proceed"]):
        print("[ler1] E1 headroom rule did not pass: stop", flush=True)
        sys.exit(6)
    guard("e2", 0.30)
    with Run(STAGE, "e2", dev, {"sites": sites, "seeds": [DEV[0], DEV[-1]]}) as run:
        provenance()
        fp = load_fp(dev)
        wref = ref_weights(fp, sites, dev)
        ms = {"fp": fp, "fb": build(dev, "fb", wref, sites), "mp": build(dev, "fb", wref, sites, mp_sites=sites)}
        d = make_diffusion(50)
        e1lat = torch.load(os.path.join(CK, "dev_e1_latents.pt"), map_location="cpu")
        arms = ["L4", "L8", "LA4", "LA8", "MP", "MPA", "C4", "C8"]
        lat, cnt, refr, _ = gen(ms, d, DEV, 16, arms, sites)
        for a in ("FP50", "G", "A", "Q"):
            lat[a] = e1lat[a]
        Mx = metrics({a: lat[a] for a in ["FP50", "G", "A", "Q"] + arms}, dev)
        e = lambda a: mean_rel(Mx, a)
        lp = lambda a: float(np.mean(Mx[a]["lpips"]))
        kstar = 8 if (e("LA8") <= 1.05 * e("LA4") and lp("LA8") <= lp("LA4") + 0.005) else 4
        stop = not (e("LA4") <= 0.95 * e("G") or e("LA8") <= 0.95 * e("G"))
        shuf = None
        if not stop:
            sa = f"LA{kstar}_shuf"
            l2, c2, r2, _ = gen(ms, d, DEV, 16, [sa], sites)
            lat[sa] = l2[sa]
            cnt.update(c2); refr.update(r2)
            M2 = metrics({"FP50": lat["FP50"], sa: lat[sa]}, dev)
            Mx[sa] = M2[sa]
            g_gen, g_shuf = e("A") - e(f"LA{kstar}"), e("A") - e(sa)
            labs = labels_for_seeds(DEV)
            pm = derangement(16, "cpu")[:16].tolist()
            shuf = {"arm": sa, "genuine_gain_over_A": g_gen, "shuffled_gain_over_A": g_shuf,
                    "recovered_fraction": (g_shuf / g_gen) if g_gen > 0 else None,
                    "mean_bias_control_triggered": bool(g_gen > 0 and g_shuf >= 0.8 * g_gen),
                    "derangement": "image j uses the correction measured on image (j+1) mod 16 (cond->cond, uncond->uncond)",
                    "label_repeats": int(sum(labs[j] == labs[pm[j]] for j in range(16)))}
        atomic_torch_save(lat, os.path.join(CK, "dev_e2_latents.pt"))
        del ms, fp
        torch.cuda.empty_cache()
        pairs = [("LA4", "L4"), ("LA8", "L8"), ("L4", "G"), ("L8", "G"), ("LA4", "A"), ("LA8", "A"), ("LA4", "G"), ("LA8", "G"), ("L4", "MP"), ("LA4", "MPA"),
                 ("L8", "MP"), ("LA8", "MPA"), ("LA4", "C4"), ("LA8", "C8"), ("MP", "G"), ("MPA", "A"), ("C4", "G"), ("G", "Q")]
        if shuf:
            pairs += [(shuf["arm"], f"LA{kstar}"), (shuf["arm"], "A")]
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in pairs}
        out = {"sites": sites, "seeds": [DEV[0], DEV[-1]], "arms": Mx, "summary": summary(Mx), "calls": cnt, "refresh": refr, "contrasts": C,
               "K_rule": {"K_star": kstar, "E_LA8_over_E_LA4": e("LA8") / e("LA4"), "LPIPS_LA8_minus_LA4": lp("LA8") - lp("LA4"),
                          "rule": "K8 if E(LA8) <= 1.05 E(LA4) and LPIPS(LA8) <= LPIPS(LA4) + 0.005, else K4 (exploratory accuracy rule)"},
               "stop_rule": {"E_LA4_over_G": e("LA4") / e("G"), "E_LA8_over_G": e("LA8") / e("G"), "stop_before_confirmation": stop},
               "shuffle": shuf, "interaction_K4": interaction(Mx, "L4", "LA4"), "interaction_K8": interaction(Mx, "L8", "LA8"),
               "note": "G, A, Q, FP50 reused from E1 (same seeds, batch 16, code, precision)", **run.record()}
    jdump("dev_e2.json", out)
    show(Mx, C)
    print(f"  K* = {kstar}; stop before confirmation: {stop}; shuffle: {json.dumps(shuf)}", flush=True)


# ------------------------------------------------------------------------------------------------------------------- diag
@torch.no_grad()
def diag(dev):
    need_checks()
    sites = sites_sel()
    E2 = jload("dev_e2.json")
    kstar = E2["K_rule"]["K_star"]
    guard("diag", 0.08)
    with Run(STAGE, "diag", dev, {"sites": sites, "K": kstar, "seeds": [DIAG[0], DIAG[-1]]}) as run:
        provenance()
        fp = load_fp(dev)
        wref = ref_weights(fp, sites, dev)
        del fp
        torch.cuda.empty_cache()
        ms = {"fb": build(dev, "fb", wref, sites)}
        R, jq = ms["fb"]["R"], ms["fb"]["jq"]
        d = make_diffusion(50)
        nz, lab = noise_for_seeds(DIAG, dev), labels_for_seeds(DIAG)
        out = {"sites": sites, "K": kstar, "seeds": [DIAG[0], DIAG[-1]], "arms": {},
               "note": "layer-local accounting relative to W h on each arm's own current input; a and v can cancel or reinforce; not an endpoint "
                       "decomposition; cross-arm differences include trajectory changes"}
        for arm in (f"L{kstar}", f"LA{kstar}"):
            R.diag, jq.acct = {}, {}
            x, c, r = sample(arm_spec(arm, sites), ms, d, nz, lab)
            D, Acc = R.diag, jq.acct
            R.diag, jq.acct = None, None
            layer = {}
            for k in sites:
                v = D[k]
                layer[k] = {**{kk: vv for kk, vv in v.items() if kk != "per_step"}, "per_step": v["per_step"],
                            "identity_rel_residual": (v["E[identity_residual]"] / v["E[lhs]"]) ** 0.5,
                            "live_defect_rel": (v["E[live_defect]"] / v["E[y_concept]"]) ** 0.5,
                            "check_sum": (v["E[a]"] + v["E[v]"] + v["E[db]"] + v["2<a,v>"] + v["2<a,db>"] + v["2<v,db>"]) / v["E[lhs]"]}
            grp = {}
            for key, a in Acc.items():
                gname = "selected:" + key if key in sites else "type:" + key.split(".", 1)[1]
                G_ = grp.setdefault(gname, {})
                for kk, vv in a.items():
                    G_[kk] = G_.get(kk, 0.0) + vv
                T_ = grp.setdefault("all_112_sites", {})
                for kk, vv in a.items():
                    T_[kk] = T_.get(kk, 0.0) + vv
            for G_ in grp.values():
                for kk in [x_ for x_ in G_ if x_.startswith("rate_")]:
                    G_[kk] = G_[kk] / G_["n"]
                G_["identity_rel_residual"] = (G_["E[identity_residual]"] / G_["E[q-x]"]) ** 0.5
            out["arms"][arm] = {"layer": layer, "feedback_accounting": grp, "calls": c, "refresh": r}
            print(f"  {arm}: " + "; ".join(f"{k} E[a]/E[lhs] {layer[k]['E[a]'] / layer[k]['E[lhs]']:.3f} E[v]/E[lhs] {layer[k]['E[v]'] / layer[k]['E[lhs]']:.3f} "
                                           f"2<a,v>/E[lhs] {layer[k]['2<a,v>'] / layer[k]['E[lhs]']:+.3f} res {layer[k]['identity_rel_residual']:.1e}" for k in sites), flush=True)
        out.update(run.record())
    jdump("diag.json", out)


# ------------------------------------------------------------------------------------------------------------------- conf
CONF_PAIRS = (("LA", "L"), ("L", "G"), ("LA", "A"), ("LA", "G"), ("L", "MP"), ("LA", "MPA"), ("LA", "C"), ("A", "G"), ("MP", "G"), ("MPA", "A"), ("C", "G"))


@torch.no_grad()
def conf(dev):
    fr = jload("frozen.json")
    if not (fr and jload("dev_e2.json")) or fr.get("stop_before_confirmation"):
        print("[ler1] committed frozen.json (not stopped) required", flush=True)
        sys.exit(6)
    sites, kstar = fr["sites"], fr["K_star"]
    guard("conf", CONF_RESERVE_H, reserve_h=0.0)
    t0, base = time.time(), spent()
    names = {"G": "G", "A": "A", "L": f"L{kstar}", "LA": f"LA{kstar}", "MP": "MP", "MPA": "MPA", "C": f"C{kstar}", "FP50": "FP50"}
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "batch": BS, "sites": sites, "K": kstar}) as run:
        provenance()
        fp = load_fp(dev)
        wref = ref_weights(fp, sites, dev)
        ms = {"fp": fp, "fb": build(dev, "fb", wref, sites), "mp": build(dev, "fb", wref, sites, mp_sites=sites)}
        d = make_diffusion(50)
        torch.cuda.reset_peak_memory_stats()
        timing = {}
        lat, cnt, refr, full = gen(ms, d, CONF, BS, list(names.values()), sites, t0=t0, base=base, est_batch_h=0.2, timing=timing)
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt" if full else "conf_latents_partial.pt"))
        if not full:
            jdump("conf_partial.json", {"calls": cnt, **run.record()})
            sys.exit(5)
        del ms, fp
        torch.cuda.empty_cache()
        lat = {k: lat[v] for k, v in names.items()}
        Mx = metrics(lat, dev)
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in CONF_PAIRS}
        out = {"seeds": [CONF[0], CONF[-1]], "sites": sites, "K_star": kstar, "arm_names": names, "arms": Mx, "summary": summary(Mx),
               "calls": {k: cnt[v] for k, v in names.items()}, "refresh": {k: refr[v] for k, v in names.items()}, "contrasts": C,
               "interaction": interaction(Mx), "reading_integration_LA_vs_L": reading(C["LA_vs_L"]), "reading_local_L_vs_G": reading(C["L_vs_G"]),
               "reading_LA_vs_A": reading(C["LA_vs_A"]), "reading_L_vs_MP": reading(C["L_vs_MP"]), "reading_LA_vs_MPA": reading(C["LA_vs_MPA"]),
               "reading_LA_vs_C": reading(C["LA_vs_C"]),
               "timing_s_per_batch32": {k: {"median": float(np.median(timing[v])), "all": timing[v]} for k, v in names.items()},
               "timing_note": "CUDA-synchronized wall per trajectory during confirmation, FP + two cheap instances resident; simulation (fake-quant, fp32 weights); "
                              "see cost.json for isolated timing", "peak_mem_live": peak, **run.record()}
    jdump("conf.json", out)
    show(Mx, C)
    print("  interaction:", json.dumps(out["interaction"]), flush=True)
    for k_ in [x for x in out if x.startswith("reading")]:
        print(f"  {k_}: {out[k_]}", flush=True)


# ------------------------------------------------------------------------------------------------------------------- cost
def tensor_bytes(t):
    return int(t.numel() * t.element_size())


@torch.no_grad()
def cost(dev):
    fr = jload("frozen.json")
    sites, kstar = fr["sites"], fr["K_star"]
    prof = jload("conf.json")["timing_s_per_batch32"]
    reps = 6
    est = sum(v["median"] for v in prof.values()) / 32 * 8 * reps / 3600 * 1.6 + 0.03     # batch 8 is less efficient per image
    bs = 8
    if spent() + est > CAP_GPU_H:
        est, bs = est / 1.8, 4
    guard(f"cost (batch {bs})", est, reserve_h=0.0)
    seeds = DEV[:bs]
    names = {"G": "G", "A": "A", "L": f"L{kstar}", "LA": f"LA{kstar}", "MP": "MP", "MPA": "MPA", "C": f"C{kstar}", "FP50": "FP50"}
    with Run(STAGE, "cost", dev, {"batch": bs, "repeats": "1 warm-up + 5", "sites": sites, "K": kstar}) as run:
        provenance()
        d = make_diffusion(50)
        nz, lab = noise_for_seeds(seeds, dev), labels_for_seeds(seeds)
        out = {"batch": bs, "seeds": [seeds[0], seeds[-1]], "sites": sites, "K_star": kstar, "arms": {}}
        fp0 = load_fp(dev)
        wref = {k: v.cpu() for k, v in ref_weights(fp0, sites, dev).items()}
        del fp0
        for key, arm in names.items():
            spec = arm_spec(arm, sites)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            m0 = torch.cuda.memory_allocated()
            torch.cuda.synchronize()
            tl = time.perf_counter()
            ms = {}
            wr = {k: v.to(dev) for k, v in wref.items()} if spec["model"] != "fp" else None
            if spec["model"] == "fp" or spec.get("anchor"):
                ms["fp"] = load_fp(dev)
            if spec["model"] == "fb":
                ms["fb"] = build(dev, "fb", wr, sites if spec.get("K") else [])
            if spec["model"] == "mp":
                ms["mp"] = build(dev, "fb", wr, [], mp_sites=sites)
            wr = None                                                          # Refresh keeps only the selected W it needs
            torch.cuda.synchronize()
            load_s = time.perf_counter() - tl
            resident = torch.cuda.memory_allocated() - m0
            ts = []
            for rep in range(reps):
                torch.cuda.synchronize()
                t_ = time.perf_counter()
                x, c, r = sample(spec, ms, d, nz, lab)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t_)
            peak = torch.cuda.max_memory_allocated() - m0
            sb = 0
            if spec["model"] in ("fb", "mp"):
                B = ms[spec["model"]]
                sb_fb = 0 if spec.get("G") else B["jq"].state_bytes()
                sb_b = sum(tensor_bytes(v) for v in B["R"].b.values())
                sb = {"feedback_state_bytes_per_image_incl_cfg": sb_fb / bs, "held_correction_bytes_per_image_incl_cfg": sb_b / bs}
            out["arms"][key] = {"arm": arm, "load_s": load_s, "timed_s": ts[1:], "warmup_s": ts[0], "median_s": float(np.median(ts[1:])),
                                "per_image_s": float(np.median(ts[1:])) / bs, "resident_bytes_after_load": int(resident), "peak_bytes_during_sampling": int(peak),
                                "state": sb, "calls": c}
            print(f"  {key} ({arm}): load {load_s:.1f} s, median {np.median(ts[1:]):.2f} s / {bs} images, peak {peak / 1e9:.2f} GB", flush=True)
            del ms
        vae = load_vae(dev)
        torch.cuda.synchronize()
        t_ = time.perf_counter()
        for _ in range(3):
            decode(vae, x.to(dev))
        torch.cuda.synchronize()
        out["decode_s_per_batch"] = (time.perf_counter() - t_) / 3
        # analytic storage (deployable formats are not built in this pilot)
        an = {}
        for k in sites:
            o_, i_ = wref[k].shape
            an[k] = {"shape": [o_, i_], "W_ref_fp16_bytes": o_ * i_ * 2, "W4_packed_codes_bytes": o_ * i_ // 2, "carry_fp16_bytes": (i_ * 8 + o_ * 8) * 2,
                     "held_b_bytes_per_image_incl_cfg_fp16": 2 * 256 * o_ * 2,
                     "refresh_macs_per_token_per_row": {"W_h": o_ * i_, "M_h_dense_Q": o_ * i_, "M_h_carry": 8 * (i_ + o_)},
                     "cheap_linear_macs_per_token_per_row": o_ * i_ + 8 * (i_ + o_)}
        out["analytic"] = {"per_site": an, "refreshes_per_trajectory": len(range(0, 50, kstar)),
                           "local_extra_weight_bytes_fp16": sum(v["W_ref_fp16_bytes"] for v in an.values()),
                           "MP_optimized_delta_bytes": sum(v["W_ref_fp16_bytes"] - v["W4_packed_codes_bytes"] - v["carry_fp16_bytes"] for v in an.values()),
                           "note": "W4 packed size excludes scales/zeros; simulation stores fp32 fake-quant weights, so measured memory is not deployable memory. "
                                   "MP replaces the selected cheap linears (no added computation); C adds ceil(N/K) full FP calls and keeps the full FP model resident."}
        out["note"] = ("warm CUDA-synchronized full-trajectory wall time (denoising only), 1 warm-up + 5 timed repeats, diagnostics off, no concurrent jobs, "
                       "only the arm's own models resident (FP only for C and FP50); fake-quant simulation, not packed-kernel evidence")
        out.update(run.record())
    jdump("cost.json", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["checks", "screen", "e1", "e2", "diag", "conf", "cost"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"checks": checks, "screen": screen, "e1": e1, "e2": e2, "diag": diag, "conf": conf, "cost": cost}[a.cmd](dev)


if __name__ == "__main__":
    main()
