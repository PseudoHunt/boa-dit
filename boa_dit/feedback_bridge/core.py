"""Shared core for the reference-feedback (rfb1) and coupled-gain (cfb1) pilots: one quantizer class with explicit rules on top of the validated
abn1 code (activation_bridge.run: coefficients, bridge, sampler conventions, ledger helpers). Arithmetic of the existing rules is not touched:
rule 'fb' defers to JointQuant._fb (via LocalQuant; G = 'fb' with zero_state), 'R-full' / 'R-dct' (anchor-referenced, handoff section 3.2) and
'J' (gain-scaled zero-reference feedback, section 4.1) are written out explicitly with the same casts, bounds and clamps.
"""
import math

import torch

from boa_dit.activation_bridge import run as AB
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as DJ
from boa_dit.local_refresh.run import LocalQuant
from boa_dit.model import NULL_CLASS
from boa_dit.sample import CTX
from boa_dit.temporal import act as ACT
from boa_dit.weight_anchor.run import guided

T = AB.T
coeffs, weights, bridge_coef, rl2 = AB.coeffs, AB.weights, AB.bridge_coef, AB.rl2


def _clip(v, lo, hi):
    return torch.minimum(torch.maximum(v, lo), hi)


class FBQuant(LocalQuant):
    """rule: 'fb' (parent DCT-4 feedback; with zero_state = G), 'R-full', 'R-dct', 'J'. anchors: set of sampler steps whose (single) cheap call is
    an anchor call; gains = (g_anchor, g_between) for 'J'. State: state_in/state_out (rolling residual) and ref (stored anchor reference, R only).
    racct: optional dict of passive R accounting energies per interval (keyed by the interval's anchor step)."""

    def __init__(self, model, dev):
        super().__init__(model, dev, "fb")
        self.rule, self.gains, self.anchors = "fb", (1.0, 1.0), set()
        self.ref = {}
        self.racct = None

    def reset(self):
        self.state_in, self.state_out, self.ref = {}, {}, {}

    def _fb(self, key, t, xs, D, Z, qm, lo):
        if self.rule == "fb":
            return super()._fb(key, t, xs, D, Z, qm, lo)
        if self.rule == "J":
            return self._fb_gain(key, t, xs, D, Z, qm, lo)
        return self._fb_ref(key, t, xs, D, Z, qm, lo)

    # ---------------------------------------------------------------------------------------------------- J: g_t * s_prev, zero reference
    def _fb_gain(self, key, t, xs, D, Z, qm, lo):
        U = self.U
        shp = (xs.shape[0], U.shape[1], xs.shape[2])
        if key not in self.state_in or tuple(self.state_in[key].shape) != shp or (self.auto and t == 0):
            self.state_in[key] = torch.zeros(shp, device=xs.device, dtype=torch.float16)
        g = self.gains[0] if t in self.anchors else self.gains[1]
        s_prev = torch.einsum("tk,rkc->rtc", U, self.state_in[key].float())
        hi_rep = (qm - Z[t]) * D[t]
        xcl = _clip(xs, lo, hi_rep)
        r_fed = _clip(g * s_prev, lo - xcl, hi_rep - xcl)
        u = xcl + r_fed
        uc = _clip(u, lo, hi_rep)
        q = ACT.q_static(uc, D[t], Z[t], torch.zeros((), device=xs.device), qm, hi_code=qm)
        consumed = q.to(torch.float16).float()
        r = (uc - consumed).clamp(-D[t] / 2, D[t] / 2)
        s = torch.einsum("tk,rtc->rkc", U, r).to(torch.float16)
        self.state_out[key] = s
        if self.auto:
            self.state_in[key] = s
        self.clip[t] += torch.tensor([float((u != uc).sum()), float(u.numel())], dtype=torch.float64)
        return consumed

    # ---------------------------------------------------------------------------------------------------- R: anchor-referenced
    def _fb_ref(self, key, t, xs, D, Z, qm, lo):
        full = self.rule == "R-full"
        U = self.U
        dshp = (xs.shape[0], U.shape[1], xs.shape[2])
        sshp = tuple(xs.shape) if full else dshp
        proj = lambda v: torch.einsum("tk,rtc->rkc", U, v)
        rec = (lambda s: s.float()) if full else (lambda s: torch.einsum("tk,rkc->rtc", U, s.float()))
        if self.auto and t == 0:                                             # batch start: both states reset
            self.ref.pop(key, None)
            self.state_in.pop(key, None)
        hi_rep = (qm - Z[t]) * D[t]
        xcl = _clip(xs, lo, hi_rep)
        Dt = D[t]
        if t in self.anchors:
            # exactly G's arithmetic on this input (zero carried state, same guard / representable range / spare code / cast)
            s0 = torch.einsum("tk,rkc->rtc", U, torch.zeros(dshp, device=xs.device, dtype=torch.float16).float())
            r_fed = _clip(s0, lo - xcl, hi_rep - xcl)
            u = xcl + r_fed
            uc = _clip(u, lo, hi_rep)
            q = ACT.q_static(uc, Dt, Z[t], torch.zeros((), device=xs.device), qm, hi_code=qm)
            consumed = q.to(torch.float16).float()
            m_raw = consumed - xcl                                           # rounding relative to the CLIPPED input (not clipping error)
            ref = (m_raw if full else proj(m_raw)).to(torch.float16)
            s = torch.zeros(sshp, device=xs.device, dtype=torch.float16)      # s_a = 0: the anchor residual belongs to the reference
            self.ref[key] = ref
            self.state_out[key] = s
            if self.auto:
                self.state_in[key] = s
            if self.racct is not None:
                Pm = m_raw if full else rec(proj(m_raw))
                self._acc(t, {"anchor_calls": 1.0, "E[m_raw]": m_raw, "E[Pm-m]": Pm - m_raw, "E[ref_storage]": rec(ref) - Pm,
                              "E[ref_rec-m]": rec(ref) - m_raw, "E[xcl-xs]_anchor": xcl - xs, "E[q-x]_anchor": consumed - xs})
            self.clip[t] += torch.tensor([float((u != uc).sum()), float(u.numel())], dtype=torch.float64)
            return consumed
        st = self.state_in.get(key)
        if st is None or tuple(st.shape) != sshp:
            st = torch.zeros(sshp, device=xs.device, dtype=torch.float16)
        s_prev = rec(st)
        rf = self.ref.get(key)
        tgt = s_prev if rf is None else rec(rf) + s_prev                     # no anchors: zero reference (R-dct == ordinary feedback)
        fed = _clip(tgt, lo - xcl, hi_rep - xcl)
        u = xcl + fed
        uc = _clip(u, lo, hi_rep)
        q = ACT.q_static(uc, Dt, Z[t], torch.zeros((), device=xs.device), qm, hi_code=qm)
        consumed = q.to(torch.float16).float()
        r0 = uc - consumed
        r = r0.clamp(-Dt / 2, Dt / 2)
        s = (r if full else proj(r)).to(torch.float16)
        self.state_out[key] = s
        if self.auto:
            self.state_in[key] = s
        if self.racct is not None:
            Pr = r if full else rec(proj(r))
            terms = {"held_calls": 1.0, "E[q-x]": consumed - xs, "E[xcl-xs]": xcl - xs, "E[fed]": fed, "E[guard_discard]": fed - tgt,
                     "E[uc-u]": uc - u, "E[r0]": r0, "E[clamp a=r0-r]": r0 - r, "E[(I-P)r]": r - Pr, "E[state_storage]": rec(s) - Pr,
                     "E[identity_residual]": (consumed - xs) - ((xcl - xs) + fed + (uc - u) - r0)}
            if rf is not None:
                terms["E[q-xcl-ref]"] = consumed - xcl - rec(rf)
                terms["E[s_prev]"] = s_prev
            self._acc(t, terms)
            A = self.racct.setdefault("max_abs_identity_residual", [0.0])
            A[0] = max(A[0], float(terms["E[identity_residual]"].abs().max()))
        self.clip[t] += torch.tensor([float((u != uc).sum()), float(u.numel())], dtype=torch.float64)
        return consumed

    @torch.no_grad()
    def _acc(self, t, terms):
        a = max([s for s in self.anchors if s <= t], default=-1)
        A = self.racct.setdefault(a, {})
        for k, v in terms.items():
            A[k] = A.get(k, 0.0) + (v if isinstance(v, float) else float(v.double().pow(2).sum()))
        A["n_elem"] = A.get("n_elem", 0.0) + (0.0 if not terms else float(next(v for v in terms.values() if not isinstance(v, float)).numel()))

    def state_bytes(self):
        """bytes actually held between calls (rolling state + reference), from the real tensors."""
        return int(sum(v.numel() * v.element_size() for v in self.state_in.values()) + sum(v.numel() * v.element_size() for v in self.ref.values()))


class CheapFB:
    """common base + FBQuant (one instance serves G, fb, R-full, R-dct and J; reconfigured per trajectory) + the original carry."""

    def __init__(self, dev):
        self.m = DJ.common_base_nocarry(dev)
        self.jq = FBQuant(self.m, dev)                                       # registered after the smoothing pre-hooks (as abn1 Cheap)
        self.jq.t, self.jq.auto = None, True
        self.hs = install(self.m, torch.load(AB.CARRY["0"], map_location="cpu"))

    def use(self, carry):
        assert carry == "0"


def spec(cheap, K=4, bridge=False, gains=(1.0, 1.0)):
    """cheap: 'G' | 'fb' | 'R-full' | 'R-dct' | 'J'. High model FP, original carry."""
    return {"cheap": cheap, "high": "fp", "K": K, "bridge": bridge, "carry": "0", "gains": tuple(float(g) for g in gains)}


def configure(jq, sp, anchors):
    jq.rule = {"G": "fb", "fb": "fb"}.get(sp["cheap"], sp["cheap"])
    jq.zero_state = sp["cheap"] == "G"
    jq.gains = sp.get("gains", (1.0, 1.0))
    jq.anchors = set(anchors)


# ------------------------------------------------------------------------------------------------------------------- sampler
@torch.no_grad()
def sample(M, sp, d, noise, labels, lam=1.0, diag=None, stop_at=None, resume=None, rec=None, force=None):
    """abn1 sample() with the quantizer rule configured from sp and the reference saved/resumed with the rolling state. M: {'fp', 'fb'}.
    force: optional list of latents x_i (i = 0..49) for the common-forced-trajectory diagnostic (cheap calls on the forced latent, returns
    the per-step discrepancy dict; the live sampler output is then meaningless and not returned)."""
    n = len(noise)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    Ac, bc = coeffs(d)
    tmap = torch.as_tensor(d.timestep_map, device=noise.device, dtype=torch.long)
    if "high_only" in sp:
        hm, cm, jq, anchors = M[sp["high_only"]], None, None, None
    else:
        hm = M[sp["high"]]
        ch = M["fb"]
        cm, jq = ch.m, ch.jq
        anchors = set(range(0, T, sp["K"])) if sp["K"] else set()
        configure(jq, sp, anchors)
    if resume is None:
        x, i0 = noise.float().clone(), 0
        st = {"d": None, "a": None, "pre": None}
        cnt = {"cheap": 0, "high": 0, "corrections": 0}
        if jq is not None:
            jq.reset()
    else:
        x, i0 = resume["x"].to(noise.device), resume["i"]
        st = {"d": None if resume["d"] is None else resume["d"].to(noise.device), "a": resume["a"], "pre": None}
        cnt = dict(resume["cnt"])
        torch.set_rng_state(resume["rng_cpu"]); torch.cuda.set_rng_state(resume["rng_cuda"])
        if jq is not None:
            jq.state_in, jq.state_out = {k: v.to(noise.device) for k, v in resume["fb_state"].items()}, {}
            jq.ref = {k: v.to(noise.device) for k, v in resume["ref"].items()}
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        if i < i0:
            continue
        if stop_at is not None and i == stop_at:
            return {"x": x.cpu(), "i": i, "d": None if st["d"] is None else st["d"].cpu(), "a": st["a"], "cnt": dict(cnt),
                    "rng_cpu": torch.get_rng_state(), "rng_cuda": torch.cuda.get_rng_state(),
                    "fb_state": {} if jq is None else {k: v.detach().clone().cpu() for k, v in jq.state_in.items()},
                    "ref": {} if jq is None else {k: v.detach().clone().cpu() for k, v in jq.ref.items()}}
        CTX.step, CTX.n = i, n
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        if force is not None:
            xf = force[i].to(noise.device)
            tsm = tmap[t]
            rc = guided(cm, xf, tsm, y, n)
            diag["e"][i] = (rc[:, :4] - guided(hm, xf, tsm, y, n)[:, :4]).double().cpu()
            cnt["cheap"] += 1
            continue
        if diag is not None:
            diag["x"][i] = x.detach().clone()
        if cm is None:
            def model_fn(xx, ts):
                cnt["high"] += 1
                return guided(hm, xx, ts, y, n)
        elif i in anchors:
            tsm = tmap[t]
            rc = guided(cm, x, tsm, y, n)                                    # the single cheap call of this step (state commits here)
            rh = guided(hm, x, tsm, y, n)
            cnt["cheap"] += 1
            cnt["high"] += 1
            if rec is not None:
                rec.append((x.clone(), tsm.clone(), y.clone()))
            dnew = rc[:, :4] - rh[:, :4]
            if diag is not None:
                diag["e"][i] = dnew.double().cpu()
                diag["anchor"].add(i)
            if sp.get("bridge") and st["d"] is not None:
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
                if diag is not None:                                        # diagnostic same-state FP call (stateless FP; never used live)
                    diag["e"][_i] = (rc[:, :4] - guided(diag["fp"], xx, ts, y, n)[:, :4]).double().cpu()
                if st["d"] is None:
                    return rc
                res = rc.clone()
                res[:, :4] -= st["d"]
                return res
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    if force is not None:
        return diag, cnt
    if diag is not None:
        diag["x"][T] = x.detach().clone()
    return x, cnt


def expect(sp):
    return AB.expect(sp)


def interval_stats(e, Ac, bc, K=4, lam=1.0):
    """frozen-window quantities over the K-intervals of a recorded per-step discrepancy dict e (abn1 s5 definitions):
    H = sum_j w_j (e_{a+j} - e_a), v = (e_c - e_a) sum_j (j/k) w_j; returns energies, cross term and cosine."""
    S = {k: 0.0 for k in ("raw", "held", "jump", "HH", "vv", "Hv", "res")}
    S["raw_anchor"], S["raw_between"] = 0.0, 0.0
    for a in range(0, T - K, K):
        c = a + K
        if c > T - 1 or c not in e:
            break
        w = weights(Ac, bc, a, K)
        H = sum(wj * (e[a + j] - e[a]) for j, wj in zip(range(1, K), w))
        v = bridge_coef(Ac, bc, a, K) * (e[c] - e[a])
        S["raw"] += sum(float(e[a + j].pow(2).sum()) for j in range(0, K))
        S["raw_anchor"] += float(e[a].pow(2).sum())
        S["raw_between"] += sum(float(e[a + j].pow(2).sum()) for j in range(1, K))
        S["held"] += sum(float((e[a + j] - e[a]).pow(2).sum()) for j in range(1, K))
        S["jump"] += float((e[c] - e[a]).pow(2).sum())
        S["HH"] += float(H.pow(2).sum()); S["vv"] += float(v.pow(2).sum()); S["Hv"] += float((H * v).sum())
        S["res"] += float((H - lam * v).pow(2).sum())
    S["cos_H_v"] = S["Hv"] / math.sqrt(max(S["HH"] * S["vv"], 1e-300))
    S["lambda_ls_frozen_window"] = S["Hv"] / max(S["vv"], 1e-300)
    S["frozen_window_objective_at_lam"] = S["HH"] - 2 * lam * S["Hv"] + lam ** 2 * S["vv"]
    return S
