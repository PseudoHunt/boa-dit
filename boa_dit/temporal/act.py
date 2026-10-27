"""B′ (rule.md): bed-independent pieces of the static activation quantizer study.

Quantizer: static per-tensor asymmetric, qmax = 2^bits - 1 codes, calibrated on qmax - 1 intervals (one code spare so a
dither offset o in [-Δ/2, Δ) never adds clipping):
    x̂ = (clamp(round((clip(x) + o)/Δ) + z, 0, qmax) - z)·Δ - o,   clip to [-zΔ, (qmax - 1 - z)Δ].
Deployment note: o_t folds into the preceding affine's per-step β (adaLN shift / previous bias) and the next layer's
per-step bias (b - W·o_t); on a fixed-encoding NPU it is a small per-step input tensor, the encodings stay fixed.
Tables: {key: {"D": [T], "Z": [T], "qmax": int, "alpha": [T]}}; a single table repeats one (Δ, z) over all T steps.
"""
import math

import numpy as np
import torch

ALPHAS = tuple(round(1.0 - 0.02 * i, 2) for i in range(41))           # 1.00 ... 0.20
KINDS = {"step8": (8, False), "single8": (8, True), "single6": (6, True)}
DITHERS = ("none", "sym2", "golden", "half2", "unif", "cyc4a", "cyc4b")
CYC = {"cyc4a": (0, 1, 2, 3), "cyc4b": (0, 2, 1, 3)}


def q_static(x, D, Z, O, qmax=255, hi_code=None):
    """hi_code: top of the clip interval in codes (default qmax - 1, the calibrated interval; feedback uses qmax)."""
    xc = torch.clamp(x, -Z * D, ((qmax - 1 if hi_code is None else hi_code) - Z) * D)
    q = torch.clamp(torch.round((xc + O) / D) + Z, 0, qmax)
    return (q - Z) * D - O


def q_dynamic(x, qmax=255):
    lo, hi = x.amin(-1, keepdim=True), x.amax(-1, keepdim=True)
    D = ((hi - lo) / qmax).clamp_min(1e-8)
    Z = torch.round(-lo / D)
    return (torch.clamp(torch.round(x / D) + Z, 0, qmax) - Z) * D


def offset_frac(dither, T, key_index=0):
    """o_t / Δ_t for t = 0..T-1 (Δ_t from the layer's table). 'unif' is independent per layer (key_index) and step:
    U(-1/2, 1/2), generator seed 0 + key_index; 'cyc4a'/'cyc4b': d_j = (j - 1.5)/4 with j = order[t mod 4]."""
    if dither == "unif":
        return torch.rand(T, generator=torch.Generator().manual_seed(key_index)) - 0.5
    if dither in CYC:
        return torch.tensor([(CYC[dither][t % 4] - 1.5) / 4 for t in range(T)])
    if dither == "none":
        return torch.zeros(T)
    if dither == "sym2":                                                # zero mean per element: -Δ/4, +Δ/4
        return torch.tensor([-0.25 if t % 2 == 0 else 0.25 for t in range(T)])
    if dither == "golden":
        return torch.tensor([math.modf(t * 0.6180339)[0] for t in range(T)])
    if dither == "half2":                                               # biased: 0, Δ/2 (on-grid inputs: mean -Δ/4)
        return torch.tensor([0.0 if t % 2 == 0 else 0.5 for t in range(T)])
    raise ValueError(dither)


NBINS = 65536


class Calib:
    """Joint calibration of step8 / single8 / single6 tables. Pass 1: add_minmax(key, t, x). Pass 2: add_hist(key, t, x)
    (all tokens): a 65,536-bin histogram per (layer, step) over that step's [min, max]; the squared rounding + clipping
    error of every candidate range is evaluated at the bin centres (bin width <= Δ/50 at α = 0.20, A8). Then tables(T).
    (add_err, the direct token-subsample version, is kept for reference; it is too slow at this scale.)"""

    def __init__(self, dev):
        self.dev, self.mn, self.mx, self.err = dev, {}, {}, {}
        self.al = torch.tensor(ALPHAS, device=dev).view(-1, 1, 1)

    def add_minmax(self, key, t, x):
        a, b = float(x.amin()), float(x.amax())
        k = (key, t)
        self.mn[k], self.mx[k] = min(self.mn.get(k, a), a), max(self.mx.get(k, b), b)

    def gminmax(self, key):
        ts = [t for (k, t) in self.mn if k == key]
        return min(self.mn[(key, t)] for t in ts), max(self.mx[(key, t)] for t in ts)

    def _errs(self, xs, lo, hi, qmax):
        D = (hi - lo) / (qmax - 1)
        Z = torch.round(-lo / D)
        out = []
        for i in range(0, len(ALPHAS), 4):                              # chunks of 4 alphas (memory)
            q = q_static(xs[None], D[i:i + 4], Z[i:i + 4], torch.zeros((), device=self.dev), qmax)
            out.append((q - xs[None]).pow(2).sum((1, 2)).double())
        return torch.cat(out).cpu()

    def add_err(self, key, t, xs):
        if not hasattr(self, "_g"):
            self._g = {}
        if key not in self._g:
            self._g[key] = self.gminmax(key)
        gl, gh = self._g[key]
        mn, mx = self.mn[(key, t)], self.mx[(key, t)]
        for kind, (bits, single) in KINDS.items():
            lo, hi = (self.al * gl, self.al * gh) if single else (self.al * mn, self.al * mx)
            e = self._errs(xs, lo, hi, 2 ** bits - 1)
            k = (kind, key) if single else (kind, key, t)
            self.err[k] = self.err.get(k, 0) + e

    def add_hist(self, key, t, x):
        if not hasattr(self, "hist"):
            self.hist = {}
        mn, mx = self.mn[(key, t)], self.mx[(key, t)]
        h = torch.histc(x.float(), bins=NBINS, min=mn, max=mx).double().cpu()
        k = (key, t)
        self.hist[k] = h if k not in self.hist else self.hist[k] + h

    def _centres(self, key, t):
        mn, mx = self.mn[(key, t)], self.mx[(key, t)]
        w = (mx - mn) / NBINS
        c = (mn + w * (torch.arange(NBINS, device=self.dev, dtype=torch.float64) + 0.5)).view(1, -1)
        return c, self.hist[(key, t)].to(self.dev).view(1, -1)

    @staticmethod
    def _err_rows(c, h, lo, hi, qmax):
        """lo, hi [n] candidate ranges -> squared error [n] at the bin centres c (counts h)."""
        lo, hi = lo.view(-1, 1), hi.view(-1, 1)
        D = (hi - lo) / (qmax - 1)
        Z = torch.ceil(-lo / D)                                               # nudge: the range always covers lo
        xc = torch.minimum(torch.maximum(c, -Z * D), (qmax - 1 - Z) * D)
        q = (torch.clamp(torch.round(xc / D) + Z, 0, qmax) - Z) * D
        return ((q - c).pow(2) * h).sum(1)

    def finish_hist(self, lo_min_keys=()):
        """Independent lo / hi search: lo = α_lo·min, hi = α_hi·max, α_lo, α_hi ∈ ALPHAS (41 x 41). For keys whose layer
        type is in lo_min_keys (one-sided post-GELU inputs), lo = min (unsigned, zero point at the lobe minimum)."""
        al = torch.tensor(ALPHAS, device=self.dev, dtype=torch.float64)
        self._g = {key: self.gminmax(key) for key in {k for k, _ in self.mn}}
        self.lo_min = lambda key: key.split(".", 1)[1] in lo_min_keys
        for (key, t) in self.hist:
            c, h = self._centres(key, t)
            gl, gh = self._g[key]
            mn, mx = self.mn[(key, t)], self.mx[(key, t)]
            for kind, (bits, single) in KINDS.items():
                bl, bh = (gl, gh) if single else (mn, mx)
                los = al[:1] * bl if self.lo_min(key) else al * bl
                E = torch.cat([self._err_rows(c, h, los[i:i + 8].repeat_interleave(len(al)), (al * bh).repeat(len(los[i:i + 8])),
                                              2 ** bits - 1).view(-1, len(al)) for i in range(0, len(los), 8)]).cpu()
                k = (kind, key) if single else (kind, key, t)
                self.err[k] = self.err.get(k, 0) + E                         # [n_lo, 41]

    def tables(self, T, lo_min_keys=()):
        if getattr(self, "hist", None):
            self.err = {}
            self.finish_hist(lo_min_keys)
        out = {kind: {} for kind in KINDS}
        keys = sorted({k for k, _ in self.mn})
        for kind, (bits, single) in KINDS.items():
            qmax = 2 ** bits - 1
            for key in keys:
                D, Z, A, AL, CL = [], [], [], [], []
                for t in range(T):
                    E = self.err[(kind, key)] if single else self.err[(kind, key, t)]
                    j = int(torch.argmin(E))
                    jl, jh = divmod(j, E.shape[1])
                    bl, bh = self._g[key] if single else (self.mn[(key, t)], self.mx[(key, t)])
                    lo, hi = ALPHAS[jl] * bl, ALPHAS[jh] * bh
                    d = (hi - lo) / (qmax - 1)
                    zq = math.ceil(-lo / d)                                   # zero-point nudge: covers lo
                    D.append(d); Z.append(zq); A.append(ALPHAS[jh]); AL.append(ALPHAS[jl])
                    c, h = self._centres(key, t)
                    rlo, rhi = -zq * d, (qmax - 1 - zq) * d
                    CL.append(float(h[(c < rlo) | (c > rhi)].sum() / h.sum()))
                out[kind][key] = {"D": torch.tensor(D, dtype=torch.float64), "Z": torch.tensor(Z, dtype=torch.float64),
                                  "qmax": qmax, "alpha": torch.tensor(A), "alpha_lo": torch.tensor(AL), "clip_rate": torch.tensor(CL)}
        return out


def clip_summary(tabs):
    """Mean clip rate per (kind, layer type) of the kept ranges (histogram estimate)."""
    out = {}
    for kind, tb in tabs.items():
        by = {}
        for k, v in tb.items():
            by.setdefault(k.split(".", 1)[1], []).append(float(v["clip_rate"].mean()))
        out[kind] = {t: sum(x) / len(x) for t, x in by.items()}
    return out


def variants(tabs, T):
    """B1′ layer-level quantizer variants: name -> (table kind or 'dynamic', dither)."""
    return {"single8_none": ("single8", "none"), "single8_sym2": ("single8", "sym2"), "single8_golden": ("single8", "golden"),
            "single8_half2": ("single8", "half2"), "step8_none": ("step8", "none"), "dynamic8": ("dynamic", "none"),
            "single6_none": ("single6", "none"), "single6_sym2": ("single6", "sym2")}


class Coherence:
    """Per (variant, layer): ρ_adj(t) Pearson of e_t and e_{t+1}; coherent fraction ||Σ_t e_t||² / (T Σ_t ||e_t||²) over a
    batch's rows; output error energy Σ||W e||²; exact-zero / on-grid fractions (single8 table). Call step(...) at every
    (layer, step) of a batch, end_batch() after each batch."""

    def __init__(self, tabs, T, dev):
        self.tabs, self.T, self.dev = tabs, T, dev
        self.V = variants(tabs, T)
        self.fr = {d: offset_frac(d, T) for d in DITHERS}
        self.acc, self.grid, self.split = {}, {}, {}

    def _q(self, vname, key, t, x):
        kind, dither = self.V[vname]
        if kind == "dynamic":
            return q_dynamic(x)
        tb = self.tabs[kind][key]
        D, Z = float(tb["D"][t]), float(tb["Z"][t])
        Dt, Zt = torch.tensor(D, device=self.dev), torch.tensor(Z, device=self.dev)
        return q_static(x, Dt, Zt, torch.tensor(float(self.fr[dither][t]) * D, device=self.dev), tb["qmax"])

    def step(self, key, t, xs, W):
        """xs [rows, n, c] fp32 inputs; W [d_out, c] weight (any dtype)."""
        tb = self.tabs["single8"][key]
        D, Z = float(tb["D"][0]), float(tb["Z"][0])
        g = self.grid.setdefault(key, [0, 0, 0])
        u = xs / D + Z
        g[0] += xs.numel(); g[1] += int((xs == 0).sum()); g[2] += int((u == torch.round(u)).sum())
        for kind in ("single8", "single6"):                            # stage 2: rounding vs clipping split
            tb2 = self.tabs[kind][key]
            D2, Z2 = float(tb2["D"][t]), float(tb2["Z"][t])
            lo, hi = -Z2 * D2, (tb2["qmax"] - 1 - Z2) * D2
            cl = xs.clamp(lo, hi)
            sp = self.split.setdefault((kind, key), [0.0, 0.0, 0, 0])
            sp[0] += float((xs - cl).double().pow(2).sum())
            sp[1] += float((self._q(kind + "_none", key, t, xs) - cl).double().pow(2).sum())
            sp[2] += int((xs != cl).sum()); sp[3] += xs.numel()
        for vn in self.V:
            e = self._q(vn, key, t, xs) - xs
            a = self.acc.setdefault((vn, key), {"S": None, "prev": None, "SS": 0.0, "num": 0.0, "den": 0.0, "out": 0.0,
                                                "rho": torch.zeros(self.T - 1, 6, dtype=torch.float64)})
            a["S"] = e.clone() if a["S"] is None else a["S"] + e
            a["SS"] += float(e.double().pow(2).sum())
            if vn == "single8_none":
                a["out"] += float((e.to(W.dtype) @ W.T).float().pow(2).sum())
            if a["prev"] is not None:
                p, c = a["prev"].double().flatten(), e.double().flatten()
                a["rho"][t - 1] += torch.tensor([float(p.sum()), float(c.sum()), float((p * p).sum()), float((c * c).sum()),
                                                 float((p * c).sum()), float(p.numel())], dtype=torch.float64)
            a["prev"] = e

    def end_batch(self):
        for a in self.acc.values():
            if a["S"] is not None:
                a["num"] += float(a["S"].double().pow(2).sum()); a["den"] += a["SS"]
            a["S"], a["prev"], a["SS"] = None, None, 0.0

    def result(self):
        res = {"variants": {}, "grid": {}}
        for (vn, key), a in self.acc.items():
            sx, sy, sxx, syy, sxy, n = (a["rho"][:, i] for i in range(6))
            rho = ((sxy / n - sx / n * sy / n) / torch.sqrt((sxx / n - (sx / n) ** 2) * (syy / n - (sy / n) ** 2))).tolist()
            res["variants"].setdefault(vn, {})[key] = {"rho_adj": rho, "rho_adj_mean": float(np.nanmean(rho)),
                                                       "coherent_frac": a["num"] / (self.T * a["den"]) if a["den"] else None,
                                                       "out_err_energy": a["out"]}
        base = res["variants"]["single8_none"]
        tot = sum(v["out_err_energy"] for v in base.values())
        share = {k: v["out_err_energy"] / tot for k, v in base.items()}
        order = sorted(share, key=lambda k: -share[k])
        top, cum = [], 0.0
        for k in order:
            top.append(k); cum += share[k]
            if cum >= 0.5:
                break
        w = np.array([share[k] for k in top])
        res["top_half_layers"], res["out_err_share"] = top, share
        res["top_half_weighted"] = {vn: {"rho_adj": float((w * np.array([L[k]["rho_adj_mean"] for k in top])).sum() / w.sum()),
                                         "coherent_frac": float((w * np.array([L[k]["coherent_frac"] for k in top])).sum() / w.sum())}
                                    for vn, L in res["variants"].items()}
        ltype = lambda k: k.split(".", 1)[1]
        by = {}
        for vn, L in res["variants"].items():
            for k, v in L.items():
                by.setdefault(vn, {}).setdefault(ltype(k), []).append((v, share[k]))
        res["by_type"] = {vn: {t: {"rho_adj_mean": float(np.mean([v["rho_adj_mean"] for v, _ in vs])),
                                   "coherent_frac": float(np.mean([v["coherent_frac"] for v, _ in vs])),
                                   "out_err_share": float(sum(s for _, s in vs))} for t, vs in d.items()} for vn, d in by.items()}
        gt = {}
        for k, (n, z, og) in self.grid.items():
            g = gt.setdefault(ltype(k), [0, 0, 0]); g[0] += n; g[1] += z; g[2] += og
        res["grid"] = {t: {"frac_exact_zero": z / n, "frac_on_grid_single8": og / n} for t, (n, z, og) in gt.items()}
        st = {}
        for (kind, k), (ce, re_, nc, n) in self.split.items():
            a = st.setdefault(kind, {}).setdefault(ltype(k), [0.0, 0.0, 0, 0])
            a[0] += ce; a[1] += re_; a[2] += nc; a[3] += n
        res["round_clip_split"] = {kind: {t: {"clip_energy_share": a[0] / (a[0] + a[1]), "clip_rate": a[2] / a[3]} for t, a in d.items()}
                                   for kind, d in st.items()}
        return res


def gate(b1p, floor_key="dynamic8"):
    """Gate per bed: single8 C/I - floor >= 0.03 and top-half energy-weighted ρ_adj under sym2 <= -0.3."""
    ex = b1p["net"]["single8"]["C_over_I"] - b1p["net"][floor_key]["C_over_I"]
    rho = b1p["layer"]["top_half_weighted"]["single8_sym2"]["rho_adj"]
    return {"ci_excess": ex, "rho_sym2_top_half": rho, "pass": bool(ex >= 0.03 and rho <= -0.3)}
