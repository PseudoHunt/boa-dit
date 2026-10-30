"""CPU analysis stages for qo1 (results/quantizer_observer/PROTOCOL.md): inventory headroom accounting errstats observer_select observer_confirm cost report."""
import csv
import json
import os
import subprocess

import numpy as np
import torch

from boa_dit.common import ROOT, RESULTS, atomic_write_json, atomic_write_text
from boa_dit.objective.d2 import band_masks
from boa_dit.quantizer_observer.run import CK, EFEAT, KEYS, OUT, PRIMARY, PRIMARY_SHA, SPLITS, STAGE, T, TERMS, jdump, jload

NB, BW = 5, 10                                       # five fixed 10-step buckets
LAMS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
NPC = 32
NPERM = 20
TARGETS = ("eW", "eAW", "etot")
TYPES = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def boot(fn, n_img, n=2000, seed=0):
    idx = np.random.default_rng(seed).integers(0, n_img, (n, n_img))
    v = fn(np.arange(n_img))
    d = np.array([fn(i) for i in idx])
    return {"v": float(v), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def load_trace(split):
    return torch.load(os.path.join(CK, f"trace_{split}.pt"), map_location="cpu")


# ------------------------------------------------------------------------------------------------ inventory / manifest
def stage_inventory():
    git = lambda *a: subprocess.check_output(["git", "-C", ROOT, *a], text=True).strip()
    heads = {}
    for b in ("dit-joint-feedback", "weights-joint", "dns-response-pilot", "analytic-carry-pilot", "quantizer-observer-pilot"):
        try:
            heads[b] = git("rev-parse", f"origin/{b}")
        except Exception:
            heads[b] = git("rev-parse", b)
    from boa_dit.dit_joint import run as J
    obs = torch.load("/home/boa-dit/ckpt/observability/traces_cal.pt", map_location="cpu")
    man = {"branch_heads": heads, "code_commit": git("rev-parse", "HEAD"),
           "primary_carry": {"path": PRIMARY, "sha256": PRIMARY_SHA, "selection": "E_L u100, dev-selected in results/dit_joint/select.json (frozen 2026-10-29)"},
           "secondary_carry": {"path": J.BASE_CARRY, "sha256": J.sha(J.BASE_CARRY)},
           "base_model": {"path": J.BASE_STATE, "sha256": J.sha(J.BASE_STATE)}, "tables": {"path": J.TABLES, "sha256": J.sha(J.TABLES)},
           "smoothing": {"path": J.SMOOTH, "sha256": J.sha(J.SMOOTH)},
           "splits": {k: [v[0], v[-1], len(v)] for k, v in SPLITS.items()},
           "numerics": {"sampler": "sample.ddim_sample, DDIM-50, eta 0, cfg 1.5 on the first 3 eps channels", "autocast": "fp16", "tf32": False,
                        "batch": 32, "accounting": "float64, rows chunked by 16", "feature storage": "fp16 (E: fp32)"},
           "cache_inventory": {
               "dit_joint eval_latents.pt": "final latents only (no per-step errors, no internal features): insufficient",
               "dit_joint eval.json diag": "aggregate per-image C and I only: insufficient",
               "observability traces_{cal,eval}.pt": {"keys": {k: list(v.shape) if torch.is_tensor(v) else type(v).__name__ for k, v in obs.items()},
                                                       "bed": "original heig-8 W4A16, teacher-forced on FP states; no carry bottlenecks or quantizer signals: not reusable"},
               "unpublished local observer tests": "none found (searched all worktrees, branches and ckpt/)"}}
    jdump("manifest.json", man)
    print(json.dumps(man, indent=1, default=str))


# ------------------------------------------------------------------------------------------------ A1 headroom
def headroom_terms(F, W, WA):
    nf = F.flatten(1).double().pow(2).sum(1)
    dW, dA = (W - F).flatten(1).double(), (WA - W).flatten(1).double()
    tot = (WA - F).flatten(1).double().pow(2).sum(1) / nf
    tW, tA, cr = dW.pow(2).sum(1) / nf, dA.pow(2).sum(1) / nf, 2 * (dW * dA).sum(1) / nf
    return {"mse_WA": tot.numpy(), "dW2": tW.numpy(), "dA2": tA.numpy(), "cross": cr.numpy(), "closure_max_abs": float((tot - tW - tA - cr).abs().max())}


def headroom_summary(h):
    n = len(h["mse_WA"])
    out = {k: boot(lambda i, k=k: h[k][i].mean(), n) for k in ("mse_WA", "dW2", "dA2", "cross")}
    out["gain_remove_A8"] = boot(lambda i: 1 - h["dW2"][i].mean() / h["mse_WA"][i].mean(), n)
    out["closure_max_abs"] = h["closure_max_abs"]
    out["per_image"] = {k: h[k].tolist() for k in ("mse_WA", "dW2", "dA2", "cross")}
    out["n"] = n
    return out


def stage_headroom(splits=("fit", "dev")):
    res = jload("headroom.json") or {}
    for sp in splits:
        tr = load_trace(sp)
        res[sp] = headroom_summary(headroom_terms(tr["F_end"], tr["W_end"], tr["end"]))
        res[sp]["config"] = "primary E_L u100: F, W (A16, fb off), WA (A8 + DCT-4 feedback)"
    if "dev" in splits and os.path.exists(os.path.join(CK, "dev_orig.pt")):
        do = torch.load(os.path.join(CK, "dev_orig.pt"))
        F = load_trace("dev")["F_end"]
        res["dev_original_carry"] = headroom_summary(headroom_terms(F, do["W_end"], do["WA_end"]))
        res["dev_original_carry"]["config"] = "carry-transfer check: original common carry (configuration B), same F"
    if "dev" in res:
        g = res["dev"]["gain_remove_A8"]
        rule = "proceed" if g["v"] >= 0.20 else ("uncertain" if g["ci"][1] >= 0.20 else "park activation-observer live arm")
        res["activation_prioritization"] = {"dev_gain": g, "rule": rule,
                                            "definition": ">= 20% proceed; point < 20% but CI spans 20% uncertain (cached regression only, no activation live arm); upper CI < 20% park"}
    jdump("headroom.json", res)
    print(json.dumps({k: ({q: v[q] for q in ("mse_WA", "dW2", "dA2", "cross", "gain_remove_A8", "closure_max_abs")} if isinstance(v, dict) and "mse_WA" in v else v)
                      for k, v in res.items()}, indent=1))


# ------------------------------------------------------------------------------------------------ site accounting
def stage_accounting():
    tr = load_trace("fit")
    acc = tr["acc"]
    agg = {}
    tot = {"G": np.zeros((9, 9)), "qx_g": 0.0, "res_g": 0.0}
    for (t, k), r in acc.items():
        typ, grp, bkt = k.split(".", 1)[1], int(k.split(".")[0][1:]) // 7, t // BW
        for name in (f"type:{typ}", f"depth:{grp}", f"steps:{bkt * BW}-{bkt * BW + BW - 1}", "all"):
            a = agg.setdefault(name, {"G": np.zeros((9, 9)), "qx_g": 0.0, "res_g": 0.0, "cnt": np.zeros(5), "n": 0, "qx": 0.0})
            a["cnt"] += r["cnt"].numpy()
            a["n"] += r["n"]
            a["qx"] += float(r["qx"])
            if "G" in r:
                a["G"] += r["G"].numpy()
                a["qx_g"] += float(r["qx_g"])
                a["res_g"] += float(r["res_g"])
    out = {"terms": TERMS, "identity": "q - x = p - p_next + c0 + g + rho_u + c1 - (I-P)r - a + eta (rho_u = fp32 rounding of u = xcl + r_fed)",
           "gram_images": "first fit batch (32 images, both CFG rows), all 50 steps, 112 sites; rates over all 64 fit images", "groups": {}}
    rel_res = []
    for (t, k), r in acc.items():
        if "G" in r:
            rel_res.append(float(r["res_g"]) / max(float(r["qx_g"]), 1e-30))
    for name, a in agg.items():
        G, q = a["G"], a["qx_g"]
        out["groups"][name] = {
            "energy_q_minus_x": q, "term_energy": dict(zip(TERMS, np.diag(G).tolist())), "term_energy_frac_of_q_minus_x": dict(zip(TERMS, (np.diag(G) / max(q, 1e-30)).tolist())),
            "cross_terms_2Gij": {f"{TERMS[i]}*{TERMS[j]}": float(2 * G[i, j]) for i in range(9) for j in range(i + 1, 9)},
            "gram_sum_minus_qx_rel": float((G.sum() - q) / max(q, 1e-30)), "identity_resid_energy_rel": a["res_g"] / max(q, 1e-30),
            "rates": dict(zip(("input_clip(xcl!=x)", "state_guard(rfed!=p)", "second_clip(u!=uc)", "rounding_clamp(|r_raw|>D/2)", "spare_code(code==qmax)"), (a["cnt"] / max(a["n"], 1)).tolist()))}
    out["identity_resid_rel_max_site_step"] = max(rel_res)
    out["note"] = ("site-level accounting of the consumed-activation error, not an attribution of endpoint damage. Energies do not add: the cross terms are reported. "
                   "The old clip counter recorded only u != uc (second clip); input clipping (xcl != x) is counted here separately.")
    jdump("site_accounting.json", out)
    a = out["groups"]["all"]
    print(json.dumps({"fractions": a["term_energy_frac_of_q_minus_x"], "rates": a["rates"], "resid": out["identity_resid_rel_max_site_step"]}, indent=1))


# ------------------------------------------------------------------------------------------------ A2 error statistics
def dct1(n=32, k=4):
    i = np.arange(n)
    return np.stack([np.ones(n) / np.sqrt(n) if f == 0 else np.sqrt(2 / n) * np.cos(np.pi * (2 * i + 1) * f / (2 * n)) for f in range(k)])


B4 = dct1()


def proj16(e):
    """first 4x4 spatial DCT modes per latent channel: [..., 4, 32, 32] -> [..., 64]."""
    e = np.asarray(e, dtype=np.float64)
    c = np.einsum("fi,gj,...cij->...cfg", B4, B4, e)
    return c.reshape(*e.shape[:-3], 64)


def err_stats(E):
    """E: [n, 50, 4, 32, 32] float64."""
    n = E.shape[0]
    flat = E.reshape(n, T, -1)
    en = (flat ** 2).sum(2)                                                    # [n, T]
    cos = lambda k: float(np.mean((flat[:, :-k] * flat[:, k:]).sum(2) / np.sqrt(en[:, :-k] * en[:, k:] + 1e-60)))
    dc = (E.mean((3, 4)) ** 2).sum(2) * 32 * 32
    low = (proj16(E) ** 2).sum(2)
    F = np.abs(np.fft.fft2(E)) ** 2 / (32 * 32)
    bands = [float((F * m.numpy()).sum() / F.sum()) for m in band_masks(32)]
    const = np.ones(T) / np.sqrt(T)
    alt = np.array([(-1) ** t for t in range(T)]) / np.sqrt(T)
    pc = np.einsum("t,ntd->nd", const, flat)
    pa = np.einsum("t,ntd->nd", alt, flat)
    tot = en.sum()
    tdct = np.stack([np.cos(np.pi * (np.arange(T) + 0.5) * f / T) for f in range(4)])
    tdct = tdct / np.linalg.norm(tdct, axis=1, keepdims=True)
    slow = float((np.einsum("ft,ntd->nfd", tdct, flat) ** 2).sum() / tot)
    S = flat.sum(1)
    coh = (S ** 2).sum(1) / (T * en.sum(1))
    return {"energy_mean_per_image": float(en.sum(1).mean()), "energy_per_step_mean": en.mean(0).tolist(), "lag1_cos": cos(1), "lag5_cos": cos(5),
            "spatial_DC_frac": float(dc.sum() / tot), "dct4x4_frac": float(low.sum() / tot), "radial_band_frac": bands,
            "temporal_constant_frac": float((pc ** 2).sum() / tot), "temporal_period2_frac": float((pa ** 2).sum() / tot),
            "temporal_remaining_frac": float(1 - ((pc ** 2).sum() + (pa ** 2).sum()) / tot), "temporal_first4_dct_frac (optional)": slow,
            "coherence_c_mean_imagewise": float(coh.mean()), "coherence_c_ratio_of_sums": float((S ** 2).sum() / (T * en.sum())), "coherence_iid_expectation": 1 / T}


def stage_errstats():
    out = {"definition": "c(e) = ||sum_t e_t||^2 / (T sum_t ||e_t||^2); old joint C/I must be divided by T", "splits": ["fit", "dev"]}
    for src in TARGETS:
        Es = []
        for sp in ("fit", "dev"):
            tr = load_trace(sp)
            Es.append((tr["eW"] + tr["eAW"]).double().numpy() if src == "etot" else tr[src].double().numpy())
        out[src] = err_stats(np.concatenate(Es))
    tr = load_trace("fit")
    clo = (tr["eps_wa"] - (tr["eAW"] + tr["eW"])).double()
    out["closure_note"] = "e_total = e_A|W + e_W by construction from (f_WA - f_W) + (f_W - f_F); independent closure checked in checks.json c6"
    jdump("errstats.json", out)
    print(json.dumps({s: {k: v for k, v in out[s].items() if k != "energy_per_step_mean"} for s in TARGETS}, indent=1))


# ------------------------------------------------------------------------------------------------ observer data
def shift(a):
    p = np.zeros_like(a)
    p[:, 1:] = a[:, :-1]
    return p


def observer_data(split):
    tr = load_trace(split)
    n = len(tr["seeds"])
    Y = {"eW": proj16(tr["eW"]), "eAW": proj16(tr["eAW"])}
    Y["etot"] = Y["eW"] + Y["eAW"]
    e = proj16(tr["eps_wa"])
    valid = np.ones((n, T, 1))
    valid[:, 0] = 0
    tt = np.broadcast_to((np.arange(T) / T)[None, :, None], (n, T, 1))
    H = np.concatenate([e, shift(e), e - shift(e), proj16(tr["x"]), tt, valid], 2)
    fl = lambda k: tr[k].float().numpy().reshape(n, T, -1).astype(np.float64)
    C = fl("C")
    R = fl("R")
    Ef = tr["E"].numpy().astype(np.float64)                                    # [n, T, 2, 112, 10]
    Ef = np.concatenate([np.log10(Ef[..., :6] + 1e-30), Ef[..., 6:]], -1).reshape(n, T, -1)
    S = np.concatenate([fl("S0"), fl("Sg")], 2)
    fam = {"H": H, "C": np.concatenate([C, C - shift(C)], 2), "R": np.concatenate([R, R - shift(R)], 2), "E": Ef, "S": S,
           "O": shift(Y["etot"])}
    return fam, Y, n, tr["labels"]


MODELS = {"H": ("H",), "H+C": ("H", "C"), "H+R": ("H", "R"), "H+E": ("H", "E"), "H+R+E+S": ("H", "R", "E", "S"), "H+C+R+E+S": ("H", "C", "R", "E", "S"),
          "Oracle": ("H", "O")}
ADDED = {"H+C": ("C",), "H+R": ("R",), "H+E": ("E",), "H+R+E+S": ("R", "E", "S"), "H+C+R+E+S": ("C", "R", "E", "S")}


class Prep:
    """per-bucket preprocessing fitted on training images only: H (and O) standardized; each added family PCA -> <= 32 comps, standardized."""

    def __init__(self, fam, imgs, b, fams):
        sl = slice(b * BW, b * BW + BW)
        self.p = {}
        for f in fams:
            X = fam[f][imgs, sl].reshape(-1, fam[f].shape[2])
            mu = X.mean(0)
            if f in ("H", "O"):
                sd = X.std(0)
                self.p[f] = ("std", mu, np.where(sd > 1e-8, sd, 1.0))
            else:
                Xc = torch.from_numpy(X - mu).float()
                _, s, Vt = torch.linalg.svd(Xc, full_matrices=False)
                k = int(min(NPC, (s > s[0] * 1e-6).sum()))
                P = Vt[:k].T.double().numpy()
                Z = (X - mu) @ P
                sd = Z.std(0)
                self.p[f] = ("pca", mu, P, np.where(sd > 1e-12, sd, 1.0))

    def tf(self, fam, imgs, b, fams, perm=None):
        sl = slice(b * BW, b * BW + BW)
        out = []
        for f in fams:
            src = imgs if (perm is None or f in ("H", "O")) else imgs[perm]
            X = fam[f][src, sl].reshape(-1, fam[f].shape[2])
            p = self.p[f]
            out.append((X - p[1]) / p[2] if p[0] == "std" else ((X - p[1]) @ p[2]) / p[3])
        return np.concatenate(out, 1)


def m0(Y, imgs, b):
    return Y[imgs, b * BW:b * BW + BW].mean(0)                                # [10, 64] per-step training mean


def ridge(X, Yc, lams):
    N = len(X)
    A, Bm = X.T @ X / N, X.T @ Yc / N
    return {l: np.linalg.solve(A + l * np.eye(len(A)), Bm) for l in lams}


def fit_predict(famA, YA, imgsA, famB, imgsB, fams, tgt, lams, permA=None, permB=None, prep_cache=None):
    """train on (famA, imgsA), predict (famB, imgsB). Returns {lam: pred [nB, T, 64]}, m0 pred."""
    nB = len(imgsB)
    preds = {l: np.zeros((nB, T, 64)) for l in lams}
    base = np.zeros((nB, T, 64))
    for b in range(NB):
        key = (b, fams)
        prep = prep_cache.get(key) if prep_cache is not None else None
        if prep is None:
            prep = Prep(famA, imgsA, b, fams)
            if prep_cache is not None:
                prep_cache[key] = prep
        mu = m0(YA[tgt], imgsA, b)
        XA = prep.tf(famA, imgsA, b, fams, permA)
        YcA = (YA[tgt][imgsA, b * BW:b * BW + BW] - mu[None]).reshape(-1, 64)
        W = ridge(XA, YcA, lams)
        XB = prep.tf(famB, imgsB, b, fams, permB)
        base[:, b * BW:b * BW + BW] = mu[None]
        for l in lams:
            preds[l][:, b * BW:b * BW + BW] = mu[None] + (XB @ W[l]).reshape(nB, BW, 64)
    return preds, base


def cv_lambda(fam, Y, n, fams, tgt):
    imgs = np.arange(n)
    err = {l: 0.0 for l in LAMS}
    for f in range(4):
        te, trn = imgs[imgs % 4 == f], imgs[imgs % 4 != f]
        P, _ = fit_predict(fam, Y, trn, fam, te, fams, tgt, LAMS)
        for l in LAMS:
            err[l] += float(((P[l] - Y[tgt][te]) ** 2).sum())
    return min(LAMS, key=lambda l: (err[l], -l)), err


def metrics(y, pred, base, predH=None):
    """per-image low-band metrics; y, pred, base: [n, T, 64]."""
    n = len(y)
    mse = ((y - pred) ** 2).sum((1, 2))
    out = {"mse_per_image": mse.tolist(), "ratio_raw": float(mse.sum() / (y ** 2).sum((1, 2)).sum()),
           "ratio_centered": float(mse.sum() / ((y - base) ** 2).sum((1, 2)).sum()),
           "temporal_mean_left_frac": float(((y - pred).sum(1) ** 2).sum() / ((y.sum(1)) ** 2).sum()),
           "dot_frac": float(((pred - base) * (y - base)).sum() / ((y - base) ** 2).sum()),
           "cos": float(((pred - base) * (y - base)).sum() / np.sqrt(((pred - base) ** 2).sum() * ((y - base) ** 2).sum() + 1e-60)),
           "finite": bool(np.isfinite(pred).all())}
    if predH is not None:
        mH = ((y - predH) ** 2).sum((1, 2))
        out["vs_H_rel_change"] = boot(lambda i: mse[i].mean() / mH[i].mean() - 1, n)
        out["median_per_image_improvement_vs_H"] = float(np.median((mH - mse) / mH))
        out["wins_vs_H"] = int((mse < mH).sum())
    return out


def derange(n, rng):
    while True:
        p = rng.permutation(n)
        if not (p == np.arange(n)).any():
            return p


def evaluate_all(famF, YF, nF, famE, YE, nE, lam_tab, models, n_perm=NPERM, labelsF=None, labelsE=None):
    trn, tst = np.arange(nF), np.arange(nE)
    cache = {}
    res = {}
    for tgt in TARGETS:
        res[tgt] = {}
        PH, base = fit_predict(famF, YF, trn, famE, tst, MODELS["H"], tgt, [lam_tab[tgt]["H"]], prep_cache=cache)
        predH = PH[lam_tab[tgt]["H"]]
        res[tgt]["M0"] = metrics(YE[tgt], base, base, predH)
        for name in models:
            l = lam_tab[tgt][name]
            P, _ = fit_predict(famF, YF, trn, famE, tst, MODELS[name], tgt, [l], prep_cache=cache)
            r = metrics(YE[tgt], P[l], base, predH if name != "H" else None)
            r["lambda"] = l
            r["n_features"] = int(sum(min(NPC, famF[f].shape[2]) if f not in ("H", "O") else famF[f].shape[2] for f in MODELS[name]))
            if name in ADDED and n_perm:
                sh = []
                for k in range(n_perm):
                    rng = np.random.default_rng(1000 + k)
                    Ps, _ = fit_predict(famF, YF, trn, famE, tst, MODELS[name], tgt, [l], permA=derange(nF, rng), permB=derange(nE, rng), prep_cache=cache)
                    mse = ((YE[tgt] - Ps[l]) ** 2).sum((1, 2))
                    mH = ((YE[tgt] - predH) ** 2).sum((1, 2))
                    sh.append(float(mse.mean() / mH.mean() - 1))
                r["shuffle_vs_H_rel_change"] = sh
            res[tgt][name] = r
    return res


def stage_observer_select():
    famF, YF, nF, labF = observer_data("fit")
    famD, YD, nD, labD = observer_data("dev")
    hr = jload("headroom.json")["activation_prioritization"]
    lam_tab, cv = {}, {}
    for tgt in TARGETS:
        lam_tab[tgt], cv[tgt] = {}, {}
        for name in MODELS:
            l, err = cv_lambda(famF, YF, nF, MODELS[name], tgt)
            lam_tab[tgt][name], cv[tgt][name] = l, err
        print(f"  CV lambdas {tgt}: {lam_tab[tgt]}", flush=True)
    dev = evaluate_all(famF, YF, nF, famD, YD, nD, lam_tab, [m for m in MODELS if m != "H+C+R+E+S"], n_perm=NPERM)
    rc = lambda tgt, m: dev[tgt][m]["vs_H_rel_change"]["v"]
    carry_val = min(rc("eW", "H+C"), rc("etot", "H+C")) <= -0.01
    act_val = min(rc(t, m) for t in ("eAW", "etot") for m in ("H+R", "H+E", "H+R+E+S")) <= -0.01
    if carry_val and act_val:
        extra = evaluate_all(famF, YF, nF, famD, YD, nD, lam_tab, ["H+C+R+E+S"], n_perm=NPERM)
        for tgt in TARGETS:
            dev[tgt]["H+C+R+E+S"] = extra[tgt]["H+C+R+E+S"]
    act_allowed = hr["rule"] != "park activation-observer live arm"
    cands = ["H", "H+C"] + (["H+R", "H+E", "H+R+E+S"] if act_allowed else []) + (["H+C+R+E+S"] if carry_val and act_val else [])
    mse = {c: float(np.mean(dev["etot"][c]["mse_per_image"])) for c in cands}
    best = min(mse.values())
    tied = [c for c in cands if mse[c] <= best * 1.01]
    nf = lambda c: dev["etot"][c]["n_features"]
    sel = min(tied, key=lambda c: (nf(c), mse[c]))
    out = {"lambdas": lam_tab, "cv_error": cv, "dev": dev, "carry_family_dev_value": carry_val, "activation_family_dev_value": act_val,
           "activation_rule": hr["rule"], "activation_candidates_allowed": act_allowed, "candidates": cands, "dev_total_mse": mse, "selected": sel,
           "repeated_labels_fit": int(len(labF) - len(set(labF))), "repeated_labels_dev": int(len(labD) - len(set(labD))),
           "frozen_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime())}
    jdump("observer_select.json", out)
    summ = {t: {m: (round(v["ratio_raw"], 4), round(v.get("vs_H_rel_change", {}).get("v", 0), 4)) for m, v in dev[t].items()} for t in TARGETS}
    print(json.dumps({"summary(ratio_raw, vs_H)": summ, "selected": sel, "carry_val": carry_val, "act_val": act_val}, indent=1))


def stage_observer_confirm():
    s = jload("observer_select.json")
    famF, YF, nF, _ = observer_data("fit")
    famC, YC, nC, labC = observer_data("conf")
    models = [m for m in MODELS if m in s["dev"]["etot"]]
    conf = evaluate_all(famF, YF, nF, famC, YC, nC, s["lambdas"], models, n_perm=NPERM)
    sel = s["selected"]
    r = conf["etot"][sel]
    gate = {}
    if sel == "H":
        gate = {"pass": False, "reason": "H selected: no added internal features"}
    else:
        rel = r["vs_H_rel_change"]
        sh = np.array(r["shuffle_vs_H_rel_change"])
        gate = {"g1_10pct_below_H_ci_excl_0": rel["v"] <= -0.10 and rel["ci"][1] < 0,
                "g2_beats_M0": float(np.mean(r["mse_per_image"])) < float(np.mean(conf["etot"]["M0"]["mse_per_image"])),
                "g2_temporal_mean_reduced": r["temporal_mean_left_frac"] < 1.0,
                "g3_shuffle_loses_advantage": bool(rel["v"] < sh.min() and sh.mean() > rel["v"] / 2), "g4_finite": r["finite"]}
        gate["pass"] = all(gate.values())
    out = {"selected": sel, "conf": conf, "gate": gate, "repeated_labels_conf": int(len(labC) - len(set(labC))),
           "shuffle_note": "whole-trajectory derangements of the added features across images (same timestep), 20 seeds; class labels mostly unique, "
                           "so the shuffle does not separate class from image-specific information"}
    jdump("observer.json", {**s, **out})
    stage_headroom(("conf",))
    print(json.dumps({"gate": gate, "sel_vs_H": r.get("vs_H_rel_change")}, indent=1))


# ------------------------------------------------------------------------------------------------ cost
def stage_cost():
    rows = [r for r in csv.reader(open(os.path.join(RESULTS, "gpu_hours.csv"))) if r and r[0] == STAGE]
    p = jload("profile.json") or {}
    per_img_state = sum(2 * 4 * (4608 if k.endswith("fc2") else 1152) * 2 for k in KEYS)
    feat_bytes = {"C": 2 * 112 * 4 * 8 * 2, "R": 2 * 112 * 4 * 8 * 2, "S": 2 * 2 * 112 * 4 * 8 * 2, "E": 2 * 112 * len(EFEAT) * 4}
    pred_bytes = NB * (258 + NPC * 4) * 64 * 8 + NB * NPC * (14336 * 3 + 2240) * 8
    out = {"runs": [[r[1], r[3], r[4], float(r[5])] for r in rows], "total_gpu_h": float(sum(float(r[5]) for r in rows)), "cap_gpu_h": 2.0,
           "feedback_state_bytes_per_image": per_img_state, "feature_bytes_per_image_per_step": feat_bytes,
           "predictor_bytes_upper_bound (all families, 5 buckets, float64)": pred_bytes, "profile": {k: v for k, v in p.items() if k.endswith("_s") or k == "peak_mem_GiB"},
           "feature_cost_note": "trace per-step time includes the W and FP reference calls and float64 accounting; deployable feature extraction is the C/R/S/E "
                                "projections only (no extra model matmul for C: the bottleneck is the carry's own z)"}
    jdump("cost.json", out)
    print(json.dumps(out, indent=1))


def stage_report():
    subprocess.run(["python", os.path.join(ROOT, "scripts", "make_quantizer_observer.py")], check=True)
