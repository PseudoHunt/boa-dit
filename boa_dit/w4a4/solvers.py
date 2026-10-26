"""Per-layer W4A4 solvers (rule.md, Methods). Orientation as in the paper: W [c_in, c_out], y = X W; the branch is
X~ L1 L2 with L1 [c_in, r], L2 [r, c_out]; the residual R^ [c_in, c_out] is returned transposed for QLinear.

stats per layer: H0 [c_in, c_in] = E[x x^T] (all tokens, fp32), amax [c_in] (all tokens), Xs [65536, c_in] token
subsample (bf16).
"""
import math

import torch

from boa_dit.w4a4.fake import GS, QMAX, QMIN, q_act


def pinv_psd(A, rcond=1e-10):
    e, V = torch.linalg.eigh(A.double())
    keep = e > rcond * e.abs().max()
    inv = torch.where(keep, 1.0 / e.clamp_min(1e-300), torch.zeros_like(e))
    return ((V * inv) @ V.T).to(A.dtype)


def psd_pow(A, p):
    e, V = torch.linalg.eigh(A.double())
    e = e.clamp_min(e.max() * 1e-12)
    return ((V * e.pow(p)) @ V.T).to(A.dtype)


@torch.no_grad()
def gptq(T, M, eps_frac=0.01, block=128):
    """GPTQ (lazy batches) of target T [c_in, c_out] under metric M [c_in, c_in]: rows of T^T quantized with INT4
    symmetric groups of 64 along c_in, group scale from the current (updated) group columns when the group starts.
    Cholesky of M + eps I, eps = eps_frac * mean diag. Returns R^ [c_in, c_out] (dequantized)."""
    W = T.T.contiguous().float().clone()                                    # [c_out, c_in]
    n = W.shape[1]
    Ms = (M.double() + M.double().T) / 2
    eye = torch.eye(n, device=M.device, dtype=torch.float64)
    frac = eps_frac
    for _ in range(6):                                                      # damping raised x10 on failure (logged)
        try:
            Hinv = torch.cholesky_inverse(torch.linalg.cholesky(Ms + frac * Ms.diagonal().mean() * eye))
            U = torch.linalg.cholesky(Hinv, upper=True).float()
            break
        except RuntimeError:
            frac *= 10
    else:
        raise RuntimeError("GPTQ metric not positive-definite even at damping %g" % frac)
    GPTQ_DAMP.append(frac)
    Q = torch.zeros_like(W)
    for i1 in range(0, n, block):
        i2 = min(i1 + block, n)
        W1 = W[:, i1:i2].clone()
        E1 = torch.zeros_like(W1)
        U1 = U[i1:i2, i1:i2]
        s = None
        for i in range(i2 - i1):
            if (i1 + i) % GS == 0:
                s = W1[:, i:i + GS].abs().amax(1).clamp_min(1e-8) / QMAX
            w = W1[:, i]
            q = torch.clamp(torch.round(w / s), QMIN, QMAX) * s
            Q[:, i1 + i] = q
            err = (w - q) / U1[i, i]
            W1[:, i:] -= err[:, None] * U1[i, i:][None, :]
            E1[:, i] = err
        W[:, i2:] -= E1 @ U[i1:i2, i2:]
    return Q.T.contiguous()


GPTQ_DAMP = []   # damping fraction actually used by each gptq call (the driver records and clears it per layer)


def rtn(Wt):
    from boa_dit.w4a4.fake import q_weight
    return q_weight(Wt.T).T


def output_mse(Wt, Xs, d, L1, L2, R):
    """W4A4 output MSE on the token subsample: y_fp = X W vs X~ L1 L2 + Q_A(X~) R^ (bias cancels)."""
    X = Xs.float()
    yf = X @ Wt
    Xt = X / d if d is not None else X
    y = q_act(Xt) @ R
    if L1 is not None and L1.shape[1]:
        y = y + (Xt @ L1) @ L2
    return float((y - yf).pow(2).mean())


def act_surrogate(Xt, kappa=7.0):
    """(s_A)_i = E_t[max_{j in G(i)} X~_tj^2] / (12 kappa^2), groups of 64 channels."""
    g = Xt.float().reshape(Xt.shape[0], -1, GS).pow(2).amax(-1)              # [T, c_in/64]
    return (g.mean(0) / (12 * kappa ** 2)).repeat_interleave(GS)


def subspace(Ht_reg, Wt, r, kind, beta=0.5):
    """L1 [c_in, r] in the H~_reg-orthonormal gauge. kind: 'beta' (L^(beta) = H^-beta [H^beta W]_r), 'heig'."""
    if r == 0:
        return Wt.new_zeros(Wt.shape[0], 0)
    if kind == "heig":
        e, V = torch.linalg.eigh(Ht_reg.double())
        Vr, er = V[:, -r:], e[-r:]
        L1 = Vr / er.sqrt()                                                  # V_r Lambda_r^-1/2 (H~_reg-orthonormal)
        return L1.to(Wt.dtype)
    Hb = psd_pow(Ht_reg, beta) if beta else None
    M = Hb @ Wt if beta else Wt
    U = torch.linalg.svd(M.double(), full_matrices=False)[0][:, :r].to(Wt.dtype)
    L1 = psd_pow(Ht_reg, -beta) @ U if beta else U
    G = L1.T @ Ht_reg @ L1                                                   # re-gauge to L1^T H~_reg L1 = I
    return L1 @ psd_pow(G, -0.5)


def refit_L2(L1, Ht, Target):
    return pinv_psd(L1.T @ Ht @ L1) @ (L1.T @ Ht @ Target)


def deflate(H, A):
    """H - H A (A^T H A)^+ A^T H in fp64, symmetrized (fp32 cancellation broke PD-ness on PixArt's outlier channels)."""
    if A is None or A.shape[1] == 0:
        return H
    Hd, Ad = H.double(), A.double()
    HA = Hd @ Ad
    out = Hd - HA @ pinv_psd(Ad.T @ HA) @ HA.T
    return ((out + out.T) / 2).to(H.dtype)


@torch.no_grad()
def hsvdquant(W, st, r=4, lam_A=0.05, kappa=7.0, tau_frac=1e-2, eps_frac=0.01, kind="beta", beta=0.5, iters=2,
              smooth=True):
    """H-SVDQuant, Algorithm 1 (handoff §3). W [c_out, c_in] (nn.Linear layout). Returns dict for QLinear + info."""
    Wt = W.T.float()                                                         # [c_in, c_out]
    H0, Xs = st["H0"].float(), st["Xs"]
    A_prev, L2_prev, best = None, None, None
    info = {"iters": []}
    for k in range(iters):
        P_prev = Wt if A_prev is None else Wt - A_prev @ L2_prev
        Hp = H0 if A_prev is None else deflate(H0, A_prev)
        if smooth:
            h = Hp.diagonal().clamp_min(1e-12)
            pn = P_prev.norm(dim=1).clamp_min(1e-12)
            d = h.pow(0.25) / pn.pow(0.5)
            d = d / torch.exp(torch.log(d).mean())                           # geometric mean 1
        else:
            d = torch.ones(Wt.shape[0], device=Wt.device)
        Wtil = d[:, None] * Wt
        Ht = H0 / d[:, None] / d[None, :]
        Xt = Xs.float() / d
        sA = act_surrogate(Xt, kappa)
        Ht_reg = Ht + tau_frac * Ht.diagonal().mean() * torch.eye(Ht.shape[0], device=Ht.device)
        L1 = subspace(Ht_reg, Wtil, r, kind, beta)
        L2_0 = refit_L2(L1, Ht, Wtil) if r else None
        Hperp = deflate(Ht, L1) if r else Ht
        if lam_A > 0:
            MQ = Hperp + lam_A * torch.diag(sA)
            TQ = pinv_psd(MQ) @ (Hperp @ Wtil)
        else:
            MQ = Hperp
            TQ = Wtil - (L1 @ L2_0 if r else 0)
        R = gptq(TQ, MQ, eps_frac)
        L2 = refit_L2(L1, Ht, Wtil - R) if r else None
        score = output_mse(Wt, Xs, d, L1 if r else None, L2, R)
        it = {"k": k + 1, "output_mse": score, "trace_ratio_Hperp": float(Hperp.diagonal().sum() / Ht.diagonal().sum()),
              "d_range": [float(d.min()), float(d.max())]}
        info["iters"].append(it)
        if best is None or score < best["score"]:
            best = {"score": score, "R": R, "L1": L1 if r else None, "L2": L2, "d": d, "k": k + 1}
        A_prev, L2_prev = (L1 / d[:, None], L2) if r else (None, None)
        if not r:
            A_prev = None
    info["chosen_iter"] = best["k"]
    info["output_mse"] = best["score"]
    return {"R": best["R"].T.contiguous(), "L1": best["L1"], "L2": best["L2"], "d": best["d"] if smooth else None}, info


@torch.no_grad()
def gptq_r0(W, st, eps_frac=0.01):
    """GPTQ-r0: target W, metric H0, no smoothing, no branch."""
    Wt = W.T.float()
    R = gptq(Wt, st["H0"].float(), eps_frac)
    return {"R": R.T.contiguous(), "L1": None, "L2": None, "d": None}, {"output_mse": output_mse(Wt, st["Xs"], None, None, None, R)}


@torch.no_grad()
def rtn_layer(W, st):
    Wt = W.T.float()
    R = rtn(Wt)
    return {"R": R.T.contiguous(), "L1": None, "L2": None, "d": None}, {"output_mse": output_mse(Wt, st["Xs"], None, None, None, R)}


@torch.no_grad()
def svdquant(W, st, r=32, alphas=(0.3, 0.4, 0.5, 0.6, 0.7, 0.8), eps_frac=0.01):
    """SVDQuant reimplementation: SmoothQuant migration (alpha grid by W4A4 output MSE), SVD r of the smoothed weight
    (plain, L1 = U_r S_r, L2 = V_r^T), GPTQ on the residual against H~."""
    Wt = W.T.float()
    H0, Xs, amax = st["H0"].float(), st["Xs"], st["amax"].float().clamp_min(1e-6)
    wmax = Wt.abs().amax(1).clamp_min(1e-6)
    best = None
    for al in alphas:
        d = amax.pow(al) / wmax.pow(1 - al)
        d = d / torch.exp(torch.log(d).mean())
        Wtil = d[:, None] * Wt
        U, S, Vh = torch.linalg.svd(Wtil.double(), full_matrices=False)
        L1, L2 = (U[:, :r] * S[:r]).float(), Vh[:r].float()
        Ht = H0 / d[:, None] / d[None, :]
        R = gptq(Wtil - L1 @ L2, Ht, eps_frac)
        sc = output_mse(Wt, Xs, d, L1, L2, R)
        if best is None or sc < best["score"]:
            best = {"score": sc, "alpha": al, "R": R, "L1": L1, "L2": L2, "d": d}
    return ({"R": best["R"].T.contiguous(), "L1": best["L1"], "L2": best["L2"], "d": best["d"]},
            {"output_mse": best["score"], "alpha": best["alpha"]})


@torch.no_grad()
def alloc_spectrum(W, st, keep=256):
    """X2 input: e_k = lam_k ||v_k^T E||^2, E = W~ - GPTQ(W~; H~) with the base's iteration-1 smoothing
    (h = diag H0, P = W), (lam_k, v_k) eigenpairs of H~ descending. Returns raw e_k[:keep], total, r90."""
    Wt = W.T.float()
    H0 = st["H0"].float()
    d = H0.diagonal().clamp_min(1e-12).pow(0.25) / Wt.norm(dim=1).clamp_min(1e-12).pow(0.5)
    d = d / torch.exp(torch.log(d).mean())
    Wtil, Ht = d[:, None] * Wt, H0 / d[:, None] / d[None, :]
    E = Wtil - gptq(Wtil, Ht)
    lam, V = torch.linalg.eigh(Ht.double())
    lam, V = lam.flip(0).clamp_min(0), V.flip(1)
    e = lam * (V.T @ E.double()).pow(2).sum(1)
    cum = e.cumsum(0) / e.sum()
    return {"e_k": e[:keep].tolist(), "e_total": float(e.sum()), "r90": int((cum < 0.9).sum()) + 1}
