"""Parts 1-2 (PROTOCOL.md): instrumented static GPTQ (per-column incremental cost), joint two-table GPTQ, and static
extra precision on selected columns, all on heig-8's frozen per-row grid and deflated Hessian, with BoA's preprocessing
(dead-column replacement, 1% damping) and BoA's Cholesky convention: U = upper Cholesky of H_damped^-1, U_jj^2 = [H_F^-1]_jj.

Column-j cost (per row, summed over rows): static (w_j - q_j)^2 / U_jj^2; joint
  [(e_A + e_B)^2/4 + beta (e_A - e_B)^2/4] / U_jj^2,  e_X = w_X,j - q_X  (working weights after compensation).
Each table is compensated with the usual GPTQ update W_X[:, j:] -= (e_X / U_jj) ⊗ U[j, j:]. Targets are never clamped.
Telescoping: sum_j cost_j = tr(E H_damped E^T) for static, = L(Q_A, Q_B) for joint.
"""
import torch
import torch.nn as nn

from boa_dit import common  # noqa: F401  (sys.path for external/boa)
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from quantizers.utils import get_cholesky_of_inverse


def prep(W, H, bits, replace):
    """BoA.preprocess on (W, H): returns W [rows, d] fp32 (dead columns zeroed), H damped [d, d], U upper chol of H^-1."""
    sub = nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device, dtype=torch.float32)
    sub.weight.data = W.float().clone()
    q = BoA(sub, {"qparam_comput": "Hessian", "act_order_col": False, "act_order_row": False}, {"replace": replace})
    q.quantizer = MinMaxQuantizer()
    q.quantizer.configure(bits, per_channel=True, sym=False, mse=False)
    q.H_col = H.to(W.device, torch.float32).clone()
    q.H_row = None
    Wp, Hc, _ = q.preprocess()
    U = get_cholesky_of_inverse(Hc.clone())                                   # [1, d, d]; damps further only if needed
    return Wp.reshape(W.shape[0], W.shape[1]).contiguous(), Hc[0], U[0]


def levels(scale, zero, maxq, half=False):
    """[rows, n] dequantized grid levels scale*(c - zero), c = 0..maxq; half=True: the half-step grid on the same range."""
    c = torch.arange(0, 2 * maxq + 1 if half else maxq + 1, device=scale.device, dtype=torch.float32)
    step = scale / 2 if half else scale
    z = zero * 2 if half else zero
    return step * (c[None, :] - z)


@torch.no_grad()
def gptq_static(W, U, scale, zero, maxq, half_cols=None):
    """BoA's gptq loop with logged per-column cost; half_cols (bool [d]): those columns use the half-step grid (static
    extra precision). Returns Q [rows, d], codes [rows, d] (half-step codes on half columns), cost [d]."""
    W = W.clone()
    rows, d = W.shape
    L4, L5 = levels(scale, zero, maxq), levels(scale, zero, maxq, half=True)
    Q = torch.zeros_like(W)
    codes = torch.zeros(rows, d, dtype=torch.int16, device=W.device)
    cost = torch.zeros(d, dtype=torch.float64, device=W.device)
    for j in range(d):
        Lv = L5 if (half_cols is not None and bool(half_cols[j])) else L4
        e = W[:, j:j + 1] - Lv
        c = e.abs().argmin(1)                                                 # nearest legal level (BoA: round + clamp)
        q = Lv.gather(1, c[:, None])[:, 0]
        Q[:, j], codes[:, j] = q, c.to(torch.int16)
        err = (W[:, j] - q) / U[j, j]
        cost[j] = err.double().pow(2).sum()
        W[:, j:] -= err[:, None] * U[j, j:][None, :]
    return Q, codes, cost


@torch.no_grad()
def joint(W, U, scale, zero, maxq, beta, vary, WB0=None, tie_transpose=False):
    """Joint two-table GPTQ. vary: bool [d], columns allowed to differ (elsewhere q_A = q_B). Init W_A = W_B = W (the
    pilot's setting); WB0 / tie_transpose exist only for the swap-equivariance validation. Returns QA, QB, codesA,
    codesB, cost [d] (= the surrogate's column-j increase)."""
    WA, WB = W.clone(), (W if WB0 is None else WB0).clone()
    rows, d = W.shape
    Lv = levels(scale, zero, maxq)                                            # [rows, 16]
    n = Lv.shape[1]
    QA, QB = torch.zeros_like(W), torch.zeros_like(W)
    cA = torch.zeros(rows, d, dtype=torch.int16, device=W.device)
    cB = torch.zeros_like(cA)
    cost = torch.zeros(d, dtype=torch.float64, device=W.device)
    ar = torch.arange(rows, device=W.device)
    for j in range(d):
        eA = WA[:, j:j + 1] - Lv                                              # [rows, 16]
        eB = WB[:, j:j + 1] - Lv
        if bool(vary[j]):
            s = (eA[:, :, None] + eB[:, None, :]) / 2
            df = (eA[:, :, None] - eB[:, None, :]) / 2
            sc = s * s + beta * df * df                                       # [rows, a, b]
            if tie_transpose:
                k = sc.transpose(1, 2).reshape(rows, n * n).argmin(1)
                b, a = k // n, k % n
            else:
                k = sc.reshape(rows, n * n).argmin(1)                         # lowest flattened index on ties
                a, b = k // n, k % n
        else:
            s, df = (eA + eB) / 2, (eA - eB) / 2
            sc = s * s + beta * df * df
            a = sc.argmin(1)
            b = a
        qa, qb = Lv[ar, a], Lv[ar, b]
        QA[:, j], QB[:, j], cA[:, j], cB[:, j] = qa, qb, a.to(torch.int16), b.to(torch.int16)
        ea, eb = WA[:, j] - qa, WB[:, j] - qb
        cost[j] = (((ea + eb) / 2).double().pow(2) + beta * ((ea - eb) / 2).double().pow(2)).sum() / float(U[j, j]) ** 2
        WA[:, j:] -= (ea / U[j, j])[:, None] * U[j, j:][None, :]
        WB[:, j:] -= (eb / U[j, j])[:, None] * U[j, j:][None, :]
    return QA, QB, cA, cB, cost


def quad(E, H):
    return float(torch.einsum("ij,jk,ik->", E.double(), H.double(), E.double()))


def surrogate(W, QA, QB, H, beta):
    EA, EB = QA - W, QB - W
    return quad((EA + EB) / 2, H) + beta * quad((EA - EB) / 2, H)
