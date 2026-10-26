"""Two-sided (Kronecker) BoA solves, row-batched (TurboBoA's n_quant_rows scheme), plus the qkv split helper.

The fork's BoA.boa() quantizes one row at a time in a Python loop (each row a full GPTQ column loop), which is too slow
for fc1's 4,608 rows. RowBatchedBoA replaces only boa(): rows are taken in groups of n_rows; inside a group each row runs
the fork's GPTQ column loop with H_col (no row-to-row compensation inside the group); after the group, the remaining
rows are compensated with U_row^T[r2:, g] U_gg^-T (W_g - Q_g), the OBS update for the Kronecker Hessian
H_row (x) H_col. n_rows = 1 is the fork's exact boa(). Grid search, dead-column filtering, damping and act_order are the
fork's own (BoA.quant / preprocess), unchanged.
"""
import time

import torch
import torch.nn as nn

from boa_dit import common  # noqa: F401  (sys.path for external/boa)
from quantizers.boa import BoA  # external/boa
from quantizers.minmax import MinMaxQuantizer
from quantizers.utils import get_cholesky_of_inverse
from utils.quant_utils import fake_quantize

C = 1152
QKV_SLICES = {"q": slice(0, C), "k": slice(C, 2 * C), "v": slice(2 * C, 3 * C)}


class GroupGPTQGraph:
    """The fork's BoA.gptq column loop (same ops, same order) for one row group, captured once as a CUDA graph and
    replayed per group: the loop is ~10k tiny kernels per group, so launch overhead, not arithmetic, set the cost.
    U_col is computed once per layer (the fork recomputes the identical factor inside every gptq call)."""

    def __init__(self, n_heads, n, d, U_col, maxq, dev):
        z = lambda *s: torch.zeros(*s, device=dev, dtype=torch.float32)
        self.W, self.s, self.z, self.Q, self.E = z(n_heads, n, d), z(n_heads, n, 1), z(n_heads, n, 1), z(n_heads, n, d), z(n_heads, n, d)
        self.U, self.maxq, self.d = U_col, float(maxq), d
        st = torch.cuda.Stream()
        st.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(st):
            self._body()                                                          # warm-up before capture
        torch.cuda.current_stream().wait_stream(st)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g):
            self._body()

    def _body(self):
        W, U = self.W, self.U
        for j in range(self.d):
            w = W[..., j].unsqueeze(-1)
            q = fake_quantize(w, self.s, self.z, self.maxq)
            self.Q[..., j] = q.squeeze(-1)
            err = (w - q) / U[..., j, j][:, None, None]
            self.E[..., j] = err.squeeze(-1)
            W[..., j:] -= err @ U[..., j, j:].unsqueeze(-2)

    def run(self, W_g, s, z):
        self.W.copy_(W_g); self.s.copy_(s); self.z.copy_(z)
        self.g.replay()
        return self.Q.clone(), self.E.clone()


class RowBatchedBoA(BoA):
    n_rows = 64
    use_graph = True

    def boa(self, W, H_col, H_row, scale, zero):
        U_col = get_cholesky_of_inverse(H_col)
        U_row = get_cholesky_of_inverse(H_row)
        n_heads, head_dim, d = W.shape
        Q = torch.zeros_like(W)
        N = self.n_rows
        graphs = {}
        for r1 in range(0, head_dim, N):
            r2 = min(r1 + N, head_dim)
            if self.use_graph:
                if r2 - r1 not in graphs:
                    graphs[r2 - r1] = GroupGPTQGraph(n_heads, r2 - r1, d, U_col, self.quantizer.maxq, W.device)
                Q_g, Err = graphs[r2 - r1].run(W[:, r1:r2, :], scale[:, r1:r2, :], zero[:, r1:r2, :])
            else:
                Q_g, Err = self.gptq(W[:, r1:r2, :], H_col, scale[:, r1:r2, :], zero[:, r1:r2, :], return_err=True)
            Q[:, r1:r2, :] = Q_g
            if r2 < head_dim:
                U_gg = U_row[:, r1:r2, r1:r2]
                eye = torch.eye(r2 - r1, device=W.device, dtype=W.dtype).expand(n_heads, -1, -1)
                U_gg_inv = torch.linalg.solve_triangular(U_gg, eye, upper=True)
                P2 = U_row.transpose(-1, -2)[:, r2:, r1:r2] @ U_gg_inv.transpose(-1, -2)    # [h, rest, g]
                # W_g - Q_g == Err @ U_col (GPTQ identity); the fork's boa() uses Err @ U_col, so do the same
                W[:, r2:, :] -= P2 @ (Err @ U_col)
        return Q


def _quantizer(rows_weight, bits, sym):
    q = MinMaxQuantizer()
    q.configure(bits, per_channel=True, sym=sym, mse=False)
    q.find_params(rows_weight)  # same call order as qutils.boa_gptq_rows / the fork's quantize.py
    return q


@torch.no_grad()
def solve_rows(W_rows, H_col, H_row=None, bits=4, replace=1 / 256, sym=False, act_order_col=False,
               act_order_row=False, n_rows=64, qparam_comput="Hessian"):
    """Quantize W_rows [R, d] (fp32). H_col: [d, d] or per-head [h, d, d]. H_row: None (one-sided, the fork's GPTQ
    path: identical to qutils.boa_gptq_rows) or [h, R/h, R/h]. Returns (Q [R, d] fp32, seconds)."""
    torch.cuda.synchronize()
    t0 = time.time()
    dev = W_rows.device
    sub = nn.Linear(W_rows.shape[1], W_rows.shape[0], bias=False, device=dev, dtype=torch.float32)
    sub.weight.data = W_rows.float().clone()
    cls = BoA if H_row is None else RowBatchedBoA
    q = cls(sub, {"qparam_comput": qparam_comput, "act_order_col": act_order_col,
                  "act_order_row": act_order_row and H_row is not None}, {"replace": replace})
    if H_row is not None:
        q.n_rows = n_rows
    q.quantizer = _quantizer(sub.weight.data, bits, sym)
    q.H_col = H_col.to(dev, torch.float32).clone()
    q.H_row = None if H_row is None else H_row.to(dev, torch.float32).clone()
    if q.H_row is not None and q.H_row.dim() == 2:
        q.H_row = q.H_row.unsqueeze(0)
    q.quant()
    out = sub.weight.data.float().clone()
    q.free()
    torch.cuda.synchronize()
    return out, time.time() - t0


@torch.no_grad()
def solve_linear(lin, H_col, H_row=None, **kw):
    """Whole layer: writes the fake-quant weight back into `lin`. Returns seconds."""
    Q, s = solve_rows(lin.weight.data, H_col, H_row, **kw)
    lin.weight.data = Q.to(lin.weight.dtype)
    return s


@torch.no_grad()
def solve_qkv_split(lin, specs, **kw):
    """qkv split helper: q, k, v row blocks solved separately and written back into the fused weight.
    specs: {"q"|"k"|"v": (H_col, H_row or None, extra kwargs dict)}. Returns {slice name: seconds}."""
    W = lin.weight.data
    out, t = {}, {}
    for name, sl in QKV_SLICES.items():
        H_col, H_row, extra = specs[name]
        out[name], t[name] = solve_rows(W[sl], H_col, H_row, **{**kw, **extra})
    for name, sl in QKV_SLICES.items():
        W[sl] = out[name].to(W.dtype)
    return t


def kron_objective(dW, H_col, H_row):
    """tr(dW H_col dW^T H_row) summed over heads; dW [h, r, d], H_col [1|h, d, d], H_row [h, r, r] (fp64)."""
    dW, H_col, H_row = dW.double(), H_col.double(), H_row.double()
    return float(((H_row @ dW @ H_col) * dW).sum())
