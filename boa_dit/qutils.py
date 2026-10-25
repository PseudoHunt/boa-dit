"""Round-to-nearest helpers and the BoA wrapper (no solver changes: we only call it)."""
import torch
import torch.nn as nn

from boa_dit import common  # noqa: F401  (sys.path for external/boa)
from quantizers.boa import BoA  # external/boa
from quantizers.minmax import MinMaxQuantizer, get_min_max_range
from utils.quant_utils import fake_quantize

# The fork's quantize.py disables TF32 on import of quantizers.boa; keep it that way.


@torch.no_grad()
def rtn_weight(W, bits, sym=False):
    """Per-output-channel min-max round-to-nearest (no grid search). Returns fake-quant W (same dtype)."""
    Wf = W.float()
    w_min, w_max = get_min_max_range(Wf, sym)
    maxq = torch.tensor(2 ** bits - 1, device=W.device)
    scale = (w_max - w_min) / maxq
    zero = torch.full_like(scale, (2 ** bits) / 2) if sym else torch.round(-w_min / scale)
    return fake_quantize(Wf, scale, zero, maxq).to(W.dtype)


@torch.no_grad()
def rtn_(linear: nn.Linear, bits, sym=False):
    linear.weight.data = rtn_weight(linear.weight.data, bits, sym)


@torch.no_grad()
def boa_gptq_rows(linear: nn.Linear, H_col, rows, bits, replace, sym=False, qparam_comput="Hessian",
                  act_order_col=False):
    """Quantize linear.weight[rows] with BoA's layer-output (GPTQ) path using H_col.

    BoA.gptq treats rows independently (per-row grid, per-row error compensation), so slicing rows
    and calling the solver once per row subset is exactly a per-row-group Hessian.
    """
    W = linear.weight.data
    rows = torch.as_tensor(rows, device=W.device, dtype=torch.long)
    sub = nn.Linear(W.shape[1], len(rows), bias=False, device=W.device, dtype=torch.float32)
    sub.weight.data = W[rows].float().clone()
    q = BoA(sub, {"qparam_comput": qparam_comput, "act_order_col": act_order_col, "act_order_row": False},
            {"replace": replace})
    q.quantizer = MinMaxQuantizer()
    q.quantizer.configure(bits, per_channel=True, sym=sym, mse=False)
    q.quantizer.find_params(sub.weight.data)  # same call order as the fork's quantize.py:76
    q.H_col = H_col.to(W.device, torch.float32).clone()
    q.H_row = None
    q.quant()
    W[rows] = sub.weight.data.to(W.dtype)
    q.free()


@torch.no_grad()
def boa_gptq_groups(linear: nn.Linear, groups, bits, replace, sym=False, qparam_comput="Hessian"):
    """Same result as calling boa_gptq_rows once per (rows, H_col) group, in ONE solver call.

    BoA's solver already supports a per-head H_col of shape [n_heads, d, d] with W viewed as
    [n_heads, head_dim, d], vectorized over heads (quantizers/boa.py:117-123). We pack each row group as
    one "head", zero-padding groups to equal size (all-zero rows quantize to zero and are discarded).
    The column loop then runs once instead of once per group.
    """
    if len(groups) == 1:
        rows, H = groups[0]
        return boa_gptq_rows(linear, H, rows, bits, replace, sym, qparam_comput)
    W = linear.weight.data
    d = W.shape[1]
    m = max(len(r) for r, _ in groups)
    packed = torch.zeros(len(groups) * m, d, device=W.device, dtype=torch.float32)
    idx = []
    for g, (rows, _) in enumerate(groups):
        rows = torch.as_tensor(rows, device=W.device, dtype=torch.long)
        packed[g * m:g * m + len(rows)] = W[rows].float()
        idx.append(rows)
    sub = nn.Linear(d, len(groups) * m, bias=False, device=W.device, dtype=torch.float32)
    sub.weight.data = packed
    q = BoA(sub, {"qparam_comput": qparam_comput, "act_order_col": False, "act_order_row": False},
            {"replace": replace})
    q.quantizer = MinMaxQuantizer()
    q.quantizer.configure(bits, per_channel=True, sym=sym, mse=False)
    q.quantizer.find_params(sub.weight.data)
    q.H_col = torch.stack([H.to(W.device, torch.float32) for _, H in groups])
    q.H_row = None
    q.quant()
    for g, rows in enumerate(idx):
        W[rows] = sub.weight.data[g * m:g * m + len(rows)].to(W.dtype)
    q.free()
