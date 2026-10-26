"""Simulated W4A4 (rule.md): INT4 symmetric, levels -8..7, scale = max|.|/7; weights in groups of 64 along the input
dim per output channel; activations per token in groups of 64 channels, dynamic. The quantized linear:
    y = X~ L1 L2 + Q_A(X~) R^T + b,   X~ = X / d   (d = smoothing vector; branch sees the unquantized X~)
"""
import torch
import torch.nn as nn

GS, QMAX, QMIN = 64, 7.0, -8.0


def q_int4(x, dim_groups=GS):
    """fake-quant along the last dim in groups of 64 (symmetric, per group max|x|/7). fp32 arithmetic."""
    sh = x.shape
    g = x.float().reshape(*sh[:-1], sh[-1] // dim_groups, dim_groups)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / QMAX
    return (torch.clamp(torch.round(g / s), QMIN, QMAX) * s).reshape(sh)


def q_weight(W):
    """W [c_out, c_in]: groups of 64 along c_in."""
    return q_int4(W)


def q_act(X):
    """X [..., c_in]: per token, groups of 64 channels."""
    return q_int4(X)


class QLinear(nn.Module):
    """W4A4 linear with optional smoothing and 16-bit low-rank branch. R is the fake-quantized residual weight
    (dequantized, stored bf16); L1 [c_in, r], L2 [r, c_out] in bf16; d [c_in] or None."""

    def __init__(self, lin, R, L1=None, L2=None, d=None, act_quant=True):
        super().__init__()
        self.register_buffer("R", R.to(torch.bfloat16))
        self.register_buffer("bias", None if lin.bias is None else lin.bias.detach().to(torch.bfloat16))
        self.register_buffer("L1", None if L1 is None else L1.to(torch.bfloat16))
        self.register_buffer("L2", None if L2 is None else L2.to(torch.bfloat16))
        self.register_buffer("d", None if d is None else d.float())
        self.act_quant = act_quant
        self.in_features, self.out_features = lin.in_features, lin.out_features

    def forward(self, x):
        dt = x.dtype
        xs = x.float() if self.d is None else x.float() / self.d
        xq = q_act(xs) if self.act_quant else xs
        y = torch.nn.functional.linear(xq.to(torch.bfloat16), self.R, self.bias)
        if self.L1 is not None and self.L1.shape[1] > 0:
            y = y + (xs.to(torch.bfloat16) @ self.L1) @ self.L2
        return y.to(dt)


def set_module(root, name, new):
    parts = name.split(".")
    m = root
    for p in parts[:-1]:
        m = m[int(p)] if p.isdigit() else getattr(m, p)
    if parts[-1].isdigit():
        m[int(parts[-1])] = new
    else:
        setattr(m, parts[-1], new)
