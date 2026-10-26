"""Deflate-and-carry: y = W_q x + b + delta (V^T x), with V [d_in, r] and delta = (W - W_q) V [d_out, r] in fp16.

The carry is a forward hook on the existing nn.Linear (which holds W_q and b), so state dicts, load_quant_state and the
samplers are unchanged. Carry files: results/deflate/quant/<arm>/carry.pt = {"b{l}.{layer}": {"V", "delta"}, ...}.
"""
import os

import torch

from boa_dit.common import RESULTS
from boa_dit.model import block_linear

QUANT = os.path.join(RESULTS, "deflate", "quant")
GATE1_QUANT = os.path.join(RESULTS, "gate1", "quant")   # Gate 1 engine arms (twosided/quantize_ts.py)
ENGINE_QUANT = os.path.join(RESULTS, "engine", "quant")  # engine round 2 arms (engine/quantize_b.py)


def carry_path(arm):
    for d in (ENGINE_QUANT, GATE1_QUANT):
        p = os.path.join(d, arm, "carry.pt")
        if os.path.exists(p):
            return p
    return os.path.join(QUANT, arm, "carry.pt")


@torch.no_grad()
def make_carry(W_fp, W_q, V):
    """delta = (W - W_q) V in fp32, stored fp16 with V."""
    delta = (W_fp.float() - W_q.float()) @ V.float()
    return {"V": V.half().cpu(), "delta": delta.half().cpu()}


def attach(linear, V, delta):
    """Register the carry on `linear`; returns the hook handle. Works under fp16 autocast and in fp32."""
    V, delta = V.to(linear.weight.device), delta.to(linear.weight.device)

    def hook(mod, inp, out):
        x = inp[0]
        dt = out.dtype
        return out + ((x.to(V.dtype) @ V) @ delta.T).to(dt) if dt == torch.float16 else \
            out + (x.float() @ V.float()) @ delta.float().T
    return linear.register_forward_hook(hook)


def install(model, carry):
    """Attach every entry of a carry dict ("b{l}.{layer}" -> {V, delta}) to the model's block linears."""
    hs = []
    for key, c in carry.items():
        l, ln = key.split(".", 1)
        hs.append(attach(block_linear(model.blocks[int(l[1:])], ln), c["V"], c["delta"]))
    return hs


def install_from(model, arm):
    return install(model, torch.load(carry_path(arm), map_location="cpu"))


def extra_bits(carry, n_params):
    """fp16 V and delta: 16 * r * (d_in + d_out) bits per layer, over the quantized weights."""
    per_type, tot = {}, 0
    for key, c in carry.items():
        b = 16 * (c["V"].numel() + c["delta"].numel())
        t = key.split(".", 1)[1]
        per_type[t] = per_type.get(t, 0) + b
        tot += b
    return tot / n_params, per_type
