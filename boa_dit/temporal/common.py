"""Temporal phase (results/temporal/rule.md): paths, step-indexed weight tables for DiT, bootstrap helpers, C/I.

A DiT arm is a list of per-step "tables": tables[k] = {layer key: (W_q, delta)}; at sampling step i the linear uses
tables[i % len(tables)] (weights and carry switch together). V is shared by every table of a layer.
"""
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS
from boa_dit.model import block_linear
from boa_dit.sample import CTX

STAGE = "temporal"
OUT = os.path.join(RESULTS, "temporal")
CK = os.path.join(CKPT_DIR, "temporal")
HEIG = "deflate_heig_r8_w4"
LAYERS_FILE = os.path.join(CK, "heig8_layers.pt")      # per layer: PHP (solver input), scale, zero, V (fp32)
PAIR_FILE = os.path.join(CK, "pair.pt")                # per layer: Wq1, Wq2, Wq3 (+ sparse variants)


class StepTables:
    """Installs step-indexed (W_q, delta) tables on the 112 block linears. tables: list of {key: (Wq, delta)}.
    V: {key: V [d_in, r]}. Weights are swapped by pointer in a forward pre-hook; the carry adds x V delta^T."""

    def __init__(self, model, tables, V):
        self.hooks = []
        dev = next(model.parameters()).device
        for key in V:
            l, ln = key.split(".", 1)
            lin = block_linear(model.blocks[int(l[1:])], ln)
            Ws = [t[key][0].to(dev, lin.weight.dtype) for t in tables]
            Ds = [t[key][1].to(dev) for t in tables]
            Vk = V[key].to(dev)
            self.hooks.append(lin.register_forward_pre_hook(self._pre(Ws)))
            self.hooks.append(lin.register_forward_hook(self._post(Vk, Ds)))

    @staticmethod
    def _k(n):
        return 0 if CTX.step is None else CTX.step % n

    def _pre(self, Ws):
        def pre(mod, inp):
            mod.weight.data = Ws[self._k(len(Ws))]
        return pre

    def _post(self, V, Ds):
        def hook(mod, inp, out):                                              # same arithmetic as deflate.carry.attach
            delta = Ds[self._k(len(Ds))]
            x = inp[0]
            dt = out.dtype
            return out + ((x.to(V.dtype) @ V) @ delta.T).to(dt) if dt == torch.float16 else \
                out + (x.float() @ V.float()) @ delta.float().T
        return hook

    def remove(self):
        for h in self.hooks:
            h.remove()


def ci_split(E):
    """E [P, T, ...] per-step errors -> C = sum_p T ||mean_t e||^2, I = sum ||e - mean_t e||^2."""
    E = E.double()
    m = E.mean(1, keepdim=True)
    C = float(E.shape[1] * m.pow(2).sum())
    I = float((E - m).pow(2).sum())
    if C + I == 0:                                                            # bit-exact floor
        return {"C": 0.0, "I": 0.0, "C_over_I": None, "C_share": None}
    return {"C": C, "I": I, "C_over_I": C / I, "C_share": C / (C + I)}


def boot_reduction(arm, ref, n=2000):
    """1 - mean(arm)/mean(ref) with paired bootstrap 95% CI over items."""
    a, b = np.asarray(arm, dtype=np.float64), np.asarray(ref, dtype=np.float64)
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        rs.append(1 - a[i].mean() / b[i].mean())
    return {"reduction": float(1 - a.mean() / b.mean()), "ci": np.percentile(rs, [2.5, 97.5]).tolist(), "n": len(a)}


def boot_ratio(a, b, n=2000):
    """mean(a)/mean(b), paired bootstrap 95% CI."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(0)
    rs = [a[i].mean() / b[i].mean() for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return {"ratio": float(a.mean() / b.mean()), "ci": np.percentile(rs, [2.5, 97.5]).tolist()}


def boot_diff(a, b, n=2000):
    """mean(a - b), paired bootstrap 95% CI."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    rng = np.random.default_rng(0)
    rs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return {"diff": float(d.mean()), "ci": np.percentile(rs, [2.5, 97.5]).tolist()}
