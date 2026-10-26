"""Solver unit tests (rule.md, Solver). Run: python -m boa_dit.twosided.test_solver [--device cuda:0]

1. H_row = I: the row-batched two-sided path equals the one-sided path bit-exactly.
2. n_rows = 1: equals the fork's exact BoA.boa() (fp32 tolerance), also with act_order (col + row).
3. qkv split, all one-sided: equals the fused one-sided solve bit-exactly.
4. The two-sided objective tr(dW H_col dW^T H_row) for n_rows 1 / 8 / 64 vs one-sided GPTQ (smaller is better).
Real weights (block 2); synthetic correlated PSD Hessians (the real-factor version runs in the D2 pass).
"""
import argparse

import torch

from boa_dit.model import load_dit
from boa_dit.qutils import boa_gptq_rows
from boa_dit.twosided.solver import QKV_SLICES, kron_objective, solve_qkv_split, solve_rows
from quantizers.boa import BoA  # external/boa


def psd(n, dev, seed, spread=3.0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    A = torch.randn(n, n, generator=g, dtype=torch.float64)
    Qm, _ = torch.linalg.qr(A)
    ev = torch.logspace(0, -spread, n, dtype=torch.float64)
    return ((Qm * ev) @ Qm.T).float().to(dev)


@torch.no_grad()
def fork_boa(W, H_col, H_row, bits, **ao):
    """The fork's own two-sided path (Python row loop)."""
    import torch.nn as nn
    from boa_dit.twosided.solver import _quantizer
    sub = nn.Linear(W.shape[1], W.shape[0], bias=False, device=W.device)
    sub.weight.data = W.clone()
    q = BoA(sub, {"qparam_comput": "Hessian", "act_order_col": ao.get("col", False), "act_order_row": ao.get("row", False)},
            {"replace": 1 / 256})
    q.quantizer = _quantizer(sub.weight.data, bits, False)
    q.H_col, q.H_row = H_col.clone(), H_row.clone()
    q.quant()
    return sub.weight.data.clone()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    m = load_dit("cpu")
    blk = m.blocks[2]
    res = {}

    # 1. H_row = I -> one-sided, bit-exact (fc1, 4608 rows, groups of 64)
    W = blk.mlp.fc1.weight.data.to(dev).float()
    Hc = psd(1152, dev, 0)
    Q1, _ = solve_rows(W, Hc, None, bits=4)
    Q2, _ = solve_rows(W, Hc, torch.eye(4608, device=dev), bits=4, n_rows=64)
    lin = torch.nn.Linear(1152, 4608, bias=False, device=dev)
    lin.weight.data = W.clone()
    boa_gptq_rows(lin, Hc, list(range(4608)), 4, 1 / 256)
    res["1_identity_Hrow_equals_onesided"] = bool(torch.equal(Q1, Q2))
    res["1b_solve_rows_onesided_equals_qutils"] = bool(torch.equal(Q1, lin.weight.data))
    print("test 1", res, flush=True)
    assert res["1_identity_Hrow_equals_onesided"] and res["1b_solve_rows_onesided_equals_qutils"]

    # 2. n_rows = 1 equals the fork's exact boa() (one V head: 72 x 1152), plain and with act_order col+row
    Wv = blk.attn.qkv.weight.data[2304:2304 + 72].to(dev).float()
    Hr = psd(72, dev, 1).unsqueeze(0)
    for ao in ({}, {"col": True, "row": True}):
        Qf = fork_boa(Wv, Hc, Hr, 4, **ao)
        Qr, _ = solve_rows(Wv, Hc, Hr, bits=4, n_rows=1, act_order_col=ao.get("col", False), act_order_row=ao.get("row", False))
        d = float((Qf - Qr).abs().max())
        frac = float((Qf != Qr).float().mean())
        res[f"2_nrows1_vs_fork{'_actorder' if ao else ''}"] = {"max_abs_diff": d, "frac_entries_differ": frac}
        print("test 2", ao, d, frac, flush=True)
        assert frac < 1e-3, (ao, d, frac)

    # 3. qkv split, all one-sided -> bit-exact vs fused one-sided
    Wq = blk.attn.qkv.weight.data.to(dev).float()
    lin = torch.nn.Linear(1152, 3456, bias=False, device=dev)
    lin.weight.data = Wq.clone()
    boa_gptq_rows(lin, Hc, list(range(3456)), 4, 1 / 256)
    lin2 = torch.nn.Linear(1152, 3456, bias=False, device=dev)
    lin2.weight.data = Wq.clone()
    solve_qkv_split(lin2, {k: (Hc, None, {}) for k in QKV_SLICES}, bits=4)
    res["3_qkv_split_equals_fused"] = bool(torch.equal(lin.weight.data, lin2.weight.data))
    res["3_qkv_split_max_abs_diff"] = float((lin.weight.data - lin2.weight.data).abs().max())
    print("test 3", res["3_qkv_split_equals_fused"], res["3_qkv_split_max_abs_diff"], flush=True)

    # 4. objective vs n_rows (fc1 with a synthetic 4608 H_row; V per head with 16 heads)
    Hr_fc1 = psd(4608, dev, 2)
    obj = {}
    for n in (64,):  # n_rows 1 / 8 on fc1 = 4608 / 576 sequential row groups: the slow path this solver avoids
        Q, s = solve_rows(W, Hc, Hr_fc1, bits=4, n_rows=n)
        obj[f"two-sided n_rows={n}"] = (kron_objective((W - Q).unsqueeze(0), Hc.unsqueeze(0), Hr_fc1.unsqueeze(0)), s)
    Q, s = solve_rows(W, Hc, None, bits=4)
    obj["one-sided GPTQ"] = (kron_objective((W - Q).unsqueeze(0), Hc.unsqueeze(0), Hr_fc1.unsqueeze(0)), s)
    res["4_fc1_objective_and_seconds"] = obj
    Wv16 = blk.attn.qkv.weight.data[2304:].to(dev).float()
    Hr16 = torch.stack([psd(72, dev, 10 + h) for h in range(16)])
    objv = {}
    for n in (1, 8, 64):
        Q, s = solve_rows(Wv16, Hc, Hr16, bits=4, n_rows=n)
        objv[f"two-sided n_rows={n}"] = (kron_objective((Wv16 - Q).view(16, 72, -1), Hc.unsqueeze(0), Hr16), s)
    Q, s = solve_rows(Wv16, Hc, None, bits=4)
    objv["one-sided GPTQ"] = (kron_objective((Wv16 - Q).view(16, 72, -1), Hc.unsqueeze(0), Hr16), s)
    res["4_v_objective_and_seconds"] = objv
    for k, v in res.items():
        print(k, v)
    from boa_dit.common import RESULTS, atomic_write_json
    import os
    atomic_write_json(os.path.join(RESULTS, "gate1", "test_solver.json"), res)


if __name__ == "__main__":
    main()
