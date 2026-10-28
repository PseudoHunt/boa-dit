"""Part 3 (PROTOCOL.md): validation on small synthetic matrices (CPU). Exit 0 only if every check passes; otherwise the
sampling queue does not start. The identical-table sampler check (6) runs on GPU in arms.py before any arm.
Run: python -m boa_dit.weights_joint.validate
"""
import os

import torch
import torch.nn as nn

from boa_dit.common import atomic_write_json
from boa_dit.weights_joint.common import OUT
from boa_dit.weights_joint.joint import gptq_static, joint, levels, prep, quad, surrogate
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer
from quantizers.utils import get_cholesky_of_inverse


def problem(rows=6, d=24, n=200, seed=0):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(n, d, generator=g) @ torch.randn(d, d, generator=g) * 0.3
    H = (X.T @ X / n).float()
    W = torch.randn(rows, d, generator=g) * 0.5
    wmin, wmax = W.amin(1, keepdim=True), W.amax(1, keepdim=True)
    scale = (wmax - wmin) * 0.9 / 15                                          # grid narrower than the range: targets fall outside
    zero = torch.round(-wmin * 0.9 / scale)
    return W, H, scale, zero


def boa_gptq(W, H, scale, zero):
    """BoA's own preprocess + gptq loop on the given grid (the reference implementation)."""
    sub = nn.Linear(W.shape[1], W.shape[0], bias=False); sub.weight.data = W.clone()
    q = BoA(sub, {"qparam_comput": "Hessian", "act_order_col": False, "act_order_row": False}, {"replace": 1 / 256})
    q.quantizer = MinMaxQuantizer(); q.quantizer.configure(4, per_channel=True, sym=False, mse=False)
    q.quantizer.maxq = torch.tensor(15)
    q.H_col = H.clone(); q.H_row = None
    Wp, Hc, _ = q.preprocess()
    return q.gptq(Wp, Hc, scale.view(1, -1, 1), zero.view(1, -1, 1)).reshape(W.shape)


def main():
    res = {}
    W, H, scale, zero = problem()
    Wp, Hc, U = prep(W, H, 4, 1 / 256)
    d = W.shape[1]
    allv = torch.ones(d, dtype=torch.bool)
    # (2) telescoping: static and joint
    Q, _, cost = gptq_static(Wp, U, scale, zero, 15)
    res["telescoping_static_rel"] = abs(float(cost.sum()) - quad(Q - Wp, Hc)) / quad(Q - Wp, Hc)
    QA, QB, cA, cB, jc = joint(Wp, U, scale, zero, 15, 0.1, allv)
    Ls = surrogate(Wp, QA, QB, Hc, 0.1)
    res["telescoping_joint_rel"] = abs(float(jc.sum()) - Ls) / Ls
    # (1) one step: the GPTQ compensation and the pair score vs a direct solve of the remaining continuous quadratic
    Hd = Hc.double()
    Hinv = torch.linalg.inv(Hd)
    w = Wp[0].double(); q0 = float(QA[0, 0]); e0 = w[0] - q0
    shift_direct = -torch.linalg.solve(Hd[1:, 1:], Hd[1:, 0] * (q0 - w[0]))   # optimum of the free columns given column 0
    shift_gptq = -(e0 / U[0, 0].double()) * U[0, 1:].double()
    res["compensation_vs_direct_max_abs"] = float((shift_direct - shift_gptq).abs().max())
    dL_direct = float(e0 ** 2 / Hinv[0, 0])
    res["score_vs_direct_rel"] = abs(float((e0 / U[0, 0].double()) ** 2) - dL_direct) / dL_direct
    # (3) beta = 1 is the independent objective
    L1 = surrogate(Wp, QA, QB, Hc, 1.0)
    res["beta1_independent_rel"] = abs(L1 - (quad(QA - Wp, Hc) + quad(QB - Wp, Hc)) / 2) / L1
    # (4) identical init + deterministic ties + beta = 1 -> ordinary GPTQ (BoA's loop), both tables; static logger too
    ref = boa_gptq(W, H, scale, zero)
    QA1, QB1, *_ = joint(Wp, U, scale, zero, 15, 1.0, allv)
    res["beta1_equals_gptq_A"] = bool(torch.equal(QA1, ref)); res["beta1_equals_gptq_B"] = bool(torch.equal(QB1, ref))
    res["static_logged_equals_gptq"] = bool(torch.equal(Q, ref))
    # (5) table-swap symmetry: objective invariant; construction equivariant (different inits, exchanged + transposed ties)
    res["swap_objective_rel"] = abs(surrogate(Wp, QB, QA, Hc, 0.1) - Ls) / Ls
    Wq = Wp + 0.01 * torch.randn(Wp.shape, generator=torch.Generator().manual_seed(1))
    J1 = joint(Wp, U, scale, zero, 15, 0.1, allv, WB0=Wq)
    J2 = joint(Wq, U, scale, zero, 15, 0.1, allv, WB0=Wp, tie_transpose=True)
    res["swap_equivariant_exact"] = bool(torch.equal(J2[0], J1[1]) and torch.equal(J2[1], J1[0]))
    # (7) grid bounds and dtype round trips
    Lv = levels(scale, zero, 15)
    res["codes_in_range"] = bool(cA.min() >= 0 and cA.max() <= 15 and cB.min() >= 0 and cB.max() <= 15)
    res["dequant_exact"] = bool(torch.equal(Lv.gather(1, cA.long()), QA) and torch.equal(Lv.gather(1, cB.long()), QB))
    res["uint8_roundtrip"] = bool(torch.equal(cA.to(torch.uint8).to(torch.int16), cA))
    res["targets_outside_grid_present"] = bool(((Wp < Lv.min(1, keepdim=True).values) | (Wp > Lv.max(1, keepdim=True).values)).any())
    # selective variation: shared columns identical
    vary = torch.zeros(d, dtype=torch.bool); vary[-4:] = True
    QAs, QBs, *_ = joint(Wp, U, scale, zero, 15, 0.1, vary)
    res["selective_shared_cols_equal"] = bool(torch.equal(QAs[:, :-4], QBs[:, :-4]))
    res["selective_varied_fraction"] = float((QAs != QBs).float().mean())
    res["beta01_tables_differ_fraction"] = float((QA != QB).float().mean())
    tol = 1e-5
    checks = {"telescoping_static": res["telescoping_static_rel"] < tol, "telescoping_joint": res["telescoping_joint_rel"] < tol,
              "compensation_vs_direct": res["compensation_vs_direct_max_abs"] < 1e-5, "score_vs_direct": res["score_vs_direct_rel"] < 1e-5,
              "beta1_independent": res["beta1_independent_rel"] < tol,
              "beta1_recovers_gptq": res["beta1_equals_gptq_A"] and res["beta1_equals_gptq_B"],
              "static_logger_equals_gptq": res["static_logged_equals_gptq"], "swap_objective": res["swap_objective_rel"] < tol,
              "swap_equivariance": res["swap_equivariant_exact"],
              "grid_and_dtype": res["codes_in_range"] and res["dequant_exact"] and res["uint8_roundtrip"],
              "targets_outside_grid_exercised": res["targets_outside_grid_present"], "selective_shared": res["selective_shared_cols_equal"]}
    res["checks"], res["all_pass"] = checks, all(checks.values())
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, "validation_synthetic.json"), res)
    for k, v in checks.items():
        print(f"  {'PASS' if v else 'FAIL'}  {k}", flush=True)
    print({k: v for k, v in res.items() if isinstance(v, float)}, flush=True)
    print("ALL PASS" if res["all_pass"] else "VALIDATION FAILED", flush=True)
    raise SystemExit(0 if res["all_pass"] else 6)


if __name__ == "__main__":
    main()
