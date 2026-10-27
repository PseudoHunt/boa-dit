"""A1 (+ A3 tables) (rule.md): complementary roundings on heig-8's grid, from the per-layer state saved by rebuild.py.
  W_q2  : GPTQ (BoA's gptq loop, natural column order) of T = 2W - W_q1 clamped to the grid range, W_q1's grid, same PHP
  W_q3  : GPTQ of W, same grid and PHP, reverse column order (the "merely different" control)
  check : the same fixed-grid solver on W, natural order, must reproduce W_q1
  W_q2s{k} (A3): W_q1 with the k weights per row of largest |E1[r,j]| ||V[j,:]|| flipped to the grid level on the
          other side of W[r,j] (inside the grid)
Every table is stored as uint8 codes + the layer grid (W = scale * (code - zero), BoA's fake_quantize arithmetic) with
its carry delta = (W - W_q) V (deflate.carry.make_carry). Output ckpt/temporal/pair.pt, results/temporal/pair.json.
Run: python -m boa_dit.temporal.pair --device cuda:0
"""
import argparse
import os

import torch
import torch.nn as nn

from boa_dit import common as C
from boa_dit.deflate.carry import make_carry
from boa_dit.model import block_linear, load_dit, load_quant_state
from boa_dit.temporal.common import HEIG, LAYERS_FILE, OUT, PAIR_FILE, STAGE
from quantizers.boa import BoA
from quantizers.minmax import MinMaxQuantizer

SPARSE_K = (8, 16)


@torch.no_grad()
def gptq_fixed(T, H, scale, zero, bits, replace, reverse=False, perm=None):
    """BoA.quant without the grid search: preprocess (dead-column filter + damping) and the gptq column loop on the
    given per-row grid, in natural, reverse, or a given column order. T [rows, d] target; returns W_q [rows, d] fp32."""
    sub = nn.Linear(T.shape[1], T.shape[0], bias=False, device=T.device, dtype=torch.float32)
    sub.weight.data = T.float().clone()
    q = BoA(sub, {"qparam_comput": "Hessian", "act_order_col": False, "act_order_row": False}, {"replace": replace})
    q.quantizer = MinMaxQuantizer()
    q.quantizer.configure(bits, per_channel=True, sym=False, mse=False)
    q.quantizer.maxq = q.quantizer.maxq.to(T.device)
    q.H_col = H.to(T.device, torch.float32).clone()
    q.H_row = None
    W, Hc, _ = q.preprocess()                                                 # W [1, rows, d], Hc [1, d, d]
    s, z = scale.to(T.device).view(1, -1, 1), zero.to(T.device).view(1, -1, 1)
    if reverse:
        perm = torch.arange(W.shape[-1] - 1, -1, -1, device=T.device)
    if perm is not None:
        perm = perm.to(T.device)
        W, Hc = W[..., perm].contiguous(), Hc[:, perm][:, :, perm].contiguous()
    Q = q.gptq(W, Hc, s, z)
    if perm is not None:
        Q = Q[..., torch.argsort(perm)]
    q.free()
    return Q.reshape(T.shape)


def codes(Wq, scale, zero, maxq):
    c = torch.round(Wq / scale + zero).clamp(0, maxq)
    assert torch.equal(scale * (c - zero), Wq), "table is not on the grid"
    return c.to(torch.uint8)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    S = torch.load(LAYERS_FILE, map_location="cpu")
    W_fp = load_dit("cpu")
    W_q = load_quant_state(load_dit("cpu"), os.path.join(C.CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    out, info = {}, {}
    with C.Run(STAGE, "a1_pair", dev, vars(a)) as run:
        for key, s in S.items():
            l, ln = key.split(".", 1)
            Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            Wq1 = block_linear(W_q.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            scale, zero, H, V = s["scale"].to(dev), s["zero"].to(dev), s["H"].to(dev), s["V"].to(dev)
            maxq = 2 ** s["bits"] - 1
            lo, hi = scale * (0 - zero), scale * (maxq - zero)
            chk = gptq_fixed(Wf, H, scale, zero, s["bits"], s["replace"])
            T = torch.minimum(torch.maximum(2 * Wf - Wq1, lo), hi)
            Wq2 = gptq_fixed(T, H, scale, zero, s["bits"], s["replace"])
            Wq3 = gptq_fixed(Wf, H, scale, zero, s["bits"], s["replace"], reverse=True)
            tabs = {"Wq1": Wq1, "Wq2": Wq2, "Wq3": Wq3}
            E1 = Wf - Wq1                                                     # A3: sparse flips of W_q1
            score = E1.abs() * V.norm(dim=1)[None, :]
            flip = Wq1 + torch.sign(E1) * scale
            ok = (flip >= lo - 1e-12) & (flip <= hi + 1e-12) & (E1 != 0)
            score = torch.where(ok, score, torch.full_like(score, -1.0))
            for k in SPARSE_K:
                idx = score.topk(k, dim=1).indices
                m = torch.zeros_like(Wq1, dtype=torch.bool).scatter_(1, idx, True) & ok
                tabs[f"Wq2s{k}"] = torch.where(m, scale * (torch.round(flip / scale + zero) - zero), Wq1)
            Hd = H.double()
            P = torch.eye(H.shape[0], device=dev, dtype=torch.float64) - V.double() @ V.double().T
            Eeff = {n: ((Wf - t).double() @ P) for n, t in tabs.items()}
            tr = lambda E: float(torch.einsum("ij,jk,ik->", E, Hd, E))
            e1 = tr(Eeff["Wq1"])
            info[key] = {"solver_mismatch_frac": float((chk != Wq1).float().mean()),
                         "frac_changed_q2": float((Wq2 != Wq1).float().mean()), "frac_changed_q3": float((Wq3 != Wq1).float().mean()),
                         "rho_pair_12": tr(Eeff["Wq1"] + Eeff["Wq2"]) / e1, "rho_pair_13": tr(Eeff["Wq1"] + Eeff["Wq3"]) / e1,
                         "err_ratio_q2_over_q1": tr(Eeff["Wq2"]) / e1, "err_ratio_q3_over_q1": tr(Eeff["Wq3"]) / e1,
                         **{f"rho_pair_1s{k}": tr(Eeff["Wq1"] + Eeff[f"Wq2s{k}"]) / e1 for k in SPARSE_K},
                         **{f"frac_changed_s{k}": float((tabs[f"Wq2s{k}"] != Wq1).float().mean()) for k in SPARSE_K}}
            rec = {"scale": scale.cpu(), "zero": zero.cpu(), "V": s["V"].half()}
            for n, t in tabs.items():
                rec["c_" + n] = codes(t, scale, zero, maxq).cpu()
                rec["d_" + n] = make_carry(Wf, t, V)["delta"]
            out[key] = rec
            print(key, {k: round(v, 4) for k, v in info[key].items()}, flush=True)
        C.atomic_torch_save(out, PAIR_FILE)
        rec = run.record()
    n = len(info)
    summ = {f"mean_{k}": sum(v[k] for v in info.values()) / n for k in next(iter(info.values()))}
    summ["max_solver_mismatch_frac"] = max(v["solver_mismatch_frac"] for v in info.values())
    summ["d_in_bits_s"] = {k: {"attn.qkv/proj/fc1 (d_in 1152)": k * (11 + 1) / 1152, "fc2 (d_in 4608)": k * (13 + 1) / 4608}
                           for k in SPARSE_K}
    C.atomic_write_json(os.path.join(OUT, "pair.json"), {"summary": summ, "layers": info, **rec})
    print(summ, flush=True)


if __name__ == "__main__":
    main()
