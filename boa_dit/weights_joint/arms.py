"""Parts 4-5 (PROTOCOL.md): sampled arms on the design seeds 4000-4031 (one batch of 32, fp16 autocast, high-precision
activations), weight tables swapped per denoiser call (same table for both CFG branches; temporal.common.StepTables).
Before any arm: identical tables alternated must give latents bit-identical to the static model (validation 6).
  arm specs: A (static heig-8), indep (W_q1/W_q3 alternating, the existing independent pair), <cons> (Q_A/Q_B alternating),
             <cons>:A, <cons>:B (static constituents), <cons>:mean (static (Q_A+Q_B)/2), <cons>:hp (static half-step on S)
Per arm: endpoint latent rel. MSE vs the FP latents (per seed), paired bootstrap vs A; prediction-error energy, C, I,
C/I on the design FP trajectory; sampling wall time; storage accounting. Results append to arms.json.
Run: python -m boa_dit.weights_joint.arms --arms A,indep,B_final1,...
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, Run, atomic_write_json
from boa_dit.model import labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample
from boa_dit.temporal.common import PAIR_FILE, StepTables
from boa_dit.weights_joint.common import CK, DESIGN, HEIG, OUT, STAGE, boot_reduction, diagnostics, fp_traj, guard, rel_mse

N_W = 445_906_944
ARMS_JSON = os.path.join(OUT, "arms.json")


def deq(scale, zero, codes, half=None):
    """scale*(code - zero) in fp32 (BoA fake_quantize arithmetic); half [d] bool: half-step codes on those columns."""
    c = codes.float()
    if half is None:
        return scale * (c - zero)
    full = scale * (c - zero)
    hs = (scale / 2) * (c - 2 * zero)
    return torch.where(half[None, :], hs, full)


def tables_for(spec):
    """-> (list of {key: (W, delta)}, V dict, storage dict)."""
    P = torch.load(PAIR_FILE, map_location="cpu")
    V = {k: p["V"] for k, p in P.items()}
    keys = list(P.keys())
    rows = {k: P[k]["c_Wq1"].shape[0] for k in keys}
    dins = {k: P[k]["c_Wq1"].shape[1] for k in keys}
    carry_V = sum(v.numel() for v in V.values()) * 16
    carry_d = sum(P[k]["d_Wq1"].numel() for k in keys) * 16
    sz = sum(rows[k] * 32 for k in keys)                                      # fp16 scale + fp16 zero per output row
    st = {"scale_zero_bits": sz, "carry_V_bits": carry_V}

    def store(n_tables_full, extra_bits=0, n_deltas=1, entropy_est=None, actual_bytes=None):
        bits = n_tables_full * 4 * N_W + extra_bits + sz + carry_V + n_deltas * carry_d
        d = {**st, "packed_bits_total": bits, "bits_per_weight": bits / N_W, "carry_delta_bits_each": carry_d, "n_deltas": n_deltas,
             "runtime_buffer_bytes_simulation": None}
        if entropy_est is not None:
            d["entropy_estimate_bits_per_weight_second_table"] = entropy_est
        if actual_bytes is not None:
            d["actual_serialized_bytes_codes"] = actual_bytes
        return d
    w = lambda p, n: p["scale"] * (p["c_" + n].float() - p["zero"])
    if spec == "A":
        T = [{k: (w(P[k], "Wq1"), P[k]["d_Wq1"]) for k in keys}]
        return T, V, store(1, actual_bytes=N_W)
    if spec == "indep":
        T = [{k: (w(P[k], n), P[k]["d_" + n]) for k in keys} for n in ("Wq1", "Wq3")]
        return T, V, store(2, n_deltas=2, actual_bytes=2 * N_W)
    cons, _, mode = spec.partition(":")
    if mode == "hp":
        H = torch.load(os.path.join(CK, f"hp_{cons}.pt"), map_location="cpu")
        T = [{k: (deq(H["scale"][k], H["zero"][k], H["layers"][k]["codes"], H["layers"][k]["half_cols"]), H["layers"][k]["delta"]) for k in keys}]
        extra = sum(int(H["layers"][k]["half_cols"].sum()) * rows[k] for k in keys)          # +1 bit on the half-step columns
        meta = sum(int(H["layers"][k]["half_cols"].sum()) * math.ceil(math.log2(dins[k])) for k in keys)
        return T, V, {**store(1, extra_bits=extra + meta), "half_step_columns": sum(int(H["layers"][k]["half_cols"].sum()) for k in keys),
                      "selection_metadata_bits": meta}
    C = torch.load(os.path.join(CK, f"{cons}.pt"), map_location="cpu")
    A = {k: (deq(C["scale"][k], C["zero"][k], C["layers"][k]["cA"]), C["layers"][k]["dA"]) for k in keys}
    B = {k: (deq(C["scale"][k], C["zero"][k], C["layers"][k]["cB"]), C["layers"][k]["dB"]) for k in keys}
    nsel = {k: int(C["sel"][k].sum()) for k in keys}
    meta = sum(nsel[k] * math.ceil(math.log2(dins[k])) for k in keys) if cons != "F_all" else 0
    diff = [(C["layers"][k]["cB"].long() - C["layers"][k]["cA"].long()) for k in keys]
    nd = sum(int((x != 0).sum()) for x in diff)
    vals, cnt = torch.unique(torch.cat([x.flatten() for x in diff]), return_counts=True)
    p = cnt.double() / cnt.sum()
    H = float(-(p * torch.log2(p)).sum())
    if mode == "A":
        return [A], V, {**store(1), "frac_codes_differ_pair": nd / N_W}
    if mode == "B":
        return [B], V, {**store(1), "frac_codes_differ_pair": nd / N_W}
    if mode == "mean":
        M = {k: ((A[k][0] + B[k][0]) / 2, ((A[k][1].float() + B[k][1].float()) / 2).half()) for k in keys}
        sel_cols = sum(nsel[k] * rows[k] for k in keys)
        return [M], V, {**store(1, extra_bits=sel_cols + meta), "note": "half-step (5-bit) codes on the selected columns; no requantization",
                        "frac_codes_differ_pair": nd / N_W, "selection_metadata_bits": meta}
    second = 4 * N_W if cons == "F_all" else sum(nsel[k] * rows[k] * 4 for k in keys)          # second table: only its varied columns
    return [A, B], V, {**store(1, extra_bits=second + meta, n_deltas=2, entropy_est=H, actual_bytes=2 * N_W),
                       "frac_codes_differ_pair": nd / N_W, "selection_metadata_bits": meta, "selected_columns": sum(nsel.values())}


@torch.no_grad()
def check_identical(m, d, dev):
    """Validation 6: [Q1, Q1] alternated == [Q1] static, bit for bit (2 seeds)."""
    P = torch.load(PAIR_FILE, map_location="cpu")
    V = {k: p["V"] for k, p in P.items()}
    T = {k: (p["scale"] * (p["c_Wq1"].float() - p["zero"]), p["d_Wq1"]) for k, p in P.items()}
    s = DESIGN[:2]
    outs = []
    for tabs in ([T], [T, T]):
        st = StepTables(m, tabs, V)
        outs.append(ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5).cpu())
        st.remove()
    return bool(torch.equal(outs[0], outs[1]))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    res = json.load(open(ARMS_JSON)) if os.path.exists(ARMS_JSON) else {"seeds": DESIGN, "arms": {}, "label": "exploratory 32-seed screen"}
    m, d = load_dit(dev), make_diffusion(50)
    load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))     # heig-8 state incl. W8 adaLN
    if "identical_tables_bitexact" not in res:
        with Run(STAGE, "check_identical_tables", dev, {}):
            res["identical_tables_bitexact"] = check_identical(m, d, dev)
        print("identical tables bit-exact:", res["identical_tables_bitexact"], flush=True)
        if not res["identical_tables_bitexact"]:
            atomic_write_json(ARMS_JSON, res)
            raise SystemExit("validation 6 failed: identical tables differ from the static model")
    with Run(STAGE, "fp_traj_design", dev, {}):
        tr = fp_traj(dev, DESIGN, True, "design_fp16")
    labels, noise = labels_for_seeds(DESIGN), noise_for_seeds(DESIGN, dev)
    for spec in a.arms.split(","):
        if spec in res["arms"]:
            print(f"  {spec}: already done", flush=True)
            continue
        guard(0.03, f"arm {spec}")
        with Run(STAGE, f"arm_{spec}", dev, {"arm": spec}) as run:
            T, V, store = tables_for(spec)
            st = StepTables(m, T, V)
            torch.cuda.synchronize(); t0 = time.time()
            x0 = ddim_sample(m, d, noise, labels, 1.5).cpu()
            torch.cuda.synchronize(); ts = time.time() - t0
            dg = diagnostics(m, d, dev, tr)
            st.remove()
            store["runtime_buffer_bytes_simulation"] = len(T) * N_W * 4               # dequantized fp32 tables held in the simulation
            r = rel_mse(x0, tr["x0"])
            rec = run.record()
        res["arms"][spec] = {"rel_mse_mean": float(r.mean()), "rel_mse_per_seed": r.tolist(), "sampling_s": ts, "diag": dg, "storage": store,
                             "n_tables": len(T), "gpu_hours": rec["gpu_hours"]}
        if "A" in res["arms"] and spec != "A":
            res["arms"][spec]["vs_A"] = boot_reduction(r.tolist(), res["arms"]["A"]["rel_mse_per_seed"])
        atomic_write_json(ARMS_JSON, res)
        v = res["arms"][spec]
        print(f"  {spec}: latent {v['rel_mse_mean']:.5f}" + (f", vs A {-100 * v['vs_A']['reduction']:+.1f}% [{-100 * v['vs_A']['ci'][1]:+.1f}, "
              f"{-100 * v['vs_A']['ci'][0]:+.1f}]" if "vs_A" in v else "") + f", C/I {dg['C_over_I']:.4f}, C {dg['C']:.4g}, "
              f"{store['bits_per_weight']:.3f} bits/w, {ts:.1f} s", flush=True)


if __name__ == "__main__":
    main()
