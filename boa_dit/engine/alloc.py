"""§3 heig-alloc: per-layer carry ranks. For each layer, E = W - W_q(uniform W4); (lam_k, v_k) eigenpairs of the
bucket-mean FP Gram (Gate A fp_grams), descending; r90 = smallest r with sum_{k<=r} lam_k ||E v_k||^2 >= 90% of the total;
r = min(r90, cap), cap = the integer making model-wide extra bits closest to 0.148 (must be within +-0.01).
Writes configs/heig_alloc_r.json and results/engine/alloc.json.
--greedy (DEVIATION, recorded in REPORT: the pre-registered rule is degenerate here, r90 >= 26 everywhere so the cap binds
in every layer and heig-alloc == heig-8): hand out carry directions one at a time by the largest marginal captured share
(score_k / sum_k score_k of that layer) per bit, rank cap 64, until model-wide extra bits reach 0.148.
Writes configs/heig_allocg_r.json and results/engine/alloc_greedy.json.
Run: python -m boa_dit.engine.alloc [--greedy] --device cuda:0
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, ROOT, atomic_write_json
from boa_dit.deflate.quantize_dc import N_PARAMS
from boa_dit.fp_grams import load_block_grams
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state

TARGET = 0.148
DIMS = {"attn.qkv": (3456, 1152), "attn.proj": (1152, 1152), "mlp.fc1": (4608, 1152), "mlp.fc2": (1152, 4608)}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--greedy", action="store_true")
    a = ap.parse_args()
    dev = a.device
    fp = load_dit("cpu")
    uq = load_dit("cpu")
    load_quant_state(uq, os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt"))
    r90, curves, marg = {}, {}, {}
    for l in range(28):
        G = load_block_grams(l)
        for ln in LINEARS:
            H = G[ln].mean(0).to(dev).double()
            lam, V = torch.linalg.eigh(H)
            lam, V = lam.flip(0).clamp_min(0), V.flip(1)
            E = (block_linear(fp.blocks[l], ln).weight.data - block_linear(uq.blocks[l], ln).weight.data).to(dev).double()
            score = lam * (E @ V).pow(2).sum(0)
            cum = (score.cumsum(0) / score.sum()).cpu().numpy()
            key = f"b{l}.{ln}"
            r90[key] = int(np.searchsorted(cum, 0.9) + 1)
            curves[key] = {"share_at_r": {str(r): float(cum[r - 1]) for r in (1, 2, 4, 8, 16, 32, 64, 128)}}
            marg[key] = (score / score.sum()).cpu().numpy()[:64]
        print(f"block {l}", {ln: r90[f'b{l}.{ln}'] for ln in LINEARS}, flush=True)

    if a.greedy:
        import heapq
        cost = {k: 16 * sum(DIMS[k.split(".", 1)[1]]) / N_PARAMS for k in marg}
        r = {k: 0 for k in marg}
        heap = [(-marg[k][0] / cost[k], k) for k in marg]
        heapq.heapify(heap)
        eb = 0.0
        while heap and eb + min(cost.values()) <= TARGET + 1e-9:
            g, k = heapq.heappop(heap)
            if eb + cost[k] > TARGET + 0.005:
                continue
            r[k] += 1
            eb += cost[k]
            if r[k] < 64:
                heapq.heappush(heap, (-marg[k][r[k]] / cost[k], k))
        vals = np.array(list(r.values()))
        cap_share = {k: float(marg[k][:r[k]].sum()) for k in r}
        res = {"label": "allocg", "extra_bits": eb, "within_tolerance": abs(eb - TARGET) <= 0.01, "r_per_layer": r,
               "captured_share": cap_share,
               "r_distribution": {"min": int(vals.min()), "median": float(np.median(vals)), "max": int(vals.max()),
                                  "n_zero": int((vals == 0).sum()),
                                  "mean_by_type": {ln: float(np.mean([r[f"b{l}.{ln}"] for l in range(28)])) for ln in LINEARS},
                                  "by_block": {str(l): {ln: r[f"b{l}.{ln}"] for ln in LINEARS} for l in range(28)}}}
        atomic_write_json(os.path.join(ROOT, "configs", "heig_allocg_r.json"), {"label": "allocg", "r_per_layer": r, "extra_bits": eb})
        atomic_write_json(os.path.join(RESULTS, "engine", "alloc_greedy.json"), res)
        print(f"greedy: extra bits {eb:.4f}; r distribution {res['r_distribution']['min']}..{res['r_distribution']['max']} "
              f"median {res['r_distribution']['median']}, zeros {res['r_distribution']['n_zero']}; by type {res['r_distribution']['mean_by_type']}", flush=True)
        return

    def bits(cap):
        return sum(16 * min(r, cap) * sum(DIMS[k.split(".", 1)[1]]) for k, r in r90.items()) / N_PARAMS

    caps = list(range(0, max(r90.values()) + 1))
    cap = min(caps, key=lambda c: abs(bits(c) - TARGET))
    r = {k: min(v, cap) for k, v in r90.items()}
    eb = bits(cap)
    vals = np.array(list(r.values()))
    res = {"cap": cap, "extra_bits": eb, "within_tolerance": abs(eb - TARGET) <= 0.01, "r90": r90, "r_per_layer": r,
           "r_distribution": {"min": int(vals.min()), "median": float(np.median(vals)), "max": int(vals.max()),
                              "n_at_cap": int((vals == cap).sum()), "n_zero": int((vals == 0).sum()),
                              "histogram": {str(k): int((vals == k).sum()) for k in sorted(set(vals.tolist()))},
                              "mean_by_type": {ln: float(np.mean([r[f"b{l}.{ln}"] for l in range(28)])) for ln in LINEARS}},
           "energy_curves": curves}
    atomic_write_json(os.path.join(ROOT, "configs", "heig_alloc_r.json"), {"r_per_layer": r, "cap": cap, "extra_bits": eb})
    atomic_write_json(os.path.join(RESULTS, "engine", "alloc.json"), res)
    print(f"cap {cap}: extra bits {eb:.4f}; r distribution {res['r_distribution']}", flush=True)


if __name__ == "__main__":
    main()
