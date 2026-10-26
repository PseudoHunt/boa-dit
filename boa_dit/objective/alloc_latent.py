"""Rank allocations at heig-8's carry bits (rule_addendum_alloc.md), both capped at 32, total 0.148 +- 0.005 bits/weight:
  latent : r_l = min(32, round(k max(c_l, 0))), c_l = band-1 contribution (layer_attrib.json), k by bisection
  energy : greedy by raw marginal e_k = lam_k ||E v_k||^2 per carry parameter (E = uniform-W4 residual, FP Grams)
Writes configs/objective_alloc_{latent,energy}.json ({"label", "r_per_layer"}). Run: python -m boa_dit.objective.alloc_latent
"""
import heapq
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, ROOT, atomic_write_json
from boa_dit.deflate.quantize_dc import N_PARAMS
from boa_dit.fp_grams import load_block_grams
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state

TARGET, TOL, CAP = 0.148, 0.005, 32
DIMS = {"attn.qkv": (3456, 1152), "attn.proj": (1152, 1152), "mlp.fc1": (4608, 1152), "mlp.fc2": (1152, 4608)}


def bits(r):
    return sum(16 * v * sum(DIMS[k.split(".", 1)[1]]) for k, v in r.items()) / N_PARAMS


def latent_alloc():
    c = {k: max(v["band1_contribution"], 0.0) for k, v in json.load(open(os.path.join(RESULTS, "objective", "layer_attrib.json")))["pairs"].items()}
    lo, hi = 0.0, 1e6
    for _ in range(200):
        k = (lo + hi) / 2
        r = {key: int(min(CAP, round(k * v))) for key, v in c.items()}
        if bits(r) > TARGET:
            hi = k
        else:
            lo = k
    r = {key: int(min(CAP, round(lo * v))) for key, v in c.items()}
    return r


@torch.no_grad()
def energy_alloc(dev="cuda:0"):
    fp, uq = load_dit("cpu"), load_dit("cpu")
    load_quant_state(uq, os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt"))
    e = {}
    for l in range(28):
        G = load_block_grams(l)
        for ln in LINEARS:
            H = G[ln].mean(0).to(dev).double()
            lam, V = torch.linalg.eigh(H)
            lam, V = lam.flip(0).clamp_min(0), V.flip(1)
            E = (block_linear(fp.blocks[l], ln).weight.data - block_linear(uq.blocks[l], ln).weight.data).to(dev).double()
            e[f"b{l}.{ln}"] = (lam * (E @ V).pow(2).sum(0))[:CAP].cpu().numpy()
    cost = {k: 16 * sum(DIMS[k.split(".", 1)[1]]) / N_PARAMS for k in e}
    r = {k: 0 for k in e}
    heap = [(-e[k][0] / cost[k], k) for k in e]
    heapq.heapify(heap)
    used = 0.0
    while heap:
        _, k = heapq.heappop(heap)
        if used + cost[k] > TARGET + TOL:
            continue
        r[k] += 1
        used += cost[k]
        if r[k] < CAP:
            heapq.heappush(heap, (-e[k][r[k]] / cost[k], k))
    return r


def summary(r):
    v = np.array(list(r.values()))
    return {"extra_bits": bits(r), "within_tol": abs(bits(r) - TARGET) <= TOL, "min": int(v.min()), "median": float(np.median(v)),
            "max": int(v.max()), "n_zero": int((v == 0).sum()), "n_at_cap": int((v == CAP).sum()),
            "mean_by_type": {ln: float(np.mean([r[f"b{l}.{ln}"] for l in range(28)])) for ln in LINEARS},
            "top_layers": sorted(r, key=lambda k: -r[k])[:12]}


def main():
    out = {}
    for lab, fn in (("latent", latent_alloc), ("energy", energy_alloc)):
        r = fn()
        atomic_write_json(os.path.join(ROOT, "configs", f"objective_alloc_{lab}.json"), {"label": f"alloc{lab}", "r_per_layer": r})
        out[lab] = {**summary(r), "r_per_layer": r}
        print(lab, {k: v for k, v in out[lab].items() if k != "r_per_layer"}, flush=True)
    atomic_write_json(os.path.join(RESULTS, "objective", "alloc_latent.json"), out)


if __name__ == "__main__":
    main()
