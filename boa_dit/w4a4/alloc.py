"""X2 rank allocation (rule.md) from the H-SVDQuant r0 run's spectra (results/w4a4/quant/hsvd_r0_b0.5.json, alloc).
90% rule: r_l = min(r90_l, cap), floor 0, cap so total branch params = fixed-r4 total +-2%. If the cap binds in every
layer (degenerate), the pre-registered fallback: hand out directions by the largest raw marginal e_k per parameter
across layers (rank cap 256) to the same total. Writes configs/w4a4_alloc.json and results/w4a4/alloc.json.
"""
import heapq
import json
import os

import numpy as np

from boa_dit.common import RESULTS, ROOT, atomic_write_json
from boa_dit.w4a4.common import LAYERS

DIMS = {"attn1.to_q": (1152, 1152), "attn1.to_k": (1152, 1152), "attn1.to_v": (1152, 1152), "attn1.to_out.0": (1152, 1152),
        "attn2.to_q": (1152, 1152), "attn2.to_k": (1152, 1152), "attn2.to_v": (1152, 1152), "attn2.to_out.0": (1152, 1152),
        "ff.net.0.proj": (1152, 4608), "ff.net.2": (4608, 1152)}


def main():
    src = json.load(open(os.path.join(RESULTS, "w4a4", "quant", "hsvd_r0_b0.5.json")))["layers"]
    keys = sorted(src, key=lambda k: (int(k.split(".", 1)[0]), LAYERS.index(k.split(".", 1)[1])))
    li = {k: int(k.split(".", 1)[0]) * len(LAYERS) + LAYERS.index(k.split(".", 1)[1]) for k in keys}
    cost = {k: sum(DIMS[k.split(".", 1)[1]]) for k in keys}
    target = 4 * sum(cost.values())
    r90 = {k: src[k]["alloc"]["r90"] for k in keys}

    def params(cap):
        return sum(min(r90[k], cap) * cost[k] for k in keys)

    cap = min(range(0, 257), key=lambda c: abs(params(c) - target))
    r = {k: min(r90[k], cap) for k in keys}
    degenerate = all(r[k] == cap for k in keys)
    rule = "r90_cap"
    if degenerate:
        rule = "greedy_marginal_e_per_param"
        e = {k: src[k]["alloc"]["e_k"] for k in keys}
        r = {k: 0 for k in keys}
        heap = [(-e[k][0] / cost[k], k) for k in keys]
        heapq.heapify(heap)
        used = 0
        while heap:
            g, k = heapq.heappop(heap)
            if used + cost[k] > target * 1.02:
                continue
            r[k] += 1
            used += cost[k]
            if r[k] < len(e[k]):
                heapq.heappush(heap, (-e[k][r[k]] / cost[k], k))
    total = sum(r[k] * cost[k] for k in keys)
    vals = np.array([r[k] for k in keys])
    by_type = {ln: {"mean": float(np.mean([r[k] for k in keys if k.endswith(ln)])),
                    "hist": {str(v): int(c) for v, c in zip(*np.unique([r[k] for k in keys if k.endswith(ln)], return_counts=True))}}
               for ln in LAYERS}
    res = {"rule": rule, "cap": cap, "degenerate_r90": degenerate, "target_params": target, "params": total,
           "params_ratio": total / target, "r90_range": [min(r90.values()), max(r90.values())],
           "r_distribution": {"min": int(vals.min()), "median": float(np.median(vals)), "max": int(vals.max()),
                              "n_zero": int((vals == 0).sum())}, "by_type": by_type, "r_per_layer": r}
    atomic_write_json(os.path.join(ROOT, "configs", "w4a4_alloc.json"), {"r": {str(li[k]): r[k] for k in keys}, "rule": rule})
    atomic_write_json(os.path.join(RESULTS, "w4a4", "alloc.json"), res)
    print({k: v for k, v in res.items() if k != "r_per_layer"})


if __name__ == "__main__":
    main()
