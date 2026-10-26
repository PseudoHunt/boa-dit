"""CFG-alloc ranks (rule_addendum_cfg.md): r_l = min(32, round(k ||E v||^2_l)), floor 0, ||E v||^2 from cfg_layer_ratio.json
(design set); k by bisection so total branch params = r4-everywhere over the quantized layers +- 2%.
Writes configs/w4a4_cfgalloc.json ({"r": {layer_index: r}}) and results/w4a4/cfg_alloc.json.
"""
import json
import os

import numpy as np

from boa_dit.common import RESULTS, ROOT, atomic_write_json
from boa_dit.w4a4.common import LAYERS

DIMS = {"attn1.to_q": 2304, "attn1.to_k": 2304, "attn1.to_v": 2304, "attn1.to_out.0": 2304, "attn2.to_q": 2304,
        "attn2.to_out.0": 2304, "ff.net.0.proj": 5760, "ff.net.2": 5760}                     # c_in + c_out


def main():
    lr = json.load(open(os.path.join(RESULTS, "w4a4", "cfg_layer_ratio.json")))
    ev = {k: v["err_v"] for k, v in lr["per_layer"].items()}
    cost = {k: DIMS[k.split(".", 1)[1]] for k in ev}
    target = 4 * sum(cost.values())

    def alloc(kk):
        return {k: int(min(32, round(kk * v))) for k, v in ev.items()}
    lo, hi = 0.0, 1e3 / max(ev.values())
    while sum(alloc(hi)[k] * cost[k] for k in ev) < target:
        hi *= 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(alloc(mid)[k] * cost[k] for k in ev) > target:
            hi = mid
        else:
            lo = mid
    r = alloc(lo)
    params = sum(r[k] * cost[k] for k in ev)
    li = {k: int(k.split(".", 1)[0]) * len(LAYERS) + LAYERS.index(k.split(".", 1)[1]) for k in ev}
    by_type = {}
    for t in DIMS:
        vals = [r[k] for k in ev if k.split(".", 1)[1] == t]
        by_type[t] = {"mean": float(np.mean(vals)), "total_rank": int(sum(vals)),
                      "share_of_branch_params": sum(r[k] * cost[k] for k in ev if k.split(".", 1)[1] == t) / params}
    res = {"prompt_set": lr["prompt_set"], "params": params, "target": target, "ratio": params / target,
           "within_2pct": abs(params / target - 1) <= 0.02, "by_type": by_type,
           "n_zero": sum(1 for v in r.values() if v == 0), "n_at_cap": sum(1 for v in r.values() if v == 32), "r_per_layer": r}
    atomic_write_json(os.path.join(ROOT, "configs", "w4a4_cfgalloc.json"), {"r": {str(li[k]): r[k] for k in r}})
    atomic_write_json(os.path.join(RESULTS, "w4a4", "cfg_alloc.json"), res)
    print({k: v for k, v in res.items() if k != "r_per_layer"})


if __name__ == "__main__":
    main()
