"""Storage of cycled tables coded as differences from W_q1 (exploratory, user request; CPU only).
Per layer: delta = c_X - c_Wq1 (integer code difference on the shared grid); p = P(delta != 0); fraction of nonzero
differences with |delta| > 1; order-0 empirical entropy H(delta) in bits per weight with one coder table per layer (the
Shannon bound an arithmetic / ANS coder approaches; table overhead negligible). Pooled per layer type weighted by size.
Pair storage = 4 (W_q1) + H(W_q3 - W_q1) bits per weight + carry (fp16 V once, delta per table).
k-cycling: 4 + sum over extra tables of H(table - W_q1) + carry; b' (unclamped mirror pair): 4 + H(W_q2u - W_q1) + carry.
Run: python -m boa_dit.temporal.diffcode
"""
import json
import os

import torch

from boa_dit.common import atomic_write_json
from boa_dit.temporal.common import CK, OUT, PAIR_FILE

N_PARAMS = 445_906_944


def stats(c1, cx):
    d = cx.long() - c1.long()
    n = d.numel()
    _, cnt = torch.unique(d, return_counts=True)
    pr = cnt.double() / n
    H = float(-(pr * torch.log2(pr)).sum())
    nz = int((d != 0).sum())
    return {"n": n, "p_diff": nz / n, "frac_gt1_of_diffs": float((d.abs() > 1).sum()) / max(nz, 1), "H_bits": H,
            "max_abs": int(d.abs().max())}


def main():
    P = torch.load(PAIR_FILE, map_location="cpu")
    xf = os.path.join(CK, "pair_extra.pt")
    X = torch.load(xf, map_location="cpu") if os.path.exists(xf) else None
    tables = {"Wq3": P}
    if X is not None:
        tables.update({"Wq4": X, "Wq5": X, "Wq2u": X})
    res = {"definition": __doc__, "tables": {}}
    for name, src in tables.items():
        per_layer, per_type = {}, {}
        for key, p in P.items():
            s = stats(p["c_Wq1"], src[key][f"c_{name}"])
            per_layer[key] = s
            t = per_type.setdefault(key.split(".", 1)[1], {"n": 0, "nz": 0, "gt1": 0.0, "bits": 0.0})
            t["n"] += s["n"]; t["nz"] += s["p_diff"] * s["n"]; t["gt1"] += s["frac_gt1_of_diffs"] * s["p_diff"] * s["n"]
            t["bits"] += s["H_bits"] * s["n"]
        types = {t: {"p_diff": v["nz"] / v["n"], "frac_gt1_of_diffs": v["gt1"] / max(v["nz"], 1), "H_bits_per_weight": v["bits"] / v["n"],
                     "n_weights": v["n"]} for t, v in per_type.items()}
        tot_n = sum(v["n"] for v in per_type.values())
        assert tot_n == N_PARAMS, tot_n
        res["tables"][name] = {"per_type": types, "overall": {
            "p_diff": sum(v["nz"] for v in per_type.values()) / tot_n,
            "frac_gt1_of_diffs": sum(v["gt1"] for v in per_type.values()) / max(sum(v["nz"] for v in per_type.values()), 1),
            "H_bits_per_weight": sum(v["bits"] for v in per_type.values()) / tot_n}, "per_layer": per_layer}
    v = sum(p["V"].numel() for p in P.values()) * 16 / N_PARAMS
    d = sum(p["d_Wq1"].numel() for p in P.values()) * 16 / N_PARAMS
    H = {n: r["overall"]["H_bits_per_weight"] for n, r in res["tables"].items()}
    res["carry_bits"] = {"V": v, "delta_per_table": d}
    res["storage_bits_per_weight"] = {"k2_pair_Wq1_Wq3": 4 + H["Wq3"] + v + 2 * d}
    if X is not None:
        res["storage_bits_per_weight"]["k3"] = 4 + H["Wq3"] + H["Wq4"] + v + 3 * d
        res["storage_bits_per_weight"]["k4"] = 4 + H["Wq3"] + H["Wq4"] + H["Wq5"] + v + 4 * d
        res["storage_bits_per_weight"]["bu"] = 4 + H["Wq2u"] + v + 2 * d
    atomic_write_json(os.path.join(OUT, "diffcode.json"), res)
    for n, r in res["tables"].items():
        print(n, "overall", {k: round(x, 4) for k, x in r["overall"].items()})
        for t, s in r["per_type"].items():
            print("   ", t, {k: round(x, 4) for k, x in s.items() if k != "n_weights"})
    print("storage", {k: round(x, 3) for k, x in res["storage_bits_per_weight"].items()}, "carry", {k: round(x, 4) for k, x in res["carry_bits"].items()})


if __name__ == "__main__":
    main()
