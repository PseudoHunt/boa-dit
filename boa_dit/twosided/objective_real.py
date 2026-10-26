"""Solver objective on real factors (rule.md, Solver test 4, real-layer version): block 2's FP weights; H_col = the FP
Gram of the layer (Gate A FP Grams, bucket mean); H_row from the D2 pass (ckpt/gate1_factors_b2.pt). Reports the
two-sided objective tr(dW H_col dW^T H_row) for one-sided GPTQ and the row-batched solver at several n_rows, plus
wall-clock. Run: python -m boa_dit.twosided.objective_real --device cuda:0
"""
import argparse
import os

import torch

from boa_dit.common import CKPT_DIR, RESULTS, atomic_write_json
from boa_dit.fp_grams import load_block_grams
from boa_dit.model import load_dit
from boa_dit.twosided.solver import kron_objective, solve_rows


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    F = torch.load(os.path.join(CKPT_DIR, "gate1_factors_b2.pt"), map_location="cpu")
    G = load_block_grams(2)
    blk = load_dit("cpu").blocks[2]
    cases = {
        "fc1 (H_row fc1)": (blk.mlp.fc1.weight.data, G["mlp.fc1"].mean(0), F["H_row_fc1"].unsqueeze(0), (64, 16)),
        "attn.proj (H_row W_o)": (blk.attn.proj.weight.data, G["attn.proj"].mean(0), F["H_row_Wo"].unsqueeze(0), (64, 16)),
        "V (H_row V1, attn Gram)": (blk.attn.qkv.weight.data[2304:], F["G_attn"], F["H_row_V1"], (8, 1)),
        "V (H_row V2, attn Gram)": (blk.attn.qkv.weight.data[2304:], F["G_attn"], F["H_row_V2"], (8, 1)),
    }
    out = {}
    for lab, (Wt, Hc, Hr, nrs) in cases.items():
        Wt, Hc, Hr = Wt.to(dev).float(), Hc.to(dev).float(), Hr.to(dev).float()
        h = Hr.shape[0]
        Hc3 = Hc if Hc.dim() == 3 else Hc.unsqueeze(0)
        r = {}
        Q, s = solve_rows(Wt, Hc, None, bits=4) if Hc.dim() == 2 else solve_rows(Wt, Hc, None, bits=4)
        r["one-sided GPTQ"] = {"objective": kron_objective((Wt - Q).view(h, -1, Wt.shape[1]), Hc3, Hr), "s": s}
        for n in nrs:
            Q, s = solve_rows(Wt, Hc, Hr, bits=4, n_rows=n)
            r[f"two-sided n_rows={n}"] = {"objective": kron_objective((Wt - Q).view(h, -1, Wt.shape[1]), Hc3, Hr), "s": s}
        base = r["one-sided GPTQ"]["objective"]
        for v in r.values():
            v["vs_gptq"] = v["objective"] / base
        out[lab] = r
        print(lab, {k: (round(v["vs_gptq"], 4), round(v["s"], 1)) for k, v in r.items()}, flush=True)
    atomic_write_json(os.path.join(RESULTS, "gate1", "objective_real_b2.json"), out)


if __name__ == "__main__":
    main()
