"""Item 2 (informs Gate A's writeup; gates nothing).

attribution : uniform W4 model, swap one whole block back to FP (64 eval seeds), then per layer inside the top-3
              blocks -> results/stage2/row_attribution.json
guard       : top-100 rows model-wide by RTN-W4 row output error x mean gate energy (and, as control, by RTN error
              alone) kept at W8 RTN on top of the uniform checkpoint; W4 and W3; 512 eval seeds.
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.eval_latent_mse import EVAL_SEEDS, fp_reference, per_image
from boa_dit.fp_grams import load_block_grams
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.qutils import rtn_weight
from boa_dit.sample import sample_seeds
from boa_dit.weights import GATE_FOR, load_gates

N_PARAMS = 445_906_944


def base_name(bits):
    return f"uniform_w{bits}_K1_lam1_brcfg"


def rel_mse(model, diffusion, xf, n):
    xq = sample_seeds(model, diffusion, EVAL_SEEDS[:n], 1.5, 32, log=False)
    return per_image(xq, xf[:n])[0]


@torch.no_grad()
def attribution(dev, n=64):
    out_json = os.path.join(RESULTS, "stage2", "row_attribution.json")
    xf, _ = fp_reference(dev)
    fp = load_dit("cpu").state_dict()
    model, diffusion = load_dit(dev), make_diffusion(50)
    load_quant_state(model, os.path.join(CKPT_DIR, f"ckpt_{base_name(4)}", "model.pt"))
    with Run("item2", "attribution_w4", dev, {"n": n}) as run:
        base = rel_mse(model, diffusion, xf, n)
        res = {"model": base_name(4), "n_seeds": n, "base_rel_mse": float(base.mean()), "blocks": {}, "layers": {}}

        def swapped(prefixes):
            keep = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if k.startswith(prefixes)}
            model.load_state_dict({k: fp[k] for k in keep}, strict=False)
            r = rel_mse(model, diffusion, xf, n)
            model.load_state_dict(keep, strict=False)
            return r

        for l in range(28):
            r = swapped((f"blocks.{l}.",))
            res["blocks"][l] = {"rel_mse": float(r.mean()), "contribution": float(base.mean() - r.mean()),
                                "contribution_share": float((base.mean() - r.mean()) / base.mean())}
            print(f"block {l}: {res['blocks'][l]['contribution_share']:+.3f}", flush=True)
        top = sorted(range(28), key=lambda b: -res["blocks"][b]["contribution"])[:3]
        res["top3_blocks"] = top
        for l in top:
            for ln in LINEARS:
                r = swapped((f"blocks.{l}.{ln}.",))
                res["layers"][f"b{l}.{ln}"] = {"rel_mse": float(r.mean()), "contribution": float(base.mean() - r.mean()),
                                               "contribution_share": float((base.mean() - r.mean()) / base.mean())}
        res.update(run.record())
    atomic_write_json(out_json, res)


@torch.no_grad()
def row_scores():
    """Per-row RTN-W4 output error e_j = dW_j H dW_j^T (static FP Gram) and mean gate energy g_j."""
    W = load_dit("cpu")
    gates = load_gates()                               # [28, 2(msa, mlp), S, D], branch-averaged
    rows = []
    for l in range(28):
        G = load_block_grams(l)
        for ln in LINEARS:
            w = block_linear(W.blocks[l], ln).weight.data.float()
            dW = rtn_weight(w, 4) - w
            H = G[ln].mean(0)
            e = ((dW @ H) * dW).sum(1).numpy()
            gi = 0 if GATE_FOR[ln] == "msa" else 1
            if ln in ("attn.proj", "mlp.fc2"):          # output channel j is gated by gate[j]
                g = gates[l, gi].mean(0)
            else:                                       # qkv / fc1: block-mean gate energy (scalar proxy)
                g = np.full(w.shape[0], gates[l, gi].mean())
            for j in range(w.shape[0]):
                rows.append((l, ln, j, float(e[j]), float(g[j]), w.shape[1]))
        del G
    return rows


@torch.no_grad()
def guard(dev, n_rows=100):
    sel_json = os.path.join(RESULTS, "stage2", "guard_rows.json")
    rows = row_scores()
    e = np.array([r[3] for r in rows]); g = np.array([r[4] for r in rows])
    sets = {"gate": np.argsort(-(e * g))[:n_rows], "err": np.argsort(-e)[:n_rows]}
    sel = {}
    for name, idx in sets.items():
        rs = [rows[i] for i in idx]
        per_layer = {}
        for l, ln, j, *_ in rs:
            per_layer[f"b{l}.{ln}"] = per_layer.get(f"b{l}.{ln}", 0) + 1
        sel[name] = {"rows": [[l, ln, j] for l, ln, j, *_ in rs], "per_layer": per_layer,
                     "n_weights": int(sum(r[5] for r in rs))}
    overlap = len({tuple(r) for r in sel["gate"]["rows"]} & {tuple(r) for r in sel["err"]["rows"]})
    atomic_write_json(sel_json, {"n_rows": n_rows, "overlap_gate_vs_err": overlap, "sets": sel,
                                 "score": "e_j = dW_j H dW_j^T (RTN-W4, static FP Gram); gate set: e_j x mean gate^2"})
    xf, _ = fp_reference(dev)
    W = load_dit("cpu")
    model, diffusion = load_dit(dev), make_diffusion(50)
    for bits in (4, 3):
        for name in ("gate", "err"):
            out_json = os.path.join(RESULTS, "stage2", "eval", f"latent_guard_{name}_w{bits}.json")
            if os.path.exists(out_json):
                continue
            with Run("item2", f"guard_{name}_w{bits}", dev, {"set": name, "bits": bits}) as run:
                load_quant_state(model, os.path.join(CKPT_DIR, f"ckpt_{base_name(bits)}", "model.pt"))
                for l, ln, j in sel[name]["rows"]:
                    w_fp = block_linear(W.blocks[l], ln).weight.data[j:j + 1]
                    block_linear(model.blocks[l], ln).weight.data[j] = rtn_weight(w_fp, 8)[0].to(dev)
                rel = rel_mse(model, diffusion, xf, len(EVAL_SEEDS))
                res = {"set": name, "bits": bits, "base": base_name(bits), "n_rows": n_rows,
                       "extra_bits_per_param": sel[name]["n_weights"] * (8 - bits) / N_PARAMS,
                       "rel_mse_mean": float(rel.mean()), "rel_mse_sem": float(rel.std() / np.sqrt(len(rel))),
                       "rel_mse_per_image": rel.tolist(), **run.record()}
            atomic_write_json(out_json, res)
            print(f"guard {name} W{bits}: {res['rel_mse_mean']:.5f}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["attribution", "guard"])
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    attribution(a.device) if a.what == "attribution" else guard(a.device)
