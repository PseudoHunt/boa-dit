"""Stage 2 / 5.3: sequential, order-aware block-by-block weight quantization of DiT-XL/2.

Per block l:  adaLN -> W8 RTN per-channel;  run block over cached inputs (quantized-prefix model) while
accumulating per-(bucket, branch) Grams for qkv/proj/fc1/fc2;  quantize qkv -> proj -> fc1 -> fc2 with BoA's
layer-output (GPTQ) path, each with its weighted Hessian (proj/fc2 per row group);  recompute block outputs
as the next block's inputs;  save the block's quantized weights (resume unit).
"""
import argparse
import os
import time

import numpy as np
import torch

from boa_dit.calib import GramAccumulator, capture_block0_inputs, run_block
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json, assert_no_grad
from boa_dit.model import LINEARS, block_linear, load_dit, make_diffusion
from boa_dit.qutils import boa_gptq_groups, rtn_, rtn_weight
from boa_dit import weights as W

CALIB_SEEDS = list(range(1000, 1032))


def block_state(block):
    return {k: v.detach().cpu().clone() for k, v in block.state_dict().items()}


def run_name(a):
    if a.weighting == "rtn":
        return f"rtn_w{a.bits}"
    n = f"{a.weighting}_w{a.bits}_K{a.K}_lam{a.lam:g}_br{a.branch}"
    if a.gain_tag != "gain_iso_rho0.01" and a.weighting in ("gain", "full"):
        n += f"_{a.gain_tag}"
    if a.gain_stat != "mean" and a.weighting in ("gain", "full"):
        n += f"_g{a.gain_stat}"
    if getattr(a, "tau", 1.0) != 1.0 and a.weighting in ("gain", "full"):
        n += f"_tau{a.tau:g}"
    if getattr(a, "clip_ratio", 0.0) > 0:
        n += f"_clip{a.clip_ratio:g}"
    return n + (a.suffix or "")


@torch.no_grad()
def dmq_lambda(layer_mod, means, layer, B, bits, branch, s):
    """Per-bucket RTN layer-output error  Lam_b = tr(dW M_b dW^T)  from the bucket's mean Gram."""
    Wt = layer_mod.weight.data.float()
    dW = (rtn_weight(Wt, bits) - Wt)
    wc, wu = (s ** 2, (s - 1) ** 2) if branch == "cfg" else (1.0, 1.0)
    wc, wu = wc / (wc + wu), wu / (wc + wu)
    out = []
    for b in range(B):
        M = (wc * means[(layer, b, 0)] + wu * means[(layer, b, 1)]).to(Wt.device)
        out.append(float(((dW @ M) * dW).sum()))
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--weighting", default="uniform", choices=["rtn", "uniform", "gain", "gates", "full", "ptq4dm", "dmq"])
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--branch", default="cfg", choices=["cfg", "equal"])
    ap.add_argument("--gain_tag", default="gain_iso_rho0.01")
    ap.add_argument("--gain_stat", default="mean", choices=["mean", "median"])
    ap.add_argument("--tau", type=float, default=1.0, help="tempered gain: p ∝ (g_t^2)^tau")
    ap.add_argument("--clip_ratio", type=float, default=0.0, help=">0: clip max/min of p across buckets")
    ap.add_argument("--adaln_bits", type=int, default=8)
    ap.add_argument("--groupsize", type=int, default=-1, help="-1 = per-channel (only option implemented)")
    ap.add_argument("--n_calib", type=int, default=32)
    ap.add_argument("--blocks", type=int, default=28, help="quantize only the first n blocks (dev)")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--buckets", type=int, default=10)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--suffix", default="")
    a = ap.parse_args()
    assert a.groupsize == -1, "groupsize not supported by the fork's find_params_H path"
    name = run_name(a)
    dev, B = a.device, a.buckets
    ck = os.path.join(CKPT_DIR, f"ckpt_{name}")
    os.makedirs(ck, exist_ok=True)
    out_json = os.path.join(RESULTS, "stage2", "quant", f"{name}.json")
    model, diffusion = load_dit(dev), make_diffusion(50)
    S, N = diffusion.num_timesteps, a.n_calib
    gates = W.load_gates()
    gain = W.load_gain(a.gain_tag, S, B, a.gain_stat) if a.weighting in ("gain", "full") else None
    log = {"blocks": {}}

    with Run("stage2", f"quant_{name}", dev, vars(a)) as run:
        t_quant = 0.0
        if a.weighting == "rtn":
            t0 = time.time()
            for l in range(a.blocks):
                blk = model.blocks[l]
                rtn_(blk.adaLN_modulation[1], a.adaln_bits)
                for ln in LINEARS:
                    rtn_(block_linear(blk, ln), a.bits)
            t_quant = time.time() - t0
        else:
            t0 = time.time()
            X, C = capture_block0_inputs(model, diffusion, CALIB_SEEDS[:N], a.cfg, dev)
            t_capture = time.time() - t0
            log["capture_s"] = t_capture
            for l in range(a.blocks):
                blk = model.blocks[l]
                bfile = os.path.join(ck, f"block{l}.pt")
                if a.resume and os.path.exists(bfile):
                    blk.load_state_dict(torch.load(bfile, map_location="cpu"))
                    run_block(blk, X, C, S, N, B, write_back=True)    # replay to get next inputs
                    print(f"block {l}: resumed", flush=True)
                    continue
                tb = time.time()
                rtn_(blk.adaLN_modulation[1], a.adaln_bits)
                acc = GramAccumulator(blk, B, dev)
                try:
                    run_block(blk, X, C, S, N, B, acc=acc)
                finally:
                    acc.remove()
                means = acc.means()
                del acc
                blog = {}
                for ln in LINEARS:
                    lin = block_linear(blk, ln)
                    Lam = dmq_lambda(lin, means, ln, B, a.bits, a.branch, a.cfg) if a.weighting == "dmq" else None
                    groups = W.bucket_weights(a.weighting, ln, l, gates, gain, S, B, a.K, a.lam, Lam,
                                              tau=a.tau, clip=a.clip_ratio)
                    glog, solve = [], []
                    for rows, p in groups:
                        if p is None:   # dead channels: round to nearest, skip BoA
                            Wt = lin.weight.data
                            idx = torch.as_tensor(rows, device=Wt.device)
                            Wt[idx] = rtn_weight(Wt[idx], a.bits)
                            glog.append({"n_rows": int(len(rows)), "dead_rtn": True})
                            continue
                        solve.append((rows, W.combine(means, ln, p, B, a.branch, a.cfg)))
                        glog.append({"n_rows": int(len(rows)), "p": [float(x) for x in p]})
                    # one solver call for all row groups (bit-identical to one call per group)
                    boa_gptq_groups(lin, solve, a.bits, replace=1 / 256)
                    del solve
                    blog[ln] = {"groups": glog, **({"dmq_Lambda": Lam} if Lam is not None else {})}
                    if ln in W.ROW_GROUPED and a.weighting in ("gates", "full"):
                        labels, cents = W.row_groups_cached(gates, l, 0 if W.GATE_FOR[ln] == "msa" else 1, a.K)
                        atomic_write_json(os.path.join(RESULTS, "stage2", "rowgroups", f"K{a.K}", f"block{l}_{ln}.json"),
                                          {"labels": labels.tolist(),
                                           "centroids": None if cents is None else cents.tolist()})
                del means
                torch.cuda.empty_cache()
                run_block(blk, X, C, S, N, B, write_back=True)
                atomic_torch_save(block_state(blk), bfile)
                blog["time_s"] = time.time() - tb
                log["blocks"][l] = blog
                print(f"block {l}: {blog['time_s']:.1f}s", flush=True)
            assert_no_grad(X, C)
            t_quant = time.time() - t0
            del X, C
        log["quant_wall_s"] = t_quant
        atomic_torch_save(model.state_dict(), os.path.join(ck, "model.pt"))
        rec = run.record()
    atomic_write_json(out_json, {"name": name, **log, **rec})
    print(f"done {name}: quant {t_quant / 60:.1f} min")


if __name__ == "__main__":
    main()
