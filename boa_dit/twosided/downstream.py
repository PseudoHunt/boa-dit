"""Block-2 downstream test: how much of an arm's block-2 output error comes from quantizing W_o and W2.

For each arm: block-2 inputs from the arm's own quantized prefix (blocks 0-1, carries active), held-out calibration
seeds 1032-1047; target = FP block 2 with the arm's W8 adaLN (oracle.py's protocol, fp16 autocast forward).
  (a) qkv and fc1 as quantized in the arm (with their carries), attn.proj (W_o) and fc2 (W2) at FP (no carry)
  (b) all four linears as in the arm (with carries)
Error = cfg-weighted mean of ||y - y_FP||^2 per token (and relative to ||y_FP - x||^2).
--all_blocks: the same (a)/(b) errors for every block l = 0..27, each on the arm's own prefix (inputs propagated
through the arm's blocks with carries), for the arms in PER_BLOCK -> downstream_per_block.json.
Run: python -m boa_dit.twosided.downstream [--all_blocks] --device cuda:0
"""
import argparse
import copy
import os

import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.carry import attach, carry_path
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.twosided.oracle import L_BLK, TEST, block_err, block_inputs, row_weights

ARMS = {"E4": "ts_uniform_vv1_fc1v1_w4", "E5": "ts_uniform_vv2_fc1v1_wov2_w4", "H4": "ts_heig8_vv1_fc1v1_w4",
        "H5gs": "ts_heig8_vv2_fc1v1_wov2_gs_w4", "uniform": "uniform_w4_K1_lam1_brcfg", "heig-8": "deflate_heig_r8_w4",
        "H4dsf": "ts_heig8_vv1_fc1v1_dsf_w4"}
DOWNSTREAM = ("attn.proj", "mlp.fc2")
PER_BLOCK = ("uniform", "heig-8", "E4", "H4", "H4dsf")


def block_cases(fp_block, arm_block, carry, l, X, C, w):
    """(a) W_o, W2 at FP; (b) as in the arm. Target: FP block with the arm's W8 adaLN."""
    fpb = copy.deepcopy(fp_block)
    fpb.adaLN_modulation[1].weight.data.copy_(arm_block.adaLN_modulation[1].weight.data)
    res = {}
    for case, fp_layers in (("a_WoW2_fp", DOWNSTREAM), ("b_as_arm", ())):
        blk = copy.deepcopy(fpb)
        for ln in LINEARS:
            if ln in fp_layers:
                continue
            lin = block_linear(blk, ln)
            lin.weight.data = block_linear(arm_block, ln).weight.data.clone()
            c = carry.get(f"b{l}.{ln}")
            if c is not None:
                attach(lin, c["V"], c["delta"])
        res[case] = block_err(blk, fpb, X, C, w)
        del blk
    res["a_over_b"] = res["a_WoW2_fp"]["mse"] / res["b_as_arm"]["mse"]
    return res


@torch.no_grad()
def per_block(dev, labels=None, out_path=None):
    from boa_dit.calib import capture_block0_inputs
    from boa_dit.deflate.carry import install
    from boa_dit.twosided.quantize_ts import run_block_cur
    diffusion = make_diffusion(50)
    S = diffusion.num_timesteps
    w = row_weights(S, len(TEST))
    out = {}
    with Run("gate1", "downstream_per_block", dev, {}) as run:
        fp = load_dit(dev)
        for lab in labels or PER_BLOCK:
            name = ARMS.get(lab, lab)
            if not os.path.exists(os.path.join(CKPT_DIR, f"ckpt_{name}", "model.pt")):
                out[lab] = "missing"
                continue
            m = load_dit(dev)
            load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{name}", "model.pt"))
            carry = torch.load(carry_path(name), map_location="cpu") if os.path.exists(carry_path(name)) else {}
            install(m, carry)
            X, C = capture_block0_inputs(fp, diffusion, TEST, 1.5, dev)
            rows = []
            for l in range(28):
                r = block_cases(fp.blocks[l], m.blocks[l], carry, l, X, C, w)
                rows.append(r)
                run_block_cur(m.blocks[l], X, C, S, len(TEST), 10, write_back=True)
            out[lab] = {"arm": name, "blocks": rows}
            print(lab, [round(r["b_as_arm"]["mse"], 3) for r in rows], flush=True)
            del m, X, C
            torch.cuda.empty_cache()
        out.update(run.record())
    atomic_write_json(out_path or os.path.join(RESULTS, "gate1", "downstream_per_block.json"), out)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--all_blocks", action="store_true")
    ap.add_argument("--arms", default=None, help="comma list of arm names (or labels) for --all_blocks")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    if a.all_blocks:
        return per_block(dev, a.arms.split(",") if a.arms else None, a.out)
    diffusion = make_diffusion(50)
    S = diffusion.num_timesteps
    out = {}
    with Run("gate1", f"downstream_b{L_BLK}", dev, vars(a)) as run:
        fp = load_dit(dev)
        w = row_weights(S, len(TEST))
        for lab, name in ARMS.items():
            if not os.path.exists(os.path.join(CKPT_DIR, f"ckpt_{name}", "model.pt")):
                out[lab] = "missing"
                continue
            m = load_dit(dev)
            load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{name}", "model.pt"))
            carry = torch.load(carry_path(name), map_location="cpu") if os.path.exists(carry_path(name)) else {}
            for key, c in carry.items():                                         # prefix blocks get their carries
                l, ln = key.split(".", 1)
                if int(l[1:]) < L_BLK:
                    attach(block_linear(m.blocks[int(l[1:])], ln), c["V"], c["delta"])
            X, C = block_inputs(m, fp, TEST, dev, diffusion)
            fpb = copy.deepcopy(fp.blocks[L_BLK])
            fpb.adaLN_modulation[1].weight.data.copy_(m.blocks[L_BLK].adaLN_modulation[1].weight.data)
            res = {}
            for case, fp_layers in (("a_WoW2_fp", DOWNSTREAM), ("b_as_arm", ())):
                blk = copy.deepcopy(fpb)
                for ln in LINEARS:
                    if ln in fp_layers:
                        continue
                    lin = block_linear(blk, ln)
                    lin.weight.data = block_linear(m.blocks[L_BLK], ln).weight.data.clone()
                    c = carry.get(f"b{L_BLK}.{ln}")
                    if c is not None:
                        attach(lin, c["V"], c["delta"])
                res[case] = block_err(blk, fpb, X, C, w)
                del blk
            res["a_over_b"] = res["a_WoW2_fp"]["mse"] / res["b_as_arm"]["mse"]
            out[lab] = {"arm": name, **res}
            print(lab, {k: (v if isinstance(v, float) else v["mse"]) for k, v in res.items()}, flush=True)
            del m, X, C, fpb
            torch.cuda.empty_cache()
        out.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "gate1", f"downstream_b{L_BLK}.json"), out)


if __name__ == "__main__":
    main()
