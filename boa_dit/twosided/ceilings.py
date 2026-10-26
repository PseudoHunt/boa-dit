"""D1: ceilings by layer type on the uniform W4 model (seeds 0-127). One arm per layer type: that layer's weights in
all 28 blocks are restored to FP (biases are never quantized). Share of base error = 1 - relMSE(arm)/relMSE(base);
base = the stored uniform W4 per-image values (valid after the rule.md §0 integrity check).
Run: python -m boa_dit.twosided.ceilings --arms fc1,fc2,proj --device cuda:0
"""
import argparse
import json
import os

import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.eval_latent_mse import per_image
from boa_dit.model import block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.sample import sample_seeds

OUT = os.path.join(RESULTS, "gate1", "eval")
W4 = os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt")
LAYER = {"fc1": "mlp.fc1", "fc2": "mlp.fc2", "proj": "attn.proj", "qkv": "attn.qkv"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="fc1,fc2,proj")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:128]
    fp = load_dit("cpu")
    for arm in a.arms.split(","):
        out = os.path.join(OUT, f"latent_ceiling_{arm}.json")
        if os.path.exists(out):
            continue
        with Run("gate1", f"d1_ceiling_{arm}", dev, {"arm": arm}) as run:
            m = load_dit("cpu")
            load_quant_state(m, W4)
            for l in range(28):
                block_linear(m.blocks[l], LAYER[arm]).weight.data.copy_(block_linear(fp.blocks[l], LAYER[arm]).weight.data)
            m = m.to(dev)
            for l in (0, 27):   # landed (model moved through CPU only)
                assert torch.equal(block_linear(m.blocks[l], LAYER[arm]).weight.data.cpu(),
                                   block_linear(fp.blocks[l], LAYER[arm]).weight.data)
            x = sample_seeds(m, make_diffusion(50), list(range(128)), 1.5, 32, log=False)
            rel = per_image(x, xf)[0]
            res = {"arm": f"ceiling_{arm}", "layer": LAYER[arm], "seeds": "0-127", "rel_mse_mean": float(rel.mean()),
                   "rel_mse_per_image": rel.tolist(), **run.record()}
            del m
            torch.cuda.empty_cache()
        atomic_write_json(out, res)
        print(arm, res["rel_mse_mean"], flush=True)


if __name__ == "__main__":
    main()
