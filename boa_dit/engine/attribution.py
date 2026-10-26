"""§1: error attribution on heig-8. Restore block l (or one layer of it) to FP, carries of the restored layers removed;
rel. latent MSE on seeds 0-63 vs FP. Base = heig-8 rerun on seeds 0-63, checked against the stored per-image values.
Run: python -m boa_dit.engine.attribution --what blocks|layers --device cuda:0
"""
import argparse
import json
import os

import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.carry import attach, carry_path
from boa_dit.eval_latent_mse import per_image
from boa_dit.model import LINEARS, block_linear, load_dit, make_diffusion
from boa_dit.sample import sample_seeds

BASE = "deflate_heig_r8_w4"
OUT = os.path.join(RESULTS, "engine", "attribution")
SEEDS = list(range(64))


@torch.no_grad()
def run_arm(sd, fp, carry, restore, dev, xf, diffusion):
    """restore: set of (block, layer) restored to FP (weight and no carry)."""
    m = load_dit("cpu")
    m.load_state_dict(sd)
    for l, ln in restore:
        block_linear(m.blocks[l], ln).weight.data.copy_(block_linear(fp.blocks[l], ln).weight.data)
    m = m.to(dev)
    for (l, ln) in restore:
        assert torch.equal(block_linear(m.blocks[l], ln).weight.data.cpu(), block_linear(fp.blocks[l], ln).weight.data)
    for key, c in carry.items():
        l, ln = key.split(".", 1)
        if (int(l[1:]), ln) not in restore:
            attach(block_linear(m.blocks[int(l[1:])], ln), c["V"], c["delta"])
    x = sample_seeds(m, diffusion, SEEDS, 1.5, 32, log=False)
    del m
    torch.cuda.empty_cache()
    return per_image(x, xf)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["blocks", "layers"], required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    sd = torch.load(os.path.join(CKPT_DIR, f"ckpt_{BASE}", "model.pt"), map_location="cpu")
    carry = torch.load(carry_path(BASE), map_location="cpu")
    fp = load_dit("cpu")
    xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:len(SEEDS)]
    diffusion = make_diffusion(50)
    if a.what == "blocks":
        arms = {"base": set()} | {f"block{l}": {(l, ln) for ln in LINEARS} for l in range(28)}
    else:
        bl = json.load(open(os.path.join(OUT, "blocks.json")))
        top = sorted(range(28), key=lambda l: -bl["arms"][f"block{l}"]["share"])[:3]
        arms = {f"block{l}.{ln}": {(l, ln)} for l in top for ln in LINEARS}
    out_p = os.path.join(OUT, f"{a.what}.json")
    res = json.load(open(out_p)) if os.path.exists(out_p) else {"arms": {}}
    with Run("engine", f"attribution_{a.what}", dev, vars(a)) as run:
        base_rel = None
        if a.what == "layers":
            base_rel = torch.tensor(json.load(open(os.path.join(OUT, "blocks.json")))["arms"]["base"]["rel_mse_per_image"])
        for name, restore in arms.items():
            if name in res["arms"]:
                continue
            rel = run_arm(sd, fp, carry, restore, dev, xf, diffusion)
            r = {"rel_mse_mean": float(rel.mean()), "rel_mse_per_image": rel.tolist()}
            if name == "base":
                stored = torch.tensor(json.load(open(os.path.join(RESULTS, "gate1", "eval",
                                                                  f"latent_{BASE}.json")))["rel_mse_per_image"][:len(SEEDS)])
                r["integrity_max_abs_diff_vs_stored"] = float((rel - stored).abs().max())
                base_rel = rel
            else:
                r["share"] = float(1 - rel.mean() / base_rel.mean())
            res["arms"][name] = r
            atomic_write_json(out_p, res)
            print(name, r["rel_mse_mean"], r.get("share", r.get("integrity_max_abs_diff_vs_stored")), flush=True)
        res.update(run.record())
    atomic_write_json(out_p, res)


if __name__ == "__main__":
    main()
