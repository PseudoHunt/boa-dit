"""Reviewer item 3: 32-image functional check at k = 10 (seeds 0-31, same layout as the earlier smoke test).
Arms: V1 per-step mean maps, V3 full 31x31 kernel, two-part kernel at K = 9 and K = 15 (floor f_K outside the K x K square,
kappa inside), uniform maps. Rel. latent MSE vs the FP reference latents. -> results/attn_share/truncation_smoke.json
"""
import os

import torch

from boa_dit.attn_share.attn import CTRL, patch
from boa_dit.attn_share.gate1 import offsets
from boa_dit.attn_share.gate2 import load_maps
from boa_dit.attn_share.kernel_truncation import truncate
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample

K_SHARE, SEEDS = 10, list(range(32))


def kernel_maps(K, dev):
    a = torch.load(os.path.join(RESULTS, "attn_share", "artifacts", "V3_kernels_k20.pt"), map_location="cpu")["kappa"]
    offs = offsets(16)
    out = []
    for br in ("cond", "uncond"):
        kap = a[br][:2]                                                           # windows s0-4, s5-9: [2, 28, 16, 31, 31]
        kK = kap if K is None else truncate(kap, K)[0]
        P = kK.flatten(-2)[..., offs]
        out.append((P / P.sum(-1, keepdim=True)).half().to(dev))                  # [2, 28, 16, 256, 256]
    return out


@torch.no_grad()
def main(dev="cuda:0"):
    torch.cuda.set_device(dev)
    model, diffusion = patch(load_dit(dev)), make_diffusion(50)
    xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:32]
    res = {"seeds": "0-31", "k": K_SHARE, "metric": "rel. latent MSE vs FP, mean over 32 images", "arms": {}}
    with Run("attn_share", "truncation_smoke", dev, {}) as run:
        for arm in ("V1_perstep", "V3_full", "two_part_K9", "two_part_K15", "uniform"):
            CTRL.mode, CTRL.k, CTRL.heads = "share", K_SHARE, None
            if arm == "V1_perstep":
                Mc, Mu = load_maps(K_SHARE, dev)
                CTRL.maps = lambda b, s: (Mc[s, b], Mu[s, b])
            elif arm == "uniform":
                U = torch.full((16, 256, 256), 1 / 256, dtype=torch.float16, device=dev)
                CTRL.maps = lambda b, s: (U, U)
            else:
                K = {"V3_full": None, "two_part_K9": 9, "two_part_K15": 15}[arm]
                Mc, Mu = kernel_maps(K, dev)
                CTRL.maps = lambda b, s: (Mc[s // 5, b], Mu[s // 5, b])
            x = ddim_sample(model, diffusion, noise_for_seeds(SEEDS, dev), labels_for_seeds(SEEDS), 1.5).cpu()
            CTRL.mode = None
            rel = ((x - xf) ** 2).flatten(1).sum(1) / (xf ** 2).flatten(1).sum(1)
            res["arms"][arm] = {"rel_mse_mean": float(rel.mean()), "rel_mse_median": float(rel.median())}
            print(arm, res["arms"][arm], flush=True)
            CTRL.maps = None
        res.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "attn_share", "truncation_smoke.json"), res)


if __name__ == "__main__":
    main()
