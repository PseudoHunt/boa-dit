"""Stage 1 / 4.7 (log-only): attention-pattern stability per grid position.

FP, cond branch only, Stage 1 seeds; blocks 0,13,27; heads 0,7,15; representative steps 2,7,..,47.
P is recomputed from the qkv output inside hooks (fused SDPA is left untouched, so the forward is
unchanged by construction); we check P@V against the SDPA output feeding attn.proj.
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_write_json, atomic_savefig
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.probe_gain import STAGE1_SEEDS
from boa_dit.sample import CTX, ddim_sample

BLOCKS = (0, 13, 27)
HEADS = (0, 7, 15)
REP_STEPS = list(range(2, 50, 5))   # one per bucket
OUT = os.path.join(RESULTS, "stage1")


def far_step(s, S=50):
    return s + 25 if s + 25 < S else s - 25


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    dev = args.device
    model, diffusion = load_dit(dev), make_diffusion(50)
    need = set(REP_STEPS) | {s + 1 for s in REP_STEPS}
    Pn = {}            # (block, head, step) -> row-normalized P, fp16 [M, 256, 256] (in memory only)
    pending = {}
    check = {b: [] for b in BLOCKS}
    hooks = []

    def qkv_hook(bi):
        @torch.autocast("cuda", enabled=False)
        def h(mod, inp, out):
            s, n = CTX.step, CTX.n
            attn = model.blocks[bi].attn
            H, hd = attn.num_heads, attn.head_dim
            qkv = out[:n].float().reshape(n, -1, 3, H, hd).permute(2, 0, 3, 1, 4)  # cond rows only
            q, k, v = qkv[0], qkv[1], qkv[2]
            P = torch.softmax((q @ k.transpose(-1, -2)) * attn.scale, dim=-1)       # [n, H, N, N]
            if s in need:
                for h_ in HEADS:
                    p = P[:, h_]
                    Pn[(bi, h_, s)] = (p / p.norm(dim=-1, keepdim=True)).half()
            if s in (REP_STEPS[0], REP_STEPS[5]):
                pending[bi] = (P @ v).transpose(1, 2).reshape(n, -1, H * hd)
        return h

    def proj_pre_hook(bi):
        @torch.autocast("cuda", enabled=False)
        def h(mod, inp):
            if bi in pending:
                ref = inp[0][:CTX.n].float()
                mine = pending.pop(bi)
                check[bi].append({"step": CTX.step, "max_abs_diff": float((ref - mine).abs().max()),
                                  "max_abs_ref": float(ref.abs().max()),
                                  "rel_fro": float((ref - mine).norm() / ref.norm())})
        return h

    for bi in BLOCKS:
        hooks.append(model.blocks[bi].attn.qkv.register_forward_hook(qkv_hook(bi)))
        hooks.append(model.blocks[bi].attn.proj.register_forward_pre_hook(proj_pre_hook(bi)))
    with Run("stage1", "probe_attn", dev, vars(args)) as run:
        try:
            ddim_sample(model, diffusion, noise_for_seeds(STAGE1_SEEDS, dev), labels_for_seeds(STAGE1_SEEDS), args.cfg)
        finally:
            for h in hooks:
                h.remove()
        res = {"blocks": BLOCKS, "heads": HEADS, "rep_steps": REP_STEPS, "seeds": STAGE1_SEEDS, "branch": "cond",
               "pv_check_vs_sdpa": check, "per": {}}
        for bi in BLOCKS:
            for h_ in HEADS:
                for s in REP_STEPS:
                    A = Pn[(bi, h_, s)].float()                    # [M, N, N]
                    M = A.shape[0]
                    ssum = A.sum(0)                                # [N, N]
                    within = ((ssum * ssum).sum(-1) - M) / (M * (M - 1))   # mean pairwise cos per position
                    adj = (A * Pn[(bi, h_, s + 1)].float()).sum(-1).mean(0)
                    far = (A * Pn[(bi, h_, far_step(s))].float()).sum(-1).mean(0)
                    res["per"][f"b{bi}.h{h_}.s{s}"] = {
                        "block": bi, "head": h_, "step": s, "far_step": far_step(s),
                        "within_median": float(within.median()), "within_p10": float(torch.quantile(within, 0.1)),
                        "adjacent_median": float(adj.median()), "far_median": float(far.median()),
                        "frac_within_gt_0.8": float((within > 0.8).float().mean())}
        # verdict for the gate1.md note
        per_block = {}
        for bi in BLOCKS:
            ok_steps = 0
            for s in REP_STEPS:
                w = np.median([res["per"][f"b{bi}.h{h_}.s{s}"]["within_median"] for h_ in HEADS])
                f = np.median([res["per"][f"b{bi}.h{h_}.s{s}"]["far_median"] for h_ in HEADS])
                ok_steps += int(w >= 0.8 and f <= 0.5)
            per_block[bi] = ok_steps
        res["steps_meeting_criterion_per_block"] = per_block
        res["pooling_plausible"] = sum(v > len(REP_STEPS) / 2 for v in per_block.values()) >= 2
        res["criterion"] = ("per (block, step): head-median of median within >= 0.8 AND head-median of median far "
                            "<= 0.5; plausible if a majority of the 10 steps qualify in >= 2 of 3 blocks")
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "attn_position_stability.json"), res)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(len(BLOCKS), len(HEADS), figsize=(13, 10), sharey=True)
    for r, bi in enumerate(BLOCKS):
        for c, h_ in enumerate(HEADS):
            ax = axs[r, c]
            for key, lab in (("within_median", "within (across images)"), ("adjacent_median", "adjacent step"),
                             ("far_median", "far (±25 steps)")):
                ax.plot(REP_STEPS, [res["per"][f"b{bi}.h{h_}.s{s}"][key] for s in REP_STEPS], marker="o", label=lab)
            ax.axhline(0.8, color="gray", ls=":", lw=1); ax.axhline(0.5, color="gray", ls="--", lw=1)
            ax.set_title(f"block {bi}, head {h_}"); ax.set_ylim(0, 1.02)
            if r == len(BLOCKS) - 1:
                ax.set_xlabel("sampling step")
    axs[0, 0].legend(fontsize=8)
    fig.suptitle("median over positions of attention-row cosine similarity (cond branch)")
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "attn_position_stability.png")); plt.close(fig)
    print({k: v for k, v in res.items() if k not in ("per",)})


if __name__ == "__main__":
    main()
