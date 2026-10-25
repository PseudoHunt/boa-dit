"""Reviewer items 4 / 6 on PixArt-Σ: functional truncation test of the 127 x 127 cond kernels at steps 0-7.

The mean maps M were not stored in the first run, so the same 16-image sampler is rerun (deterministic). Per (block, head,
step <= 7): M = cond mean map, kappa from M (checked against the stored ckpt/pixart/kappa.pt), and for each K in
{5, 9, 13, 21, 31, 63, 127} the cosine (median over rows) between M and the row-normalized map rebuilt from kappa_K
(kappa inside the centered K x K square, f_K = mean of kappa outside). K_min = smallest K with cosine >= 0.98.
Smoothness: 2-D FFT of (kappa - f_31) on the centered 31 x 31 square, energy fraction with |k_x|, |k_y| <= 31/8.
-> results/attn_share/pixart/truncation.json
"""
import os

import numpy as np
import torch

from boa_dit.attn_share.gate1 import offsets
from boa_dit.attn_share.kernel_truncation import THR, lowfreq_fraction, pct, truncate
from boa_dit.attn_share.pixart.common import (CK, CTX, G, IMAGES, OUT, embeds_for, generators, load_pipe,
                                              patch_self_attention, prompt_embeds)
from boa_dit.common import Run, atomic_write_json

KS = [5, 9, 13, 21, 31, 63, 127]
S_C = 8
KN = 2 * G - 1


@torch.no_grad()
def main(dev="cuda:0"):
    torch.cuda.set_device(dev)
    emb = prompt_embeds(dev)
    stored = torch.load(os.path.join(CK, "kappa.pt"), map_location="cpu")["kappa"]["cond"]
    pipe = load_pipe(dev)
    procs = patch_self_attention(pipe)
    L, H = len(procs), 16
    offs = offsets(G).to(dev)
    flat = offs.flatten()
    cnt = torch.zeros(KN * KN, device=dev).index_add_(0, flat, torch.ones(flat.shape[0], device=dev))
    cos = torch.full((len(KS), S_C, L, H), float("nan"))
    lowf = torch.full((S_C, L, H), float("nan"))
    kdiff = torch.zeros(S_C, L, H)

    def cb(l, s, h, Pc, Pu):
        if s >= S_C:
            return
        M = Pc.mean(0)                                                              # [N, N]
        kap = (torch.zeros(KN * KN, device=dev).index_add_(0, flat, M.flatten()) / cnt).reshape(KN, KN)
        ref = stored[s, l, h].to(dev)
        kdiff[s, l, h] = float((kap - ref).abs().max() / ref.abs().max())
        Mn = M.norm(dim=-1)
        for i, K in enumerate(KS):
            kK = truncate(kap, K)[0]
            P = kK.flatten()[offs]
            P = P / P.sum(-1, keepdim=True)
            cos[i, s, l, h] = ((P * M).sum(-1) / (P.norm(dim=-1) * Mn)).median()
        r31, f31 = truncate(kap, 31)
        c = KN // 2
        lowf[s, l, h] = lowfreq_fraction((kap - f31)[c - 15:c + 16, c - 15:c + 16])

    for p in procs:
        p.mode, p.steps, p.cb = "stats", tuple(range(S_C)), cb
    CTX.step = -1
    with Run("attn_share", "pixart_truncation", dev, {"K": KS}) as run:
        pipe(prompt=None, negative_prompt=None, num_inference_steps=20, guidance_scale=4.5,
             generator=generators(IMAGES, dev), output_type="latent", max_sequence_length=300, **embeds_for(IMAGES, emb, dev))
        rec = run.record()
    ok = cos >= THR
    idx = torch.where(ok.any(0), ok.float().argmax(0), torch.full(ok.shape[1:], -1))
    km = np.where(idx.numpy() < 0, np.inf, np.array(KS)[idx.numpy().clip(0)]).astype(float).flatten()
    rel = torch.tensor(KS)[(cos >= cos[-1] - 0.005).float().argmax(0)].numpy().flatten()
    vals, cnt_ = np.unique(km, return_counts=True)
    res = {"K": KS, "threshold": THR, "steps": f"0-{S_C - 1}", "branch": "cond",
           "kappa_reproduction_max_rel_diff": float(kdiff.max()),
           "K_min_histogram": {("none" if np.isinf(v) else str(int(v))): int(c) for v, c in zip(vals, cnt_)},
           "K_min_median": pct(km, 50), "K_min_p90": pct(km, 90),
           "frac_K_min_le_9": float((km <= 9).mean()), "frac_K_min_le_13": float((km <= 13).mean()),
           "frac_K_min_le_15": float((km <= 15).mean()), "frac_K_min_le_21": float((km <= 21).mean()),
           "frac_full_kernel_below_thr": float(np.isinf(km).mean()),
           "full_kernel_cos_median": float(cos[-1].median()),
           "cos_median_per_K": {str(K): float(cos[i].median()) for i, K in enumerate(KS)},
           "K_min_rel_median": float(np.median(rel)),
           "K_min_rel_histogram": {str(int(v)): int(c) for v, c in zip(*np.unique(rel, return_counts=True))},
           "lowfreq_frac_median": float(lowf.median()), "frac_lowfreq_ge_0.9": float((lowf >= 0.9).float().mean()),
           "C_rule_addendum": {"median_le_13": pct(km, 50) != "none" and pct(km, 50) <= 13,
                               "p90_le_21": pct(km, 90) != "none" and pct(km, 90) <= 21},
           "K_min_per_step_median": [pct(np.where(idx[s].numpy() < 0, np.inf, np.array(KS)[idx[s].numpy().clip(0)]).astype(float).flatten(), 50)
                                     for s in range(S_C)],
           "cos": cos.tolist(), **rec}
    res["C_holds_under_addendum"] = bool(res["C_rule_addendum"]["median_le_13"] and res["C_rule_addendum"]["p90_le_21"])
    atomic_write_json(os.path.join(OUT, "truncation.json"), res)
    print({k: v for k, v in res.items() if k != "cos"})


if __name__ == "__main__":
    main()
