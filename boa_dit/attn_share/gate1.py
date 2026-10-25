"""Attention sharing, Gate 1: are early-step attention maps a property of (block, head, step, position)?

FP, cfg 1.5, DDIM-50. 32 images: noise seeds 1000-1031, image m gets class CLASSES[m // 2] (16 classes drawn with
np.random.default_rng(0), so images 2c and 2c+1 share a class and differ in noise). Statistics for steps 0-24, all
28 blocks x 16 heads, computed on the fly from P rebuilt in fp32 (the forward itself stays on the fused path).
Also saves the cond/uncond mean maps M [25, 28, 16, 256, 256] fp16 to results/attn_share/maps/ (outside git) for
Gate 2. Writes results/attn_share/gate1_stats.json; verdict and plots come from scripts/make_attn_gate1.py.
"""
import argparse
import math
import os

import numpy as np
import torch

from boa_dit.attn_share.attn import CTRL, patch
from boa_dit.common import RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample

SEEDS = list(range(1000, 1032))
CLASSES = sorted(np.random.default_rng(0).choice(1000, 16, replace=False).tolist())
LABELS = [CLASSES[m // 2] for m in range(32)]
S_STATS = 25
OUT = os.path.join(RESULTS, "attn_share")
MAPS = os.path.join(OUT, "maps")
EX_BLOCKS, EX_STEPS, EX_IMAGES, EX_POS = (0, 13, 27), (2, 10, 20), (0, 2), (0, 136)
STATS = ["within_median", "within_p10", "within_sameclass", "within_diffclass", "within_uncond_median",
         "ent_median", "ent_meanmap_median", "kl_ratio_median", "kl_ratio_loo_median", "branch_cos_median",
         "kernel_cos_median"]


def offsets(g=16):
    yy, xx = torch.meshgrid(torch.arange(g), torch.arange(g), indexing="ij")
    y, x = yy.flatten(), xx.flatten()
    return ((y[None, :] - y[:, None] + g - 1) * (2 * g - 1) + (x[None, :] - x[:, None] + g - 1))   # [N, N]


def row_cos(a, b):
    return (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1))


def pair_within(P):
    """P [M, H, N, N] -> mean pairwise cosine per (h, i) for all / same-class / different-class pairs."""
    M = P.shape[0]
    A = P / P.norm(dim=-1, keepdim=True)
    G = torch.einsum("mhid,nhid->himn", A, A)                                   # [H, N, M, M]
    eye = torch.eye(M, dtype=torch.bool, device=P.device)
    cls = torch.tensor([m // 2 for m in range(M)], device=P.device)
    same = (cls[:, None] == cls[None, :]) & ~eye
    diff = cls[:, None] != cls[None, :]
    red = lambda msk: (G * msk).sum((-1, -2)) / msk.sum()
    return red(~eye), red(same), red(diff)


def kernel_cos(Mc, offs):
    H, N, _ = Mc.shape
    flat = offs.flatten()
    cnt = torch.zeros(offs.max() + 1, device=Mc.device).index_add_(0, flat, torch.ones_like(flat, dtype=Mc.dtype))
    kap = torch.zeros(H, offs.max() + 1, device=Mc.device).index_add_(1, flat, Mc.reshape(H, -1)) / cnt
    R = kap[:, offs]
    R = R / R.sum(-1, keepdim=True)
    return row_cos(R, Mc)                                                          # [H, N]


@torch.no_grad()
def parity(model, diffusion, dev):
    """(1) explicit-softmax path vs fused SDPA, all block outputs; (2) share path fed one image's own maps."""
    res = {}
    x = noise_for_seeds(SEEDS[:2], dev)
    y = torch.tensor(LABELS[:2] + [1000, 1000], device=dev)
    t = torch.full((4,), 999, device=dev)
    outs = {}
    hooks = [blk.register_forward_hook(lambda m, i, o, b=b: outs.__setitem__(b, o.float().clone()))
             for b, blk in enumerate(model.blocks)]
    CTX.step, CTX.n = 0, 2

    def run(mode):
        CTRL.mode = mode
        with torch.autocast("cuda", dtype=torch.float16):
            model(torch.cat([x, x]), t, y)
        CTRL.mode = None
        return {b: v.clone() for b, v in outs.items()}

    ref = run(None)
    exp = run("explicit")
    d = [float((exp[b] - ref[b]).abs().max() / ref[b].abs().max()) for b in range(28)]
    res["explicit_vs_fused_max_rel"] = max(d)
    res["explicit_vs_fused_per_block"] = d
    # share path: single image (n = 1), maps = that image's own explicit P at this step -> must match explicit
    own = {}
    CTRL.cb = lambda b, s, pc, pu: own.__setitem__(b, (pc[0].half(), pu[0].half()))
    x1, y1 = x[:1], torch.tensor([LABELS[0], 1000], device=dev)
    CTX.n = 1
    CTRL.mode = "stats"
    with torch.autocast("cuda", dtype=torch.float16):
        model(torch.cat([x1, x1]), t[:2], y1)
    CTRL.mode, CTRL.k, CTRL.maps, CTRL.heads = "share", 1, (lambda b, s: own[b]), None
    with torch.autocast("cuda", dtype=torch.float16):
        model(torch.cat([x1, x1]), t[:2], y1)
    sh = {b: v.clone() for b, v in outs.items()}
    CTRL.mode = "explicit"
    with torch.autocast("cuda", dtype=torch.float16):
        model(torch.cat([x1, x1]), t[:2], y1)
    ex1 = {b: v.clone() for b, v in outs.items()}
    CTRL.mode = None
    d = [float((sh[b] - ex1[b]).abs().max() / ex1[b].abs().max()) for b in range(28)]
    res["share_own_maps_vs_explicit_max_rel"] = max(d)
    for h in hooks:
        h.remove()
    CTX.step = None
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    os.makedirs(MAPS, exist_ok=True)
    model, diffusion = patch(load_dit(dev)), make_diffusion(50)
    offs = offsets().to(dev)
    L, H, N = 28, 16, 256
    st = {k: torch.zeros(S_STATS, L, H) for k in STATS}
    maps = {br: torch.zeros(S_STATS, L, H, N, N, dtype=torch.float16) for br in ("cond", "uncond")}
    ex = {}
    logN = math.log(N)

    def cb(b, s, Pc, Pu):
        if s >= S_STATS:
            return
        Mn = Pc.shape[0]
        Mc, Mu = Pc.mean(0), Pu.mean(0)
        maps["cond"][s, b] = Mc.half().cpu()
        maps["uncond"][s, b] = Mu.half().cpu()
        w, ws, wd = pair_within(Pc)
        st["within_median"][s, b] = w.median(-1).values.cpu()
        st["within_p10"][s, b] = torch.quantile(w, 0.1, dim=-1).cpu()
        st["within_sameclass"][s, b] = ws.median(-1).values.cpu()
        st["within_diffclass"][s, b] = wd.median(-1).values.cpu()
        st["within_uncond_median"][s, b] = pair_within(Pu)[0].median(-1).values.cpu()
        logP = Pc.clamp_min(1e-30).log()
        ent = -(Pc * logP).sum(-1)                                                    # [M, H, N] nats
        st["ent_median"][s, b] = (ent / logN).permute(1, 0, 2).reshape(H, -1).median(-1).values.cpu()
        st["ent_meanmap_median"][s, b] = (-(Mc * Mc.clamp_min(1e-30).log()).sum(-1) / logN).median(-1).values.cpu()
        kl_u = logN - ent
        kl_m = (Pc * (logP - Mc.clamp_min(1e-30).log())).sum(-1)
        loo = (Mn * Mc - Pc) / (Mn - 1)
        kl_l = (Pc * (logP - loo.clamp_min(1e-30).log())).sum(-1)
        st["kl_ratio_median"][s, b] = (kl_u.mean(0) / kl_m.mean(0)).median(-1).values.cpu()
        st["kl_ratio_loo_median"][s, b] = (kl_u.mean(0) / kl_l.mean(0)).median(-1).values.cpu()
        st["branch_cos_median"][s, b] = row_cos(Mc, Mu).median(-1).values.cpu()
        st["kernel_cos_median"][s, b] = kernel_cos(Mc, offs).median(-1).values.cpu()
        if b in EX_BLOCKS and s in EX_STEPS:
            ex[f"b{b}.s{s}"] = Pc[list(EX_IMAGES)][:, :, list(EX_POS)].half().cpu()  # [2 img, H, 2 pos, N]

    with Run("attn_share", "gate1", dev, vars(a)) as run:
        par = parity(model, diffusion, dev)
        print("parity", par, flush=True)
        CTRL.mode, CTRL.cb = "stats", cb
        try:
            ddim_sample(model, diffusion, noise_for_seeds(SEEDS, dev), LABELS, 1.5)
        finally:
            CTRL.mode = None
        # step-to-step stability of the cond mean map
        adj = torch.full((S_STATS, L, H), float("nan"))
        far = torch.full((S_STATS, L, H), float("nan"))
        for b in range(L):
            Mb = maps["cond"][:, b].to(dev).float()                                     # [S, H, N, N]
            adj[:-1, b] = row_cos(Mb[:-1], Mb[1:]).median(-1).values.cpu()
            far[:-5, b] = row_cos(Mb[:-5], Mb[5:]).median(-1).values.cpu()
        st["step_cos_s1_median"], st["step_cos_s5_median"] = adj, far
        rec = run.record()
    for br in ("cond", "uncond"):
        atomic_torch_save(maps[br], os.path.join(MAPS, f"M_{br}.pt"))
    atomic_torch_save(ex, os.path.join(OUT, "gate1_examples.pt"))
    atomic_write_json(os.path.join(OUT, "gate1_stats.json"), {
        "seeds": SEEDS, "classes": CLASSES, "labels": LABELS, "steps": list(range(S_STATS)), "branch": "cond unless named",
        "shape": "[step][block][head]", "parity": par,
        "notes": {"within": "mean pairwise cosine of row i across the 32 images, then median / p10 over i",
                  "ent": "row entropy / log 256; ent_median = median over (image, position)",
                  "kl_ratio": "per position mean_m KL(P||U) / mean_m KL(P||M), then median over positions; "
                              "M includes the image itself (as specified); _loo uses the leave-one-out mean",
                  "kernel": "31x31 offset kernel averaged from M, row-normalized, cosine to M per row",
                  "step_cos": "cosine of M[s] vs M[s+1] (s1) and M[s+5] (s5) per row, median; NaN where undefined"},
        "stats": {k: v.tolist() for k, v in st.items()}, **rec})
    print("done", flush=True)


if __name__ == "__main__":
    main()
