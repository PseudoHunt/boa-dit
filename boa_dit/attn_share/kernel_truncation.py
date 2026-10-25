"""Functional truncation test of the offset kernels (reviewer items 1-2; replaces the 95%-mass side).

For each kernel kappa (n x n, n = 2g - 1) and side K: floor f_K = mean of kappa over the cells outside the centered K x K
square (least-squares floor); kappa_K = kappa inside the square, f_K outside. Rebuild the row-normalized map
kappa_K[pos_j - pos_i] and take its cosine to the mean map M per row, median over rows. K_min = smallest K with
cosine >= 0.98 (None if even the full kernel stays below). Secondary: K_min_rel = smallest K within 0.005 of the
full-kernel cosine.
Smoothness: kappa_res = kappa - f at K = 31 (DiT: 31 is the full kernel, which has no outside cells, so f = f_21 there),
2-D FFT, fraction of energy with |k_x|, |k_y| <= n/8 (the lowest quarter of frequencies on each axis).

DiT-XL/2 (CPU): kernels from results/attn_share/artifacts/V3_kernels_k20.pt, M = the window-mean maps
(artifacts/V2_windowed_maps_k20.pt) they were built from. -> results/attn_share/kernel_truncation.json
"""
import os

import numpy as np
import torch

from boa_dit.attn_share.gate1 import offsets
from boa_dit.common import RESULTS, atomic_write_json

OUT = os.path.join(RESULTS, "attn_share")
THR = 0.98


def truncate(kap, K):
    """kap [..., n, n] -> kappa_K (floor outside the centered K x K square) and the floor."""
    n = kap.shape[-1]
    c = n // 2
    inside = torch.zeros(n, n, dtype=torch.bool, device=kap.device)
    inside[c - K // 2:c + K // 2 + 1, c - K // 2:c + K // 2 + 1] = True
    if inside.all():
        return kap, None
    f = kap[..., ~inside].mean(-1)
    out = torch.where(inside, kap, f[..., None, None])
    return out, f


def map_cos(kK, M, offs):
    """kK [H, n, n], M [H, N, N] -> median over rows of cos(row-normalized kK[offs], M)."""
    P = kK.flatten(-2)[:, offs]
    P = P / P.sum(-1, keepdim=True)
    c = (P * M).sum(-1) / (P.norm(dim=-1) * M.norm(dim=-1))
    return c.median(-1).values


def lowfreq_fraction(r):
    n = r.shape[-1]
    F = torch.fft.fft2(r.double()).abs() ** 2
    k = torch.fft.fftfreq(n, d=1.0 / n).abs()
    low = (k[:, None] <= n / 8) & (k[None, :] <= n / 8)
    return F[..., low].sum(-1) / F.flatten(-2).sum(-1)


def pct(km, q):
    """Lower percentile over K_min with 'none' (never reaches the threshold) sorted above every K."""
    v = np.sort(km)[int(np.ceil(q / 100 * len(km))) - 1]
    return "none" if np.isinf(v) else float(v)


def summarize(kmin, kmin_rel, cos_full, lowf, Ks):
    km = np.array([np.inf if k is None else k for k in kmin.flatten().tolist()], dtype=float)
    kr = kmin_rel.flatten().numpy().astype(float)
    vals, cnt = np.unique(km, return_counts=True)
    return {"K_min_histogram": {("none" if np.isinf(v) else str(int(v))): int(c) for v, c in zip(vals, cnt)},
            "K_min_median": pct(km, 50), "K_min_p90": pct(km, 90),
            "frac_K_min_le_9": float((km <= 9).mean()), "frac_K_min_le_15": float((km <= 15).mean()),
            "frac_K_min_le_21": float((km <= 21).mean()), "frac_full_kernel_below_thr": float(np.isinf(km).mean()),
            "full_kernel_cos_median": float(cos_full.median()),
            "K_min_rel_median": float(np.median(kr)), "K_min_rel_histogram": {str(int(v)): int(c) for v, c in zip(*np.unique(kr, return_counts=True))},
            "lowfreq_frac_median": float(lowf.median()), "frac_lowfreq_ge_0.9": float((lowf >= 0.9).float().mean())}


def dit():
    a = torch.load(os.path.join(OUT, "artifacts", "V3_kernels_k20.pt"), map_location="cpu")
    mw = torch.load(os.path.join(OUT, "artifacts", "V2_windowed_maps_k20.pt"), map_location="cpu")["maps"]
    Ks = [3, 5, 7, 9, 13, 21, 31]
    offs = offsets(16)
    res = {"model": "DiT-XL/2 256 px", "K": Ks, "threshold": THR, "windows": a["windows"], "per_branch": {}}
    for br in ("cond", "uncond"):
        kap = a["kappa"][br]                                                       # [W, L, H, 31, 31]
        W, L, H = kap.shape[:3]
        cos = torch.zeros(len(Ks), W, L, H)
        for w in range(W):
            for l in range(L):
                M = mw[br][w, l].float()
                for i, K in enumerate(Ks):
                    cos[i, w, l] = map_cos(truncate(kap[w, l], K)[0], M, offs)
        ok = cos >= THR
        kmin = np.empty((W, L, H), dtype=object)
        idx = torch.where(ok.any(0), ok.float().argmax(0), torch.full((W, L, H), -1))
        for (w, l, h), i in np.ndenumerate(idx.numpy()):
            kmin[w, l, h] = None if i < 0 else Ks[i]
        rel_ok = cos >= cos[-1] - 0.005
        kmin_rel = torch.tensor(Ks)[rel_ok.float().argmax(0)]
        r, _ = truncate(kap, 21)
        f21 = r[..., 0, 0]                                                          # outside value = f_21
        lowf = lowfreq_fraction(kap - f21[..., None, None])
        d = summarize(kmin, kmin_rel, cos[-1], lowf, Ks)
        d["cos_median_per_K"] = {str(K): float(cos[i].median()) for i, K in enumerate(Ks)}
        d["smoothness_floor"] = "f_21 (K = 31 is the full kernel, no outside cells)"
        d["K_min"] = [[[kmin[w, l, h] for h in range(H)] for l in range(L)] for w in range(W)]
        res["per_branch"][br] = d
        print(br, {k: v for k, v in d.items() if k != "K_min"}, flush=True)
    atomic_write_json(os.path.join(OUT, "kernel_truncation.json"), res)


if __name__ == "__main__":
    torch.set_num_threads(16)
    dit()
