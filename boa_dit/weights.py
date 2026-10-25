"""Stage 2 / 5.2: bucket weights p[b, k] and row groups; combines per-bucket Grams into Hessians.

Modes: uniform | gain | gates | full | ptq4dm | dmq  (see handoff 5.2).
Gate energies come from Stage 1 gates.pt (FP, Stage 1 seeds), branch-averaged.
"""
import json
import os

import numpy as np
import torch

from boa_dit.common import RESULTS

DEAD_THR = 1e-4
GATE_FOR = {"attn.qkv": "msa", "attn.proj": "msa", "mlp.fc1": "mlp", "mlp.fc2": "mlp"}
ROW_GROUPED = ("attn.proj", "mlp.fc2")   # output channel j is gated by gate[j]


def load_gain(tag, S=50, B=10, stat="mean"):
    d = json.load(open(os.path.join(RESULTS, "stage1", f"{tag}.json")))
    g = np.array(d["gain2"] if stat == "mean" else d["gain2_median_of_ratios"], dtype=np.float64)
    assert len(g) == S
    return g


def load_gates():
    g = torch.load(os.path.join(RESULTS, "stage1", "gates.pt"))["gates"]   # [28, 2, 2, S, D]
    return g.mean(2).double().numpy()                                       # branch-averaged [28, 2(msa,mlp), S, D]


def bucket_mean(v, B):
    """v [..., S] -> [..., B] mean within consecutive buckets."""
    S = v.shape[-1]
    return v.reshape(*v.shape[:-1], B, S // B).mean(-1)


def row_groups(gate_curves, K, seed=0):
    """gate_curves [S, D] for one block/gate. Returns labels [D] in {0..K-1} or -1 for dead channels."""
    from sklearn.cluster import KMeans
    D = gate_curves.shape[1]
    dead = (gate_curves < DEAD_THR).all(0)
    labels = np.full(D, -1, dtype=np.int64)
    alive = np.where(~dead)[0]
    if K == 1:
        labels[alive] = 0
        return labels, None
    curves = gate_curves[:, alive].T
    curves = curves / np.maximum(curves.mean(1, keepdims=True), 1e-30)
    km = KMeans(n_clusters=K, n_init=10, random_state=seed).fit(curves)
    labels[alive] = km.labels_
    return labels, km.cluster_centers_


def normal_prior(S, B):
    """PTQ4DM-style control: normal density over step index, mean S/2, std S/4 (handoff fallback)."""
    i = np.arange(S, dtype=np.float64)
    w = np.exp(-0.5 * ((i - S / 2) / (S / 4)) ** 2)
    return bucket_mean(w, B)


def dmq_weights(Lam, alpha=20.0):
    Lam = np.asarray(Lam, dtype=np.float64)
    return (1 - Lam / Lam.sum()) ** alpha


def normalize(p):
    p = np.asarray(p, dtype=np.float64)
    return p / p.sum()


_RG_CACHE = {}


def row_groups_cached(gates, block, gi, K, seed=0):
    """row_groups is deterministic (seeded k-means); cache it per (block, gate, K, seed)."""
    key = (block, gi, K, seed)
    if key not in _RG_CACHE:
        _RG_CACHE[key] = row_groups(gates[block, gi], K, seed)
    return _RG_CACHE[key]


def clip_ratio(p, r):
    """Raise the floor so max/min across buckets <= r, renormalize."""
    p = normalize(p)
    return normalize(np.maximum(p, p.max() / r))


def bucket_weights(mode, layer, block, gates, gain, S, B, K, lam=1.0, Lam=None, seed=0, tau=1.0, clip=0.0):
    """Returns (groups, P): groups = list of (row_index_array, p[B]) for rows to quantize with BoA;
    `dead_rows` (RTN) are returned as the group with p=None. For non-grouped layers a single group of all rows.
    tau: tempered gain, p ∝ (g_t^2)^tau per step before bucket averaging. clip > 0: max/min of p across buckets
    clipped to `clip` (applied to the normalized weight before the lambda mix)."""
    gname = GATE_FOR[layer]
    gi = 0 if gname == "msa" else 1
    curves = gates[block, gi]                                   # [S, D]
    gain_b = bucket_mean(gain ** tau, B) if gain is not None else None
    uni = np.full(B, 1.0 / B)

    def mix(p):
        p = normalize(p)
        if clip > 0:
            p = clip_ratio(p, clip)
        return (1 - lam) / B + lam * p

    out = []
    if layer in ROW_GROUPED and mode in ("gates", "full"):
        labels, _ = row_groups_cached(gates, block, gi, K, seed)
        dead = np.where(labels == -1)[0]
        if len(dead):
            out.append((dead, None))
        for k in range(K):
            rows = np.where(labels == k)[0]
            if not len(rows):
                continue
            ge = bucket_mean(curves[:, rows].mean(1), B)        # mean over rows in group and steps in bucket
            p = ge * (gain_b if mode == "full" else 1.0)
            out.append((rows, mix(p)))
        return out
    # scalar-weight layers (qkv, fc1) or K=1 / non-gate modes
    rows_all = np.arange(_out_features(layer))
    if mode == "uniform":
        p = uni
    elif mode == "gain":
        p = mix(gain_b)
    elif mode in ("gates", "full"):              # qkv / fc1: block-mean gate energy as scalar proxy
        ge = bucket_mean(curves.mean(1), B)
        p = mix(ge * (gain_b if mode == "full" else 1.0))
    elif mode == "ptq4dm":
        p = mix(normal_prior(S, B))
    elif mode == "dmq":
        p = mix(dmq_weights(Lam))
    else:
        raise ValueError(mode)
    return [(rows_all, p)]


def _out_features(layer):
    return {"attn.qkv": 3456, "attn.proj": 1152, "mlp.fc1": 4608, "mlp.fc2": 1152}[layer]


def combine(means, layer, p, B, branch="cfg", s=1.5):
    """H = sum_b p_b * M_b with M_b the branch-combined mean Gram of bucket b.
    branch='cfg': weights s^2 (cond) and (s-1)^2 (uncond), normalized; 'equal': plain pooled."""
    wc, wu = (s ** 2, (s - 1) ** 2) if branch == "cfg" else (1.0, 1.0)
    wc, wu = wc / (wc + wu), wu / (wc + wu)
    H = None
    for b in range(B):
        Mb = wc * means[(layer, b, 0)] + wu * means[(layer, b, 1)]
        H = p[b] * Mb if H is None else H + p[b] * Mb
    return H
