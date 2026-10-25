"""Attention sharing, Section 4: FLOPs accounting (analytic) and measured forward latency (--measure, idle GPU).

FLOPs = 2 x multiply-accumulates for matmuls; softmax counted as 5 ops per score (max, sub, exp, sum, div).
Per image, per CFG branch, one forward. DiT-XL/2: C = 1152, 16 heads, MLP ratio 4, 28 blocks, patch 2, 4 latent
channels in / 8 out. N = 256, 1024, 4096 tokens (256 px, 512 px, 1024 px or a short video at the same patch size).
"""
import argparse
import json
import os
import time

import torch

from boa_dit.common import RESULTS, Run, atomic_write_json

C, H, R, DEPTH, P, CIN, COUT = 1152, 16, 4, 28, 2, 4, 8
NS = (256, 1024, 4096)
OUT = os.path.join(RESULTS, "attn_share")


def block_flops(N):
    return {
        "adaLN (per image, not per token)": 2 * C * 6 * C,
        "qkv: Q+K projections": 2 * N * C * 2 * C,
        "qkv: V projection": 2 * N * C * C,
        "QK^T": 2 * N * N * C,
        "softmax": 5 * H * N * N,
        "AV (dense map)": 2 * N * N * C,
        "output projection": 2 * N * C * C,
        "fc1": 2 * N * C * R * C,
        "fc2": 2 * N * R * C * C,
    }


def other_flops(N):
    # patch embed (conv as matmul) + final layer (adaLN 2C + linear) + t/y embedders (MLP 256->C->C, negligible)
    return 2 * N * (P * P * CIN) * C + 2 * C * 2 * C + 2 * N * C * (P * P * COUT) + 2 * (256 * C + C * C)


def removed(N, variant):
    f = block_flops(N)
    rem = f["qkv: Q+K projections"] + f["QK^T"] + f["softmax"]
    av = f["AV (dense map)"]
    if variant == "V3 offset kernel (direct 31x31 conv)":
        rem += av - 2 * N * 961 * C            # AV replaced by a direct depthwise 31x31 conv per head
    elif variant == "V4 sparse top-32":
        rem += av - 2 * N * 32 * C
    return rem


VARIANTS = ["V1/V2 dense stored map", "V4 sparse top-32", "V3 offset kernel (direct 31x31 conv)"]


def analytic(k_values=(10, 15)):
    res = {"assumptions": __doc__.strip(), "per_N": {}}
    for N in NS:
        f = block_flops(N)
        blk = sum(f.values())
        model = DEPTH * blk + other_flops(N)
        e = {"block_flops": f, "block_total": blk, "model_total_per_branch": model, "variants": {}}
        for v in VARIANTS:
            r = removed(N, v)
            e["variants"][v] = {"block_frac_removed": r / blk,
                                "end_to_end_frac_saved": {str(k): (k / 50) * DEPTH * r / model for k in k_values}}
        res["per_N"][str(N)] = e
    return res


@torch.no_grad()
def measure(dev, iters=30):
    torch.cuda.set_device(dev)          # CUDA events and synchronize() act on the current device
    from boa_dit.attn_share.attn import CTRL, patch
    from boa_dit.model import load_dit
    from boa_dit.sample import CTX
    model = patch(load_dit(dev))
    n = 32
    x = torch.randn(2 * n, 4, 32, 32, device=dev)
    t = torch.full((2 * n,), 999, device=dev)
    y = torch.cat([torch.randint(0, 1000, (n,), device=dev), torch.full((n,), 1000, device=dev)])
    U = torch.full((H, 256, 256), 1 / 256, dtype=torch.float16, device=dev)
    CTX.step, CTX.n = 0, n
    out = {}
    for name, mode in (("baseline (fused SDPA)", None), ("explicit softmax (no sharing)", "explicit"),
                       ("shared map, V-only projection", "share")):
        CTRL.mode, CTRL.k, CTRL.maps, CTRL.heads = mode, 50, (lambda b, s: (U, U)), None
        for _ in range(5):
            with torch.autocast("cuda", dtype=torch.float16):
                model(x, t, y)
        torch.cuda.synchronize(dev)
        ts = []
        for _ in range(iters):
            s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
            s.record()
            with torch.autocast("cuda", dtype=torch.float16):
                model(x, t, y)
            e.record(); torch.cuda.synchronize(dev)
            ts.append(s.elapsed_time(e))
        out[name] = {"median_ms": float(torch.tensor(ts).median()), "min_ms": float(min(ts))}
    CTRL.mode = None
    CTX.step = None
    return {"batch": "32 images x 2 CFG branches = 64 rows, N = 256, fp16 autocast, fp32 weights", "iters": iters,
            "timings": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--k", default="10,15")
    a = ap.parse_args()
    res = analytic(tuple(int(k) for k in a.k.split(",")))
    path = os.path.join(OUT, "compute.json")
    if a.measure:
        with Run("attn_share", "latency", a.device, vars(a)) as run:
            res["measured"] = measure(a.device)
            res["measured"].update(run.record())
    elif os.path.exists(path):
        old = json.load(open(path))
        if "measured" in old:
            res["measured"] = old["measured"]
    atomic_write_json(path, res)
    print(json.dumps({N: {v: e["variants"][v] for v in VARIANTS} for N, e in res["per_N"].items()}, indent=1))


if __name__ == "__main__":
    main()
