"""V3 as a convolution: AV with a translation-invariant map is a depthwise 2-D conv of V on the token grid.

out_i = sum_j kappa[pos_j - pos_i] v_j / sum_j kappa[pos_j - pos_i]  (row-normalized V3 map)
      = conv2d(V, kappa, pad 15, groups=C) / conv2d(1, kappa, pad 15, groups=H)
(F.conv2d is cross-correlation, so weight[dy + 15, dx + 15] = kappa at offset (dy, dx) = pos_j - pos_i.)

verify : conv path vs dense V3 (kappa[offs] row-normalized, then P @ V) on real kappas at N = 256 (tolerance 1e-3)
bench  : ms per block attention sublayer (projections included) for full attention (fused SDPA), V1 (dense stored map,
         V-only projection) and V3 (conv), at N = 256, 1024, 4096; synthetic inputs, random weights, C = 1152, 16 heads,
         fp16, batch = 16 rows (8 images x 2 CFG branches). Also the core alone (no projections).
-> results/attn_share/convpath.json
"""
import argparse
import os

import torch
import torch.nn.functional as F

from boa_dit.attn_share.compute import C, H, block_flops
from boa_dit.attn_share.gate1 import offsets
from boa_dit.attn_share.kernel_support import kernels
from boa_dit.common import RESULTS, Run, atomic_write_json

HD = C // H
OUT = os.path.join(RESULTS, "attn_share")


def conv_av(v, kap, g):
    """v [B, H, N, hd], kap [H, 31, 31] -> [B, H, N, hd]."""
    B = v.shape[0]
    x = v.permute(0, 1, 3, 2).reshape(B, H * HD, g, g)
    w = kap.to(v.dtype).repeat_interleave(HD, 0)[:, None]                          # [C, 1, 31, 31]
    y = F.conv2d(x, w, padding=15, groups=H * HD)
    norm = F.conv2d(torch.ones(1, H, g, g, device=v.device, dtype=v.dtype), kap.to(v.dtype)[:, None], padding=15,
                    groups=H)                                                       # [1, H, g, g]
    y = y.reshape(B, H, HD, g, g) / norm[:, :, None]
    return y.reshape(B, H, HD, g * g).permute(0, 1, 3, 2)


def dense_from_kernel(kap, g=16):
    offs = offsets(g).to(kap.device)
    P = kap.flatten(1)[:, offs]                                                     # [H, N, N]
    return P / P.sum(-1, keepdim=True)


@torch.no_grad()
def verify(dev):
    M = torch.load(os.path.join(OUT, "maps", "M_cond.pt"), map_location="cpu", mmap=True)[:5]
    kap = kernels(M, 5)[0]                                                          # [L, H, 31, 31], window s0-4
    torch.manual_seed(0)
    worst = 0.0
    for l in (0, 2, 13, 27):
        k = kap[l].to(dev)
        v = torch.randn(4, H, 256, HD, device=dev)
        ref = dense_from_kernel(k) @ v
        out = conv_av(v, k, 16)
        worst = max(worst, float((out - ref).abs().max() / ref.abs().max()))
        v16 = v.half()
        out16 = conv_av(v16, k, 16).float()
        worst16 = float((out16 - ref).abs().max() / ref.abs().max())
    return {"max_rel_fp32": worst, "max_rel_fp16_last_block": worst16, "tolerance": 1e-3,
            "blocks_checked": [0, 2, 13, 27], "kernel_window": "s0-4"}


def timeit(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(torch.tensor(ts).median())


@torch.no_grad()
def bench(dev, B=16):
    torch.cuda.set_device(dev)          # CUDA events and synchronize() act on the current device
    torch.manual_seed(0)
    Wqkv = torch.randn(3 * C, C, device=dev, dtype=torch.half) * C ** -0.5
    Wp = torch.randn(C, C, device=dev, dtype=torch.half) * C ** -0.5
    res = {}
    for N in (256, 1024, 4096):
        g = int(N ** 0.5)
        x = torch.randn(B, N, C, device=dev, dtype=torch.half)
        P = torch.softmax(torch.randn(H, N, N, device=dev), -1).half()             # one stored map, shared by rows
        kap = torch.rand(H, 31, 31, device=dev)

        def full():
            q, k, v = F.linear(x, Wqkv).reshape(B, N, 3, H, HD).permute(2, 0, 3, 1, 4)
            return F.linear(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, C), Wp)

        def vproj():
            return F.linear(x, Wqkv[2 * C:]).reshape(B, N, H, HD)

        def v1_core(v):          # [B, N, H, hd] -> P_h @ V_h for all rows at once: [H, N, N] @ [H, N, B*hd]
            vv = v.permute(2, 1, 0, 3).reshape(H, N, B * HD)
            return (P @ vv).reshape(H, N, B, HD).permute(2, 1, 0, 3).reshape(B, N, C)

        def v3_core(v):
            return conv_av(v.permute(0, 2, 1, 3), kap, g).transpose(1, 2).reshape(B, N, C)

        q, k, v = F.linear(x, Wqkv).reshape(B, N, 3, H, HD).permute(2, 0, 3, 1, 4)
        vb = vproj()
        f = block_flops(N)
        attn_full = f["qkv: Q+K projections"] + f["qkv: V projection"] + f["QK^T"] + f["softmax"] + f["AV (dense map)"] + f["output projection"]
        res[str(N)] = {
            "ms_block_attention_sublayer": {
                "full attention (fused SDPA)": timeit(full),
                "V1 dense stored map": timeit(lambda: F.linear(v1_core(vproj()), Wp)),
                "V3 conv (31x31 depthwise)": timeit(lambda: F.linear(v3_core(vproj()), Wp))},
            "ms_core_only": {
                "SDPA": timeit(lambda: F.scaled_dot_product_attention(q, k, v)),
                "V1 P@V": timeit(lambda: v1_core(vb)),
                "V3 conv": timeit(lambda: v3_core(vb))},
            "gflops_per_row_attention_sublayer": {
                "full attention": attn_full / 1e9,
                "V1 dense stored map": (f["qkv: V projection"] + f["AV (dense map)"] + f["output projection"]) / 1e9,
                "V3 conv (direct)": (f["qkv: V projection"] + 2 * N * 961 * C + f["output projection"]) / 1e9},
            "stored_map_bytes_per_block_branch_step": {"V1": H * N * N * 2, "V3": H * 961 * 2}}
        del P, x, q, k, v, vb
        torch.cuda.empty_cache()
    return {"batch_rows": B, "dtype": "fp16", "notes": "ms are for the whole batch of rows; divide by rows for per-row",
            "per_N": res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    with Run("attn_share", "convpath_verify_bench", a.device, vars(a)) as run:
        res = {"verify": verify(a.device), "bench": bench(a.device)}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "convpath.json"), res)
    print(res["verify"])
    for N, r in res["bench"]["per_N"].items():
        print(N, r["ms_block_attention_sublayer"], r["ms_core_only"])


if __name__ == "__main__":
    main()
