"""wkf1: matched-shape GEMM microbenchmarks of the installed low-bit ops at DiT-XL/2 linear shapes (results/w8_kernel/PROTOCOL.md step 3, partial
feasibility only: no installed backend implements the exact cheap model (W4 per-row asymmetric x A8 static per-tensor + rank-8 branch) or the exact
H8 anchor (W8 per-row asymmetric x A16 + rank-8 branch) as packed kernels). Ledger tag wkf1. Run: python -m boa_dit.w8_kernel.bench"""
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import h1031b_budget as HB
from boa_dit.common import RESULTS, Run, atomic_write_json

OUT = os.path.join(RESULTS, "w8_kernel")
SHAPES = {"qkv": (1152, 3456), "proj": (1152, 1152), "fc1": (1152, 4608), "fc2": (4608, 1152)}   # per block (x28); adaLN 1152->6912 acts on M = rows only
MS = {"bs1_cfg": 512, "bs32_cfg": 16384}


def tm(fn, warm=10, rep=50):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(rep):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return float(np.median(ts))


def kernels(fn):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        fn(); torch.cuda.synchronize()
    return sorted({e.key for e in p.key_averages() if e.device_type.name == "CUDA"})[:8]


@torch.no_grad()
def main():
    ok, info = HB.ok("wkf1", 0.05)
    print("[wkf1 budget]", info, ok, flush=True)
    if not ok:
        sys.exit(4)
    dev = "cuda:0"
    torch.manual_seed(0)
    res = {"device": torch.cuda.get_device_name(), "cc": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "shapes": {}}
    with Run("wkf1", "bench", dev, {"shapes": SHAPES, "M": MS}) as run:
        for sn, (K, N) in SHAPES.items():
            for mn, M in MS.items():
                r = {}
                x16 = torch.randn(M, K, device=dev, dtype=torch.float16)
                w16 = torch.randn(N, K, device=dev, dtype=torch.float16) * 0.02
                r["fp16_cublas_ms"] = tm(lambda: F.linear(x16, w16))
                xb, wb = x16.bfloat16(), w16.bfloat16()
                r["bf16_cublas_ms"] = tm(lambda: F.linear(xb, wb))
                w8 = torch.randint(-127, 127, (N, K), device=dev, dtype=torch.int8)
                sc = torch.rand(N, device=dev, dtype=torch.float16) * 1e-3
                try:
                    r["w8a16_weight_int8pack_mm_ms"] = tm(lambda: torch._weight_int8pack_mm(x16, w8, sc))
                    if sn == "qkv" and mn == "bs1_cfg":
                        r["w8a16_kernels"] = kernels(lambda: torch._weight_int8pack_mm(x16, w8, sc))
                except Exception as ex:
                    r["w8a16_weight_int8pack_mm_err"] = repr(ex)[:200]
                x8 = torch.randint(-127, 127, (M, K), device=dev, dtype=torch.int8)
                w8t = w8.t().contiguous().t()                                  # [K, N] column-major view expected by cuBLASLt
                try:
                    r["w8a8_int_mm_gemm_only_ms"] = tm(lambda: torch._int_mm(x8, w8.t()))
                    # with per-tensor activation quantization and dequantization (the unavoidable glue for a W8A8 linear)
                    def w8a8():
                        q = torch.clamp(torch.round(x16 * 20.0), -128, 127).to(torch.int8)
                        return torch._int_mm(q, w8.t()).to(torch.float16) * sc
                    r["w8a8_int_mm_with_quant_glue_ms"] = tm(w8a8)
                    if sn == "qkv" and mn == "bs1_cfg":
                        r["w8a8_kernels"] = kernels(lambda: torch._int_mm(x8, w8.t()))
                except Exception as ex:
                    r["w8a8_int_mm_err"] = repr(ex)[:200]
                try:
                    wq = torch.randint(0, 16, (N, K), device=dev, dtype=torch.int32)
                    wu = ((wq[:, ::2] << 4) | wq[:, 1::2]).to(torch.uint8)
                    wp = torch._convert_weight_to_int4pack(wu, 8)
                    gs = 128
                    sz = torch.rand(K // gs, N, 2, device=dev, dtype=torch.bfloat16)
                    r["w4a16_tinygemm_int4pack_mm_ms"] = tm(lambda: torch._weight_int4pack_mm(xb, wp, gs, sz))
                    if sn == "qkv" and mn == "bs1_cfg":
                        r["w4a16_kernels"] = kernels(lambda: torch._weight_int4pack_mm(xb, wp, gs, sz))
                except Exception as ex:
                    r["w4a16_tinygemm_err"] = repr(ex)[:200]
                # rank-8 branch cost (fp16, two thin GEMMs) — present in BOTH the cheap model and H8
                V = torch.randn(K, 8, device=dev, dtype=torch.float16); D = torch.randn(N, 8, device=dev, dtype=torch.float16)
                r["rank8_branch_fp16_ms"] = tm(lambda: (x16 @ V) @ D.t())
                for k_ in list(r):
                    if k_.endswith("_ms") and not k_.startswith("fp16"):
                        r[k_.replace("_ms", "_over_fp16")] = r[k_] / r["fp16_cublas_ms"]
                res["shapes"][f"{sn}|{mn}"] = r
                print(sn, mn, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not k.endswith("kernels")}, flush=True)
        agg = {}
        for mn in MS:
            for op in ("fp16_cublas_ms", "w8a16_weight_int8pack_mm_ms", "w8a8_int_mm_gemm_only_ms", "w8a8_int_mm_with_quant_glue_ms", "w4a16_tinygemm_int4pack_mm_ms", "rank8_branch_fp16_ms"):
                vals = [res["shapes"][f"{sn}|{mn}"].get(op) for sn in SHAPES]
                if all(v is not None for v in vals):
                    agg.setdefault(mn, {})[op] = 28 * sum(vals)                 # per block-linear set x 28 blocks (adaLN excluded)
        res["per_call_block_linears_ms"] = agg
        res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "bench.json"), res)
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_device("cuda:0")
    main()
