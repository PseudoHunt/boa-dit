# wkf1 — real-kernel feasibility for DiT W8-K2 vs W8-only: report

Protocol: `PROTOCOL.md` (pre-registered 2026-10-31 14:06 UTC). **Complete; the track stops at step 2 (no usable packed pair).**
- CPU inventory: 14:10–14:33 UTC, 23 min of the 45-min cap.
- GPU: one partial-feasibility microbenchmark, **0.034 A30-h** (ledger tag wkf1).
- No porting, no kernel builds, no installs, no rental. SANA untouched.

## Answer
**No installed or official prebuilt backend implements either side of the DiT W8-K2 construction as packed kernels on the A30 (sm_80).** So a speed
advantage over W8-only sampling is **hypothetical (a cost ratio only)**, and no full-loop timing was possible.
- The cheap model is W4 per-row asymmetric weights × A8 static per-tensor asymmetric activations, plus a fp16 rank-8 branch and smoothing.
- The H8 anchor is W8 per-row asymmetric weights × fp16 activations, plus its own rank-8 branch.

Criteria, for the record (current 50-step algorithm: 50 cheap + 25 W8 calls with CFG, 24 bridges):
- **Beats W8-only** iff C_Q / C_W8 < 0.5 − (O_hybrid − O_W8only) / (50 C_W8).
  - So the cheap call must cost under half a W8 call, before the extra routing/anchor/bridge overhead.
- **K2-W8 anchors cost no more than K4-FP anchors** (equal cheap-call and overhead costs) iff C_W8/C_FP ≤ 13/25 = 0.52.
  - This is not the W8-only criterion.
  - w8s1 measured 1.17 in fake-quant simulation; that is not a kernel cost.

## Capability table (verified from installed packages and primary source)
| backend (installed version) | operation | sm_80 | packed / tensor-core? | matches the needed op? | source checked |
|---|---|---|---|---|---|
| PyTorch 2.11.0+cu130 `torch._weight_int8pack_mm` | W8 (symmetric per-channel scale) × fp16/bf16 activation | runs | **No.** Naive scalar kernel: one thread per output, fp32 FMA loop over K, no tiling or tensor cores. Profiler: `at::native::weight_int8pack_mm_kernel` | H8 side only in form (no zero point; would need a rank-1 correction plus the separate rank-8 branch). **Not usable:** 87–373× slower than fp16 cuBLAS here | `native_functions.yaml` dispatch; `aten/src/ATen/native/cuda/int8mm.cu` @ v2.11.0 |
| PyTorch `torch._int_mm` | INT8 × INT8 → INT32 GEMM | yes | **Yes** (`cutlass_80_tensorop_i16832gemm_s8_128x64_128x3_tn_align16`) | **No:** W8**A8**. H8 is A16, and the cheap model is W4. Changing anchor activation precision makes a different model and needs fresh persistence/quality tests | dispatch table; profiler |
| PyTorch `torch._weight_int4pack_mm` (tinygemm) | W4 group-wise (bf16 scale + zero) × **bf16** activation | yes | tensor-core m16n8k16, tuned for small M (decode) | **No:** W4**A16** group-wise vs W4A8 static per-row; bf16. Here 3–11× *slower* than fp16 cuBLAS (M = 512 / 16384 rows) | dispatch table; profiler `tinygemm_m16n8k16_chunk_kernel` |
| Nunchaku 1.2.1 (/home/nk_env) `SVDQW4A4Linear` | W4 group-64 × **A4** dynamic, fused rank-r branch + smoothing | yes (INT4) | yes (packed, fused low-rank) | **No:** A4 rather than A8, group-wise W4, and only the Flux/SANA/SDXL/Qwen transformer classes are implemented. DiT-XL/2 would need porting, which is out of scope, plus fresh quality tests | `nunchaku/models/linear.py` docstrings (installed) |
| Nunchaku `AWQW4A16Linear` | W4 group-64 × fp16/bf16 (GEMV) | yes | GEMV kernel | **No:** A16, GEMV-oriented, and not the W8 anchor | same file |
| Triton 3.6.0 | (compiler) | yes | — | Writing a W4A8 or W8A16 kernel = custom kernel development, excluded by the handoff | — |
| torchao, bitsandbytes, vLLM (QQQ/Marlin W4A8), TensorRT, AutoAWQ/GPTQ | — | — | — | **Not installed** in either environment. Installing would change the preserved environments and is outside "start from installed environments"; not verified, not claimed | `pip list` |

## Partial-feasibility microbenchmark (labelled; not a whole-model or full-loop result)
- **Setup:** DiT-XL/2 block-linear shapes (qkv 1152→3456, proj 1152→1152, fc1 1152→4608, fc2 4608→1152).
  - M = 512 rows (batch 1 with CFG) and 16384 (batch 32 with CFG).
  - CUDA events: 10 warmups, 50 repeats, median. Random data. `bench.json`.
- Sum over the 4 linears × 28 blocks (adaLN, attention, norms and non-GEMM work excluded):

| op (per denoiser call, block linears only) | batch 1 + CFG (ms) | ratio to fp16 | batch 32 + CFG (ms) | ratio to fp16 |
|---|---|---|---|---|
| fp16 cuBLAS (baseline) | 10.4 | 1 | 169 | 1 |
| W8A16 `_weight_int8pack_mm` (naive) | 1646 | 158 | 51,817 | 306 |
| W8A8 `_int_mm`, GEMM only | 7.7 | 0.74 | 111 | 0.66 |
| W8A8 `_int_mm` + unfused per-tensor quant/dequant glue | 19.2 | 1.84 | 278 | 1.65 |
| W4A16 tinygemm | 50.4 | 4.85 | 1,539 | 9.1 |
| rank-8 branch, fp16, unfused (present in BOTH models) | 9.0 | 0.86 | 30 | 0.18 |

**Reading:**
- The only packed low-bit GEMM that is faster than fp16 at these shapes is the INT8×INT8 tensor-core GEMM. Its quantize/dequantize glue and an unfused
  rank-8 branch already cost more than the GEMM saves.
- At batch 1 the unfused rank-8 branch alone costs ~0.86× the fp16 GEMMs.
- So neither C_Q nor C_W8 can be measured for the actual construction with installed kernels. C_Q/C_W8 < 0.5 would need a fused W4A8 + low-rank kernel
  against a fused W8A16 + low-rank kernel; neither exists here.

## Memory (both resident models; parameter counts from DiT-XL/2: 675.1 M total = 445.9 M block linears + 225.6 M adaLN + 3.6 M other)
| configuration | packed size (approx.) |
|---|---|
| FP16 model | 1.35 GB |
| Cheap W4 (block linears 4-bit + per-row scale/zero) + adaLN 8-bit + rank-8 branches (4.1 M fp16) + other fp16 | ≈ 0.46 GB |
| H8 anchor (block linears 8-bit) + adaLN 8-bit + its rank-8 branches + other fp16 | ≈ 0.69 GB |
| **W8-K2 hybrid (both resident)** | **≈ 1.15 GB** (vs 0.69 GB for W8-only, 1.35 GB for FP16) |

Memory is not a constraint on a 24 GB A30. Peak benchmark memory was 0.73 GiB.

## Status and what would be needed (not proposed for this task)
- A later efficiency study needs either a backend with fused W4A8-static + low-rank and W8A16 + low-rank kernels for DiT shapes on sm_80, or a
  changed-precision construction (e.g. W8A8 anchors via `_int_mm`).
- The latter is a different algorithm, requiring fresh persistence, parity and quality tests. The DiT W8-K2 fidelity gain does not transfer by
  assumption.
- The 10% measured full-loop margin threshold was never reachable, because no full loop could run on packed kernels.
- No speed or efficiency claim is made.
