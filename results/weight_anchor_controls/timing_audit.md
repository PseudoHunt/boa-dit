# Timing-unit audit of the wa1 numbers (written 2026-10-30, before any wac1 run)

Source: `boa_dit/weight_anchor/run.py` at 52e727c (`checks()` and `live()`), `results/weight_anchor/checks.json`, `live_dev.json` and `live_conf.json`.

| reported number (wa1) | value | denoiser call or trajectory | one image or batch | before/after CFG duplication | GPU time or wall | loading / decode / diagnostics included? | caveats |
|---|---|---|---|---|---|---|---|
| `forward_latency_batch32_cfg` fp / w4_sim (checks.json) | 384 / 450 ms median of 10 | **one denoiser call**: one forward of the 64-row CFG batch plus the CFG combination; no DDIM update | batch of 32 images | the model sees **64 rows** (32 cond + 32 uncond) | **CUDA events** around each call (stream-elapsed time; launch gaps included), 3 warm-up calls | excluded | timestep fixed at 500; one fixed input |
| `wall_s_w4_run` (checks.json) | 22.7 s | **one complete 50-step trajectory** (plain W4) | batch of 32 | 64-row forwards | host `time.time()` **without** `torch.cuda.synchronize()` before or after | loading excluded; decode excluded | first sampling call after model load (cold: allocator and kernel warm-up included). No end sync, so pending GPU work could be omitted, although the final `.cpu()` comparison happens after the timer. Only used as a budget estimate |
| `wall_s_K1_run_plus_2_w4` (checks.json) | 20.1 s | one trajectory with FP at every call, plus 2 W4 calls | batch of 32 | 64-row | same as above, unsynchronized | excluded | as above |
| `wall_s_per_arm_total` (live_conf.json) | fp 38.7, w4 45.2, anchor_only 43.5, reuse 55.2 s | **complete trajectories, summed over 2 batches** | **64 images = 2 × 32** | 64-row forwards | host wall clock with `torch.cuda.synchronize()` before and after each trajectory | loading, VAE decode and LPIPS **excluded**; only per-call Python counters inside the loop | **one repetition**. The FP arm of batch 1 was the first sampling call in the process (cold), which biases FP *upward*. Both models were resident throughout |
| `wall_s_per_arm_total` (live_dev.json) | fp 19.4, w4 22.6, anchor_only 21.7, reuse 27.6 s | complete trajectories | 32 images (1 batch) | 64-row | synchronized wall | as above | one repetition; FP cold |
| `peak_mem_bytes` (checks.json) | 6.22 GB | — | — | — | `max_memory_allocated` over the whole checks process | both models + one batch-32 trajectory | not per arm |
| `peak_mem_bytes` (live_conf.json) | 11.34 GB | — | — | — | process-wide maximum | **includes VAE decode and LPIPS**, which ran after sampling | not a sampling-memory number |
| `resident_model_bytes` | 2.70 GB each | — | — | — | parameters + buffers | — | the simulated W4 model stores **dequantized fp32** weights, the same bytes as FP; no packed W4 storage exists in this harness |

## Consequences
- **The wa1 claim "reuse is slower than plain FP (55.2 vs 38.7 s per 64 images)"** compares synchronized wall-clock times over 2 batches of 32, one repetition each.
  - The FP number includes the cold start, so it is if anything an over-estimate of FP.
  - The direction is plausible but **not yet a controlled measurement**. wac1 remeasures it as specified below.
- **Not measured anywhere in wa1:**
  - batch-1 latency (it must not be inferred by dividing batch-32 numbers, and is not);
  - per-arm peak memory during sampling only;
  - repeated trajectory timings.
- **The per-call latencies** (384 / 450 ms for a 64-row forward) are per denoiser call at batch 32. They are not trajectory latencies, and they do not cover the DDIM update or the Python loop.
- **No roofline or weight-read-time argument** enters any result. The simulated W4 model has no low-bit kernels, so its speed says nothing about deployed W4.
