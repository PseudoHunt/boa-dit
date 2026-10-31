# wkf1 — bounded real-kernel feasibility for DiT W8-K2 vs W8-only — pre-registration

Written 2026-10-31, **before any wkf1 measurement**. Specification: the handoff "audit and replicate anchor-aware calibration" (2026-10-31), task 5.
Branch `w8-kernel-feasibility` (worktree /home/boa-dit-wkf), from `dit-anchor-refit` cd9a052. Tag **wkf1**; planning 0.3 A30-h inside the common
4.0 cap (`scripts/h1031b_budget.py`, shared with bar1). CPU-only investigation of kernel availability: at most 45 minutes. No custom kernel development,
no model porting, no new GPU rental; SANA stays parked (no Nunchaku rebuild / CUDA 13 / kernel changes).

## Question
The current 50-step W8-K2 algorithm (abn1 harness, verified in w8s1) executes per trajectory 50 cheap W4A8-static calls (+ rank-8 branch, smoothing,
A8 tables) and 25 W8 anchor calls (H8: W8 weights, A16 activations), each with CFG (batch 2n), plus 24 bridge corrections.
- T_hybrid = 50 C_Q + 25 C_W8 + O_hybrid; T_W8only = 50 C_W8 + O_W8only.
- Beats W8-only iff C_Q / C_W8 < 0.5 − (O_hybrid − O_W8only) / (50 C_W8).
- Separately, K2-W8 vs K4-FP anchor cost at equal cheap-call and overhead cost requires 25 C_W8 ≤ 13 C_FP, i.e. C_W8/C_FP ≤ 13/25 = 0.52. That is NOT
  the criterion for beating W8-only; both are reported where applicable.
- Count what actually executes; any future change that omits calls or alters the correction is a different algorithm needing parity/quality tests.

## Steps
1. **Inventory (CPU, ≤ 45 min):** installed environments (main py3.10 env, /home/nk_env) and their packages; A30 = sm_80 (INT8 and INT4 tensor-core
   MMA available in hardware). For each candidate backend, verify from primary documentation / source (package source on disk or official repository
   docs) whether it supports, for the *exact* operations needed: (a) W4 weights × A8 static per-tensor asymmetric activations with a rank-8 fp16 branch
   and smoothing (the cheap model); (b) W8 weight × fp16 activation GEMM (H8 anchor); on sm_80, for DiT-XL/2's linear shapes (1152→3456, 1152→1152,
   1152→4608, 4608→1152, adaLN 1152→6912), batch 2×n rows × 256 tokens. A generic INT8 GEMM is not a drop-in diffusion implementation; name a backend
   as "supported" only for what is verified.
2. **Decision:** a *usable packed pair* exists only if both (a) and (b) are available in an installed (or official prebuilt, no build) backend on sm_80
   with packed kernels for these shapes. If not, stop the track with a capability / memory / cost table; no porting or kernel building.
3. **If a pair exists (or for whichever single side exists, as a labelled partial feasibility number):** matched-shape microbenchmarks at the DiT
   token shapes, batch 1 with CFG (M = 512 rows) and batch 32 (M = 16384), CUDA events, 10 warmups, 50 timed repeats, median; dtype-matched fp16
   cuBLAS baseline with comparable optimizations; profiler (torch.profiler) confirms the packed kernel names actually execute; tensor layouts,
   dequantization and peak memory recorded. Microbenchmarks are feasibility estimates only; a whole-model claim requires the full loop (routing,
   scale conversion, anchor measurement, bridge), batch 1, actual CFG, both models resident, synchronization.
4. **Threshold:** a later quality/latency experiment is proposed only if a measured full-loop speed margin of ≥ 10% over W8-only is shown. A raw
   break-even or a microbenchmark ratio alone is not a win.

## Reporting
Capability table (backend × operation × sm_80 × source reference × installed?), memory table (both resident models: W4 packed + branch + tables vs
W8 packed vs fp16), cost table (measured or "hypothetical ratio only"), and the explicit statement of which criterion each number addresses.

---

## Amendments and results
(appended below, dated)
