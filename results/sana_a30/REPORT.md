# snk1 — real-kernel SANA anchoring feasibility on the A30: report

Protocol: `PROTOCOL.md` (pre-registered; amendments dated). Spend: **0.163 of 1.5 A30-h** (ledger tag snk1). **Last completed stage: development screen
(stage 5); confirmation not opened (development rule failed).** Nothing is running.

## Setup
- **Environment:** `/home/nk_env` reused unchanged (Nunchaku 1.2.1 official wheel 82026d55…, torch 2.11.0+cu130, diffusers 0.36.0, transformers 4.55.2).
  Nothing was installed.
- **GPU:** NVIDIA A30, sm80, driver 580.126.20, 24 GiB, MIG off, 165 W.
- **Models:** the official v1.2.1 SANA example pair.
  - `nunchaku-ai/nunchaku-sana` svdq-int4_r32-sana1.6b (rev 437e41b9…, sha256 320be49f…);
  - `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers` (rev e2b3c0cb…, bf16 variant).
- **Workload:** 1024², batch 1 (CFG batch 2), 20 steps, guidance 4.5, native DPMSolverMultistep (dpmsolver++, order 2, midpoint, flow prediction,
  flow_shift 3).
  - One transformer evaluation per step.
  - No PAG, cache, LoRA, offload or compile.

## Answers
**1. Supported and actually executing packed kernels? Yes.**
- The INT4 call runs 260 `nunchaku::kernels::GEMM_W4A4` launches (41.3 of 59.0 ms CUDA time).
- The 20 transformer blocks are a single Nunchaku `QuantizedSanaModel`. BF16 remains only in the patch-embed conv, the time embedding, the caption
  projection and `proj_out` (6 linears).
- **Caveat found:** the INT4 transformer is **not run-to-run deterministic** on this stack.
  - Identical single calls differ by up to rel L2 0.087, against 0.227 for INT4 vs BF16.
  - It is independent of the padding mask, unpadded tokens and batch size.
  - BF16 is bitwise repeatable. The cause was not debugged (out of scope).

**2. Both denoisers fit? Yes.**
- BF16 transformer (3.21 GB), INT4 transformer, Gemma-2 text encoder (5.23 GB) and DC-AE VAE (0.63 GB) are all resident.
- Peak reserved 12.18 GiB of 23.6 GiB in a hybrid run plus decode, leaving more than the 1 GiB reserve.

**3. Actual speed margin? Yes at both K4 and K8 (timing gate passed; K4 selected before any quality result).**

| measurement | BF16 | INT4 | INT4/BF16 |
|---|---:|---:|---:|
| one call, median of 20 (steps 0 / 10 / 19) | 116.6 / 116.5 / 116.2 ms | 60.1 / 60.4 / 59.9 ms | **0.515** |
| 20-step loop, median of 5 | 2.345 s | 1.186 s | 0.506 |

| hybrid loop (cheap every step + high at anchors) | median | vs H20 | gate (≤ 0.90) |
|---|---:|---:|---|
| K4 (5 high calls) | 1.775 s | **0.757** | pass |
| K8 (3 high calls) | 1.540 s | 0.657 | pass |

- Loaded pipeline (text encoding + loop + VAE decode), median of 3: H20 2.68 s, Q20
  1.52 s, hybrid K4 2.11 s, hybrid K8
  1.88 s.
- Unlike SDXL (nk1, ratio 0.93), SANA's transformer-only denoiser gives real-kernel W4A4 a **≈ 2× per-call margin** on this A30.

**4. Does W4A4 error reuse transfer to SANA? No, on the 16-prompt development screen.**
- Endpoint fidelity to H20 (same prompt and seed). Mean latent rel. MSE: Q20 0.250, **A 0.196**,
  **R 0.218**, S 0.644. LPIPS: Q20 0.321, A 0.280,
  R 0.286, S 0.525.
- Contrasts:
  - **R/A 1.110 [0.951, 1.276], 5/16, ΔLPIPS +0.0061 [-0.0117, +0.0230]** → rule (≤ 0.90) **fails**.
  - R/Q20 0.874 [0.650, 1.129], 9/16, ΔLPIPS -0.0351 [-0.0708, -0.0033].
  - A/Q20 0.787 [0.627, 0.928], 12/16, ΔLPIPS -0.0412 [-0.0677, -0.0207].
  - R/S 0.339 [0.238, 0.483], 14/16, ΔLPIPS -0.2395 [-0.2974, -0.1758].
- Calls per image: A = 5 H + 15 Q (Q omitted at anchors, so R vs A is not matched-cost); R = 5 H + 20 Q; S = 5 H.
- R's warm loop is 1.78 s (0.756 × H20); A's is 1.48 s.
- The confirmation cohort (mjhq1k 464–527) was **not opened**.
- Diagnostics: the guided prediction difference e_i = Q − H at the same H20 states, 16 prompts, pooled.
  - Correlation between e_i and e_(i+lag) at lag 1/2/4/8: 0.24 / 0.16 / 0.08 / 0.03.
  - ‖e_(i+lag) − e_i‖² / ‖e_(i+lag)‖² at lag 1/2/4/8: 1.46 / 1.56 / 1.61 / 1.55.
  - At K4, subtracting the anchor discrepancy **leaves 1.56× the held-step error energy** (> 1 means held subtraction
    adds error locally).
  - **Q's own run-to-run variance is 57% of the raw Q − H error energy.** Much of the discrepancy is call-specific and
    cannot be reused.
  - This is consistent with R not beating A, but it is descriptive, not proof of mechanism.

**5. Does it beat BF16 at matched measured latency in fidelity? Not evaluated.**
- The Hk comparator is defined only in confirmation, which was not reached. No acceleration claim is made.
- For orientation only (not a fidelity comparison): every anchored or cheap arm here is far from H20 in fidelity (LPIPS ≈ 0.28–0.32).

**6. What remains untested:**
- Bridge correction for the multistep solver (see the design note below).
- Generation quality (ImageReward or similar; nothing here measures quality).
- A deterministic INT4 path; whether the nondeterminism is a bug or a design property.
- Other GPUs, batch sizes, Nunchaku versions, Sana 1.5 or distilled models.
- The confirmation cohort and the Hk comparator.

## Bridge-port design note (no implementation)
- Native solver `diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler`, dpmsolver++ order 2, midpoint, lower_order_final,
  final sigma 0.
- The flow prediction v is converted to m_i = x0̂ = x_i − σ_i·v_i (`convert_model_output`).
- The update (`multistep_dpm_solver_second_order_update`) is x_(i+1) = (σ_(i+1)/σ_i)·x_i − α_(i+1)(e^(−h)−1)·[m_i + ½·(m_i − m_(i−1))/r_0], with
  first-order steps at i = 0 and at the final step.
- The state is therefore (x_i, m_(i−1)).
  - A perturbation δv_i enters m_i with gain −σ_i.
  - It affects x_(i+1) through D0 and D1, and x_(i+2) again through the history term.
- The DiT bridge (single-state DDIM, x_(i+1) = A_i·x_i + b_i·ε_i) cannot be copied.
  - A port must linearize the augmented state (x, m_prev) through these coefficients and validate with replay before use.
  - It should also account for Q's ~57% call-specific noise share, which no deterministic correction can remove.
- Code locations: `scripts/sana_a30.py` (`Router` at the transformer interface; `own_loop`, an exact replica of the pipeline loop with full scheduler
  deepcopy save/resume, verified bitwise).
- Cached development outputs: `ckpt/sana_a30/dev_outputs.pt`, endpoint latents and images for H20/Q20/A/R/S. Same-state e_i are summarized in
  dev.json, not saved as tensors.

## Files
- Results: env.json (packages, GPU, hashes, pip freeze), smoke.json, determinism.json, determinism2.json, kernels.json (profiler summary; traces in
  ckpt/sana_a30/trace_*.json), timing.json, checks.json, dev.json, dev_metrics.json (per prompt), budget in `results/gpu_hours.csv`.
- Images, latents and traces are local (`ckpt/sana_a30/`); see artifacts.json for hashes. They are regenerable from RUN.md with the pinned
  revisions. Nothing was uploaded.
