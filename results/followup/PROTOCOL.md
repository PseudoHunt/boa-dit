# Follow-up experiments: sampler-side state correction, spatial shaping probes and tile-balanced rounding, trajectory-aware carry optimization (protocol)

Written 2026-10-28, **before any new result of these experiments was computed**. The same file is committed on both
follow-up branches:
- `followup-weights` (from `weights-joint`, worktree `/home/boa-dit-fw`): Experiments 1 and 3 (DiT-XL/2 256, heig-8 W4).
- `followup-act` (from `temporal-oracles`, worktree `/home/boa-dit-fa`): Experiment 2 (PixArt-Σ 1024², H-SVDQuant W4A4).

Nothing above the marker at the end changes afterwards. Amendments are appended below it, dated, **before** the arms
they affect are run. Earlier results, artifacts and protocols are not modified. Earlier gates rejected specific
constructions; none of them is a proof that sampler correction, observability, spatial shaping or weight correction is
impossible, and this protocol does not treat them as such.

## 0. Budget, data and conventions

### Budget
- **Hard cap: 4.0 A30-hours**, stage `f4`, summed over every checkout's `results/gpu_hours.csv`
  (`boa_dit/f4.py`). This includes calibration, profiling, training and evaluation. Run names carry the prefix
  `e1_`, `e2_` or `e3_`.
- Planning ceilings, transferable but with a fixed total: state correction 0.75 h, spatial shaping 1.0 h, carry
  optimization 2.0 h, diagnostics/reporting reserve 0.25 h.
- Every expensive job is profiled first (pilot timing) and starts only if the guard projects it inside the cap. A job
  that cannot fit is reported **unrun due to cost**, with its estimate. Evaluation sets are never shrunk silently, and
  an incomplete training run is never called a method failure.
- No FID campaign. All images and latents are saved under `ckpt/followup/` for later distributional evaluation.

### Data (from the 2026-10-28 registry sweep of both branches)
- **DiT:** seeds 8000–8999 appear nowhere in either branch and are used here. Seeds 7000–7063 and 7100–7163 (the
  §2 observability traces) were already inspected; they are reused only for fitting and exploratory diagnostics.
- **PixArt:** the `design` prompts (32) are the established design/selection set. mjhq1k[0:256] was used by the W4A4
  base validation (MJHQ-256), and mjhq1k[64:96] also by the 2026-10-28 D2 run. **mjhq1k[256:1000] is untouched**
  and is the confirmation source here.

| Experiment | Fitting / calibration | Design / selection | Final confirmation (untouched) |
| --- | --- | --- | --- |
| 1 (DiT) | traces 7000–7063 (inspected; 4-fold splits by seed) | 1A exploratory held-out: traces 7100–7163 (inspected). 1B placement: seeds 8000–8031 | seeds 8100–8163 (64) |
| 2 (PixArt) | — (rounding has no fitted parameters) | 2A probes: design[0:8]. 2B λ screen: design[8:24] | mjhq1k[256:288] (32) |
| 3 (DiT) | seeds 8200–8263 (training states) | seeds 8300–8331 (lr, λ, normalization) | seeds 8400–8527 (128); C/I on 8400–8431 |

- Each configuration is frozen before confirmation, with no retuning on confirmation results. Bootstraps resample
  seeds or prompts (2,000 resamples, seed 0), never tokens or timesteps. CIs are 95% percentile intervals.

### Shared definitions
- DiT bed: DiT-XL/2 256, heig-8 W4 (`deflate_heig_r8_w4`, rank-8 fp16 carry, high-precision activations), DDIM-50,
  cfg 1.5, fp16 autocast. Latent rel. MSE = ‖x0 − x0_FP‖² / ‖x0_FP‖² per seed against the FP endpoint of the same
  seed. Decoded fidelity: PSNR and LPIPS-VGG of the decoded 256² image against the decoded FP image.
- PixArt bed: H-SVDQuant W4A4 base `hsvd_r4_b0.5_kv16_u2_rs`, DPM-Solver++ 20 steps, guidance 4.5, batch 1, seed =
  prompt index. Metrics are against the bf16 batch-1 references of the same prompts.
- Bands: the 4 equal-area radial DFT bands of `objective/d2.py` (32×32 on DiT latents, 64×64 on the PixArt token
  grid); band 1 is the lowest and includes DC.

## 1. Sampler-side state correction (DiT, heig-8)

**Hypothesis.** A cheap, calibrated transformation of the quantized latent state xq_t, applied once, reduces the
accumulated state error d_t = xq_t − xfp_t (paired arm) or its low-band second-moment mismatch (moment arm). The
target is accumulated state error, not local prediction error (a different target from §2(b)).

### 1A. Cached-trace screen (no new sampling)
- **Data:** the paired free-running FP and heig-8 trajectories (identical noise and labels) stored by §2,
  `ckpt/observability/traces_{cal,eval}.pt`.
- **Placements τ:** the state after 30 denoiser calls (xq at step index 30), after 40 calls (index 40), and the final
  latent x0. Each placement gets its own coefficients.
- **Corrected bands:** bands 1 and 2 (the low half of the spectrum by area). Bands 3–4 pass through unchanged in
  every band-wise arm. Band components come from FFT masking per (seed, channel).
- **Arms** (all statistics from calibration seeds only):
  - **I (identity).**
  - **G (global moment).** One scalar for the whole state, x' = m_FP + g·(x − m_Q). Here m are the per-channel
    calibration means of the spatial-mean (DC) value, and g = sqrt(Var_FP / Var_Q), with variances across seeds per
    element, pooled over channels and positions after removing the per-element calibration mean.
  - **P (paired reconstruction, band-wise affine).** For b ∈ {1, 2}: x'_b = a_b·x_b (+ a per-channel DC bias c in
    band 1). Fitted by least squares to x^FP_b with a ridge penalty toward identity, λ_P·N·(a_b − 1)². λ_P comes from
    {0, 1e-3, 1e-2, 1e-1, 1} by 4-fold CV over calibration seeds.
    - **P-mix:** instead of a scalar, a 4×4 channel-mixing matrix per band, M_b = I + Δ_b, shared across positions
      and modes, with the same ridge toward I and its own CV λ. **Retained over the scalar only if** its CV paired
      MSE is lower than the scalar's in at least 3 of 4 folds and lower on the mean.
  - **M (band-wise moment matching).** For b ∈ {1, 2}: g_b = sqrt(Var_FP,b / Var_Q,b) (across-seed variance per
    element, pooled), plus DC-mean matching in band 1. This matches a second moment; it does not claim to restore the
    distribution.
- **Held-out report (traces 7100–7163, exploratory):**
  - paired state MSE (total and per band, relative to ‖x^FP‖²);
  - band mean mismatch and variance ratio Var_corrected/Var_FP per band;
  - tr Cov(x', d') per band (covariance of the corrected state with its error);
  - coefficients, with their spread over the 4 calibration folds (each fold fitted on 3/4 of the seeds).
  - The moment arm is not rejected because its paired MSE worsens, and corrected variance is not called better
    generation.
- **Retained for 1B:** P (or P-mix if retained) and M, each at all three placements. G is a control and is
  reported in 1A only.
- **Not included:** predicted-clean-sample features. Adding them requires an amendment, and they would have to be
  computed from quantized predictions at the actual quantized state (the existing epsq is on FP states and is not
  deployable).

### 1B. Live intervention
- Each retained correction is applied **once per trajectory**, at one placement. The ordinary heig-8 sampler then
  resumes (for τ = 30 or 40); for the final placement the correction is applied before decoding. Never at every step.
- **Placement selection (design seeds 8000–8031, 32 seeds), then frozen:**
  - P: the τ with the lowest mean endpoint latent rel. MSE.
  - M: the τ with the lowest endpoint band-1+2 moment mismatch, Σ_b∈{1,2} |log(Var_corr,b / Var_FP,b)|.
  - Ties go to the later placement (less downstream change).
- **Confirmation (seeds 8100–8163, 64, untouched):**
  - arms: contemporaneous heig-8 baseline, P at τ_P, M at τ_M; FP references on the same seeds;
  - metrics: latent rel. MSE and decoded PSNR / LPIPS, paired vs baseline; endpoint band mean/variance mismatch with a
    seed-bootstrap CI (whether the targeted mismatch is still improved after the remaining denoising steps);
  - runtime and extra state (the coefficient count);
  - qualitative pairs by a fixed rule: the seeds at the 10th, 50th and 90th percentile of baseline latent error,
    shown as FP / baseline / P / M.
- Generation quality is **unresolved** until a separately budgeted distributional evaluation. All latents and images
  are saved.
- Cost estimate: ≈ 0.15 h (sampling with branch-and-resume from cached prefixes).

## 2. Spatial shaping on PixArt W4A4

**Question.** Are spatially organized activation errors less damaging than equally energetic unstructured errors?
DiT's output-error frequency sensitivity is not assumed to transfer.

### 2A. Sensitivity screen (perturbation probes, not quantization methods)
- **Sites:** attn2.to_q in all 28 blocks (72% of measured output-space activation error), on its smoothed input x̃
  before the linear map, along the unchanged W4A4 base trajectory.
- **Prompts and steps:** design[0:8]; steps 2 (early), 10 (middle), 17 (late) of 20; both CFG rows.
- **Probes:**
  - **Low** (band 1) and **high** (band 4) random fields: Gaussian noise filtered on the 64×64 grid, then scaled per
    (sample, channel) to the RMS of the base A4 rounding error at that site, sample and channel.
  - **White** (all bands), as context.
  - 4 random directions per kind (seeds 0–3).
  - Injected at all 28 sites at one step.
- **Measured:**
  - S_k = ‖Δε̂‖² / ‖ε̂‖², the relative change of the final guided prediction at that step;
  - the ratio S_low / S_high, with a prompt-bootstrap CI;
  - downstream check: for design[0:2] × 2 directions × {low, high} × 3 steps, the resumed trajectory's endpoint latent
    change ‖Δx0‖² / ‖x0‖². Ordering survival = same sign of log(S_low / S_high) at the endpoint as at the prediction.
- These are perturbation probes. Baseline error correlation is context, not a gate.

### 2B. Tile-balanced rounding
- **Where:** the attn2.to_q quantizers only (28 sites). Weight tables, the low-rank branch, the scales and every other
  quantizer are unchanged. No state persists between steps.
- **Rounding rule:**
  - Each value keeps its existing dynamic per-token group scale s (max|x̃|/7 over its 64-channel group), zero point
    (symmetric) and legal codes [−8, 7].
  - The candidates are floor(x̃/s) and ceil(x̃/s), restricted to legal codes; values on a grid point have one
    candidate. No candidate can exceed the group max, so clipping is identical to nearest rounding and is logged
    separately.
  - For each channel and each group of 4 tokens, all ≤ 16 combinations are enumerated, choosing the one that minimizes
    Σ_i e_i² + λ·(Σ_i e_i)².
  - e_i = s_i·code_i − x̃_i is in x̃ units. The physical activation is x = x̃·d_c with one d_c per channel, so the
    objective in physical units is the same up to the common factor d_c² and has the same minimizer. Per-token scales
    s_i differ within a group and are always applied before summing; normalized code errors are never summed.
  - At λ = 0 the rule reproduces nearest rounding, except at exact ties (x̃/s exactly half-integer), where the
    enumeration takes the first minimizing combination while `torch.round` rounds half to even. Ties are counted and
    reported.
- **Arms:** (1) nearest rounding (the base); (2) adjacent 2×2 tiles on the 64×64 token grid; (3) the identical objective
  on fixed random groups of 4 tokens (one random partition, seed 0, the same for every channel, site and step).
- **λ screen:** λ ∈ {0.25, 1, 4} for arm 2 on design[8:24] (16 prompts). The selected λ* has the lowest mean endpoint
  latent rel. MSE vs bf16. Arm 3 uses the same λ* (identical objective). λ* is frozen before confirmation.
- **Measured** (design screen; endpoint metrics also on confirmation):
  - total activation-error energy and band-1 absolute energy at the modified sites; clipping count;
  - linear-output error spectrum (e·Rᵀ, 4 bands);
  - final prediction-error spectrum: ε̂ − ε_bf16 on the arm's own trajectory states, 4 bands, even steps 0–18, on
    design[8:12];
  - endpoint latent rel. MSE, PSNR, LPIPS-VGG (paired vs nearest);
  - runtime per image and peak temporary memory of the rounding kernel.
- **Confirmation:** mjhq1k[256:288] (32, untouched); arms nearest, adjacent λ*, random λ*; bf16 references generated
  for these prompts.
- An increase in actual rounding-error energy is reported, not hidden behind the equal-energy probes. If the method
  shifts error upward spatially but hurts the endpoint, that outcome is reported exactly.
- Cost estimate: 2A ≈ 0.1 h; 2B screen ≈ 0.3 h; confirmation ≈ 0.45 h.

## 3. Trajectory-aware optimization of the existing rank-8 carry (DiT)

**Question.** Can the existing rank-8 carry correct more consequential weight error when optimized through live
sampling dynamics, at the same rank, structure and parameter count?

### Constraints
- W4 codes and scales are frozen. Only the existing carry tensors are trained: V [d_in, 8] and delta [d_out, 8] for
  each of the 112 block linears, as fp32 master copies, exported to fp16 in the original shapes. No timestep tables,
  no predictor network, no second weight model.
- **Implementation check:** with untrained parameters the differentiable path must reproduce heig-8. The
  differentiable 5-step rollout must match `ddim_sample` to ≤ 1e-3 relative error, and the fp16-exported untrained
  carry must give bitwise-identical full sampling.

### Arms
All from the identical heig-8 initialization, the same training seed, Adam, the same lr and batch, and a matched GPU
training time T_train per arm. Update counts and wall time are reported.
- **C0:** original heig-8.
- **C1:** L_local.
- **C2:** L_local + λ·L_roll,full.
- **C3:** L_local + λ·L_roll,low.

### Losses and rollout
- **Training states:** heig-8 free-running trajectories of seeds 8200–8263, cached at all 50 step inputs. Each update
  samples B start states (seed, step t) uniformly.
- **L_local:** ‖ε̂_S(x_t) − ε_FP(x_t)‖² on the guided 4 eps channels at the shared start state, divided by the
  calibration constant n_loc(t) (C0's mean value at step t over the training seeds).
- **L_roll:** teacher (FP weights, no grad) and student (frozen W4 plus trainable carry) both start from the same cached
  quantized state x_t and take K = min(5, 50 − t) DDIM calls. Gradients flow through the whole student rollout,
  including its state updates; the step is a differentiable re-implementation of the DDIM update with no `no_grad` and
  no detach. Gradient checkpointing per denoiser call is used for memory without changing the objective.
  - L_roll,full = ‖x^S_{t+K} − x^T_{t+K}‖² / n_roll(t).
  - L_roll,low uses only the band-1 component of the difference, divided by C0's band-1 value n_roll,low(t).
  - The n are fixed calibration constants (C0 values on the training seeds), so noise level does not change the
    relative weights.
- **Design (seeds 8300–8331), small sweep:**
  - lr ∈ {1e-4, 3e-4}, picked by C1's design-seed local loss after a short run and then used for all arms;
  - λ ∈ {0.5, 2} for C2 and C3 separately, by endpoint latent rel. MSE of full 50-step sampling on the design seeds
    after short runs (T_train / 4).
  - Final runs restart from initialization with the selected values.
- T_train ≈ 0.3 h per arm (set from profiling; the same for C1–C3). If the budget cannot support meaningful matched
  training, the implementation and profile are saved and Experiment 3 is reported **incomplete, not negative**.

### Evaluation (seeds 8400–8527, untouched)
- Full 50-step sampling for C0–C3 plus the FP references; latent rel. MSE and decoded PSNR / LPIPS, paired bootstrap
  vs C0 and C2 vs C1, C3 vs C2.
- Absolute coherent prediction error C = T·‖mean_t e_t‖² and I = mean_t ‖e_t‖², with C/I, where e_t = ε̂_t − ε_t on FP
  trajectory states, seeds 8400–8431.
- Storage: carry bytes (unchanged shapes, fp16), full-rank-equivalent accounting as for heig-8, and runtime.
- **Reads:**
  - only C1 improves over C0: credit to additional carry fitting;
  - C2 beats C1 (CI excludes 0): credit to the trajectory objective;
  - C3 beats C2: the low-band restriction adds value.
- Cost estimate: profiling 0.1 h; design 0.4 h; training 3 × 0.3 h; evaluation 0.15 h; ≈ 1.55 h.

## 4. Report
For each experiment the report gives:
- hypothesis and scope, code commit and data split;
- completed, unrun and incomplete arms;
- paired estimates with 95% CIs;
- calibration/training cost separately from inference cost;
- what is ruled out and what is left untested.

It closes with a table recommending which arms merit larger confirmation. Wording avoids "unobservable" from one
predictor, "nothing to shape" from a white spectrum, "better generation" from moments or latent MSE, and "method
failed" for skipped, undertrained or over-budget arms.

**Run order:** Experiment 1 → 2 → 3, each started only after the c3 queue (DCT-16, D3) releases the GPU.

---

## Amendments
(appended below, dated, before the affected arms run)
