# Temporal feedback extension: W4A4 transfer, compressed state, runtime (pre-registration)

Written 2026-10-27, before any extension run. This is a separate, capped extension that starts after the current queue
(stage 3 arm 4b, FID-2k for arms 1 and 5, PixArt's four stage-3 arms) finishes. Results go to
`results/temporal/ext/`. Everything here is exploratory unless marked "confirmation". Nothing above the marker changes
afterwards; deviations are appended below it, dated.

## Ceiling and runner
- **Hard ceiling: C_ext A30-hours,** set by the user before the first extension job, counted from the logged
  stage-`temporal` total at that moment.
- **Runner rules:**
  - pilot-time the first batch of every job and project the job's cost;
  - skip any job that cannot finish under the ceiling;
  - save results incrementally;
  - after every job, commit and push with `results/temporal/STATUS.md` updated.
- **No A30-hour estimates** are attached to E1 or E2 until pilot timings exist.

## Order
1. E1 screen.
2. E2 screen.
3. E3 runtime.
4. Then **either** a confirmation run (E1 or E2) **or** the step-count / site diagnostic (E4 / E5), chosen from the
   results by the user.
- No phase-schedule expansion, and no feedback-gain tuning.

## E1: feedback on the strong W4A4 baseline (PixArt-Σ)
- **Base:** the corrected H-SVDQuant r4 arm, `w4a4.quantize --method hsvd --r 4 --kv16 --fc2_unsigned --range_search`
  (`hsvd_r4_b0.5_kv16_u2_rs`), rebuilt on this machine unchanged.
  - Weights W4 (groups of 64) plus a 16-bit low-rank branch.
  - Activations: dynamic per-token symmetric INT4 in groups of 64 on X̃ = X/d; post-GELU fc2 input shifted unsigned.
  - Cross-attention K/V at 16-bit.
  - Same sampler (DPM-Solver++ 20, guidance 4.5), prompts and seeds; batch 1 for every arm.
- **Arms:**
  - (a) base;
  - (b) base + full feedback;
  - (c) base + best stateless construction.
- **(b) feedback** on the base's own activation quantizer, in the X̃ space, per sample and per CFG row, reset at step 0:
  u = x̃ + s, q = Q_A4(u) (dynamic grid from u, as the base computes it from its input), q̂ = bf16(q),
  r⁰ = u − q̂, s = fp16(clamp(r⁰, ±Δ_g/2)), with Δ_g the group's dynamic step. A dynamic max-based grid does not clip,
  so no clip guard is needed. The low-rank branch keeps seeing the unquantized x̃, as in the base.
- **(c) stateless:** the stateless family that was best on DiT stage 3 (uniform subtractive dither). The offset is
  o ~ U(−Δ_g/2, Δ_g/2) per group, element and step, with fixed seed 0; then q = Q(x̃ + o) − o on the grid computed from
  x̃. Any clipping the offset adds at a group's max is counted.
- **Screen:** MJHQ `mjhq1k[:64]`, labelled exploratory.
  - Reported: latent rel. MSE vs bf16 (batch-1 references), PSNR / LPIPS-VGG vs bf16 (the reproduced baseline's
    quality metrics), prediction-error energy and C/I on the 20-step design cache, and clip energy.
- **"Promising"** means (b) beats (a) on latent MSE with the paired CI excluding 0. Then (a) and (b) are confirmed on
  the disjoint `mjhq1k[64:320]` (256 cases). Published numbers are context only.
- **Conditional diagnostic** (only if (b) vs (a) is flat, i.e. the CI includes 0): (a) and (b) again with FP weights
  and the same activation quantizer (W16A4 dynamic, no branch), on the same 64.

## E2: compressed feedback state (DiT-XL/2, stage-3 configuration)
- **Fixed configuration:** single-table W16A8 static, smoothed fc1/fc2, guarded feedback, sampler and protocol as in
  stage 3.
- **State:** at each site, per channel and per CFG row, a k-dimensional spatial state r ∈ R^(k × C) on the 16 × 16 token
  grid, with an orthonormal token basis U ∈ R^(256 × k). Per step:
  1. fed residual = U·r, guarded as in stage 3;
  2. ũ, q, q̂ as in stage 3;
  3. r ← Uᵀ·clamp(ũ − q̂, ±Δ/2), stored fp16.
- **Arms:**
  - no feedback;
  - full feedback (U = I);
  - **spatial mean**, k = 1 (U = 1/16);
  - **2 × 2 low-frequency DCT,** k = 4: orthonormal 2-D DCT-II modes (u, v) ∈ {0, 1}², including DC;
  - **4 random orthonormal token modes** (control): the Q of a QR of a 256 × 4 Gaussian matrix, generator seed 0, the
    same for every site.
  Applied independently to each channel and CFG row.
- **Screen:** untouched seeds 3000–3063 (64), exploratory.
- **Confirmation:** the best compressed arm and the two references (none, full) on untouched seeds 3100–3611 (512).
- **Reported:**
  - endpoint latent rel. MSE with paired CIs;
  - gain retained R = (E_none − E_comp)/(E_none − E_full), with a bootstrap CI and the underlying errors (R is not
    optimized alone);
  - prediction C/I;
  - persistent state M = 2·Σ_ℓ numel(r_ℓ) bytes, computed from the allocated tensors (both CFG rows);
  - peak device memory, separately.
- More modes only if k = 4 suggests a useful trade-off (by user decision).

## E3: runtime with diagnostics disabled (DiT)
- **Arms:** baseline, uniform dither, full feedback, and the best compressed arm from E2.
- **Method:** `log=False` (no counters, no host transfers); identical batches (32 seeds, 50 steps); one warm-up batch;
  then 5 timed repeats with `torch.cuda.synchronize()` around the workload.
- **Reported:**
  - denoiser latency per step (CUDA events around each transformer forward, mean over steps and repeats);
  - end-to-end generation latency per batch;
  - persistent residual memory (allocated tensors);
  - peak device memory (`torch.cuda.max_memory_allocated`, reset per arm).
  - Unquantized sampling is timed as the floor.

## E4 (conditional): step count within PixArt
Run only if PixArt's stage-3 or E1 feedback gain is flat or markedly weaker than DiT's.
- **Arms:** single-table W16A8 baseline and feedback at 10 and 50 DPM-Solver++ steps, plus the 20-step stage-3 result
  where compatible.
- **Data:** 16 paired MJHQ cases (`mjhq1k[:16]`).
- **Reported:** the relative improvement at each step count, with actual denoiser evaluations counted.
- The single table is step-count-independent; it is the one calibrated on the 20-step design trajectory.

## E5 (if time remains): which sites need feedback (DiT)
- **Arms:** feedback at all sites; mlp.fc2 inputs only; all MLP inputs; attention inputs only.
- **Seeds:** the E2 screen seeds; the selected configuration is validated on the E2 confirmation seeds.

---

## Deviations and results
(appended below as they happen)
- 2026-10-27 (user: "keep running experiments, don't ask me"): **C_ext = 8 A30-h**, counted from the logged total when
  queue B ends (written to `ext_ceiling.txt` at that moment). The post-screen choice (confirmation vs. E4/E5) is made by
  the assistant from the screen results, using this rule fixed now, before any extension result:
  - if E1 feedback beats the base with CI > 0 → E1 confirmation;
  - else E4 (step count) if PixArt stage-3 feedback is flat;
  - then E2 confirmation of the best compressed arm, if its R CI excludes 0;
  - then E5.
  Each choice is logged here with its trigger.
- 2026-10-27 ~21:45 UTC, E1 screen result: on the W4A4 base, feedback is **worse**: latent +22.8% [+16.0, +31.1],
  ΔPSNR −0.96 dB, ΔLPIPS +0.111. Prediction C/I halves (0.064 → 0.033), and the feedback clip rate is 11.8%. Signed
  dynamic groups cannot clip, so the clipping is on fc2's unsigned input path (zero bound at the GELU lobe minimum),
  ≈ 1/3 of fc2 inputs per step. That clip error is step-constant and not fed back. The pre-registered conditional
  diagnostic (W16A4) is defined for a *flat* result and does not fire. Decision rule: E1 not promising → no E1
  confirmation.
- **Added post hoc** (decided after seeing the E1 result; exploratory, not pre-registered): E1 feedback with the
  stage-3 clipping guard on the unsigned fc2 path, r_fed = max(s, −shift − x̃_c) with x̃_c = max(x̃, −shift), so u never
  crosses the zero bound. Discarded energy logged; same 64 prompts. It runs after the queued extension jobs.
- 2026-10-27 ~22:00 UTC, **correction:** the E1 "clip rate 11.8%" was a counter artifact. The unsigned path compared
  `(u + shift).clamp_min(0) − shift` with u for exact equality, and fp rounding makes them differ without clipping. The
  counter now counts true clips (u + shift < 0). Re-measured on design prompt 0: true clip rate **1.89%** of all A4
  elements and **5.2%** of fc2 inputs with feedback; **0.0001%** with the guard. The `clip_rate` field in
  `ext/e1_w4a4_feedback_screen.json` holds the artifact value; true rates come from runs after this fix. The guarded arm
  stays (clipping is real, if smaller).
- 23:00 UTC follow-on decision: E1 promising=0, PixArt flat=0, E2 best=none
