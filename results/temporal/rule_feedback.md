# Temporal error feedback and structured dither: staged protocol (pre-registration)

Written 2026-10-27 10:05 UTC, before stage 1 runs. This replaces B′'s arm list; **B′'s pre-registered read (±Δ/4 vs the
single table, `rule.md`) stands and is evaluated inside stage 3**. One A30; jobs run serially. Results go to
`results/temporal/stage{1..5}/`, generated from JSONs. Nothing above the marker changes afterwards; deviations are
appended below it, dated.

## Budget and accounting
- **Stages 1–4: 4.0 A30-hours.** Counted from the stage-`temporal` total of 3.5365 h logged before this protocol, so
  the stage-1–4 ceiling is a logged total of 7.54 h. Stage 5 needs separate approval.
- The B′ gate run (single-table calibration + B1′ on both beds), started 09:57 UTC before this handoff, measures stage
  2's core quantities with stage 2's quantizer. It counts toward stage 2 and this budget.
- **Gates:**
  - Stages 1–2 (≈ 0.9 h) gate the rest.
  - **Stop at any failed gate**, except read 1's pre-registered fallback: (iii) ≈ (i) cuts stages 3–4 to the feedback
    arm only.

## Conventions
- **Data separation:** design-side choices (calibration, any fitted quantity) use design/calibration data only:
  PixArt design prompts 32–63 of deepcompressor's set; DiT calibration seeds 1000–1031. Evaluation uses untouched
  sets: PixArt MJHQ-64 / MJHQ-256; DiT eval seeds 0–511.
  - Stage 1 is a sampler diagnostic, not a design choice. It uses DiT seeds 0–31, as A0 and the existing diagnostics
    did.
- **Statistical unit:** prompts are the unit when seeds share a prompt (PixArt: one seed per prompt, so the unit is the
  image). Bootstraps use 2,000 resamples, seed 0, 95% percentile CIs. "Beats by ≥ x% (CI excluding 0)" means the point
  reduction is ≥ x and the CI's lower bound is > 0.
- **Logged for every arm:**
  - per-step error energy and clipping rate;
  - temporal-mean error energy, ‖mean_t δ_t‖² (C/I as in `rule.md`);
  - prediction-error spatial bands (`objective/d2.py` bands);
  - endpoint latent error;
  - PSNR / LPIPS vs the same-seed FP output (diagnostic only).
  For any arm advancing past stage 3: FID-2k (DiT) and ImageReward on MJHQ-256 (PixArt).
- **Quantizer (stages 2–3):**
  - static per-tensor asymmetric uint8, ONE (Δ, z) per layer across all steps, W16;
  - α grid 1.00…0.20, calibrated on design data;
  - one code spare, as in `rule.md` B′, so offsets in [−Δ/2, Δ) never add clipping;
  - PixArt cross-attention K/V stay 16-bit; DiT covers all 112 block-linear inputs.
  Offsets and feedback enter the rounding operation itself. Compensation uses the deployed path: q_t − d_t feeds the
  next linear with its actual weights.

## Stage 1: replay the real error sequence (DiT-XL/2, ≈ 0.4 h)
- **Recording:** heig-8's actual guided prediction error δ_t = heig8(x_t^q) − FP(x_t^q), 4 eps channels of the
  CFG-combined output. It is taken along heig-8's own trajectory x_t^q, for eval seeds 0–31 × all 50 steps, under the
  eval sampler (DDIM-50, cfg 1.5, fp16 autocast, batch 32). E_t,s = ‖δ_t,s‖² per sample s.
- **Replay:** into the FP sampler from the same noise and labels. At every step t, e_t,s is added to the guided
  prediction's 4 eps channels with energy a²·E_t,s, at amplitude a ∈ {0.5, 1, 2}. Each pattern below is unit-normalized
  per (t, s), then scaled to √E_t,s:
  - **(i) as recorded:** δ_t;
  - **(ii) time-reversed:** δ_{49−t};
  - **(iii) sign-alternated:** δ_t·(−1)^t;
  - **(iv) shuffled across steps:** δ_{π(t)}, one fixed permutation π (generator seed 0), the same for all samples;
  - **(v) fixed direction:** m_s = mean_t δ_t,s, with alternating sign (−1)^t;
  - **(vi) fresh white direction each step:** i.i.d. Gaussian (seed 9000 + 100·t + s);
  - **(vii) coefficient-only control:** pattern (i) with the denoiser frozen to its FP predictions on the FP
    trajectory, so only the sampler coefficients propagate e_t. The model's output eps is replaced by FP eps_t + e_t.
- **Reported:** final latent error ‖x0 − x0_FP‖² per pattern and amplitude, mean over seeds, with bootstrap CIs on
  each ratio to (i).
- **Read 1:** the response is broad and relevant to real error iff, at a = 1, (iii) < 0.3·(i) and (vi) < 0.5·(i),
  **and** both inequalities also hold at a = 0.5 and a = 2. If (iii) ≈ (i), meaning (iii)/(i) ≥ 0.7, the alternation
  effect was specific to synthetic directions, and stages 3–4 are cut to the feedback arm only.

## Stage 2: single-table baseline diagnostics (both beds, ≈ 0.5 h)
- **Data:** PixArt 20 steps, design prompts. DiT 50 steps, calibration seeds 1000–1031.
- **Per layer type:**
  - the rounding vs clipping error split;
  - ρ_adj and the layer coherent fraction of the rounding error;
  - the fraction of inputs on a grid level (exact zeros first).
- **Network level:**
  - C/I vs the measured dynamic per-token A8 floor on the same bed;
  - prediction-error spatial bands.
- The same under A6 as a mechanism probe. (The B′ gate run measures all of this except the rounding/clipping split
  and the bands, which come as a supplement.)
- **Read 2:** there is a component to remove iff C/I − floor ≥ 0.03 on at least one bed at A8, **or** ≥ 0.06 at
  A6. If read 2 fails, the protocol stops after stage 2.

## Stage 3: control over the error, stateless vs stateful (≈ 1.6 h)
- **Setting:** W16A8-static single table, both beds. PixArt: MJHQ-64 paired images + latent. DiT: seeds 0–511.
- **Arms:**
  1. single table (baseline);
  2. ±Δ/4 symmetric period-2 offsets (B′'s arm; B′'s read applies);
  3. independent subtractive uniform dither, o_t ~ U(−Δ/2, Δ/2) i.i.d. per layer per step (fixed seed 0);
  4. 4-phase centred cycling, d_j = Δ(j − 1.5)/4, j = t mod 4, in two orders: (0,1,2,3) and (0,2,1,3);
  5. **full temporal error feedback:** per quantizer site keep r_{t−1} (the activation's shape, fp16), with r_0 = 0.
     Then u_t = x_t + r_{t−1}, q_t = Q(u_t), r_t = u_t − q_t, and q_t is fed forward. u_t is clipped to the range, and
     the clip rate is recorded. Check: Σ_t e_t = r_0 − r_T, verified numerically on one layer. Reported: per-step
     error energy (expected to rise), cumulative error, clipping rate;
  6. per-step scale tables (reference, recalibrated on the wide α grid).
  Note: offsets in arm 3 are in [−Δ/2, Δ/2) and covered by the spare code. Arm 4's offsets lie within ±0.375·Δ.
- **Read 3:**
  - **Feedback is the construction** iff arm 5 beats arm 1 by ≥ 10% latent MSE (CI excluding 0) on both beds and by
    ≥ 0.3 dB on PixArt, **and** beats each of arms 2–4 on latent MSE on both beds.
  - **Stateless dither is the construction** iff some arm among 2–4 clears that bar against arm 1 (≥ 10% latent on both
    beds, CI excluding 0; ≥ 0.3 dB on PixArt) and arm 5 does not add ≥ 5% latent over it.
  - If arm 5 shows verified local cancellation (bounded cumulative error) but no endpoint gain, the record states that
    the downstream response destroys the cancellation, and the protocol stops after stage 3.
  - B′'s read (`rule.md`) is evaluated on arms 1, 2, 6 and the golden-ratio arm. The golden-ratio arm is needed by B′'s
    read and is run on PixArt.

## Stage 4: make the winner cheap (≈ 1.5 h, only if read 3 passes for feedback)
- **Compressed feedback:** the residual is kept in a k-dimensional spatial subspace U per channel.
  q_t = Q(x_t + U r_{t−1}), r_t = Uᵀ(x_t + U r_{t−1} − q_t), with state k × C per site.
  - Arms: k = 1 (spatial mean), k = 4 and k = 16 (lowest 2-D DCT modes), full; plus a random orthonormal spatial
    subspace at k = 16 (control, fixed seed).
  - Reported: the state–quality curve and the prediction-error bands.
- **Read 4:** low-frequency state is the right compression iff DCT-16 captures ≥ 70% of full feedback's latent gain
  and beats random-16 by ≥ 3%; otherwise the curve is reported as is.
- **SOTA arm** (if the identity-level mechanism holds): the corrected H-SVDQuant-r4 base at W4A4 (dynamic per-token A4)
  with DCT-16 feedback, vs the base. MJHQ-64, plus ImageReward on MJHQ-256.
- **CFG ablation (PixArt):** shared vs independent phases / feedback state across the two branches. Reported: the
  cross-branch error covariance and the guided error.

## Stage 5: separate approval
Two architectures, two samplers, short vs long trajectories. FID / ImageReward against FP and against Q-Drift and DNS.
A tiny compiled producer–quantizer–linear graph on QNN or ExecuTorch-Exynos, to test whether per-step offsets or a
residual state survive lowering, before any NPU claim.

## Corrections carried into every report (from review)
- Symmetric offsets give zero-mean *offsets*, not zero-mean error for every input.
- The −0.5 dither-correlation bound holds for an unclipped uniform quantizer with uniform phase.
- Weight error is strongly step-coherent *in our measurements*, not by construction.
- A0 shows rejection of the tested patterns, not a universal low-pass law.
- The NPU premise ("offsets admitted, scales not") is unverified and is not stated as fact. The Qualcomm demo was A16.

---

## Deviations and results
(appended below as they happen)
- 2026-10-27 10:35 UTC, calibration: the α search's direct re-quantization of tokens (123 range candidates per
  (layer, step)) ran ≈ 1–2 s per (layer, step), ≈ 1–1.5 h per bed. The first DiT calibration was killed after
  0.53 A30-h (hand-logged in `gpu_hours.csv`; no tables produced).
  - Replaced by a 65,536-bin histogram per (layer, step) over **all** tokens, with errors evaluated at the bin centres.
    On a heavy-tailed test it picks the identical α for all three table kinds, its errors agree to 3e-5, and it is
    ≈ 300× faster.
  - For PixArt this also replaces the 256-token subsample with all tokens.
  - Stage 2's rounding/clipping split (clip energy share, clip rate) and DiT's prediction-error bands are added to the
    B1′ run.
- 2026-10-27 (user decisions before stage 3):
  - **Batch:** the feedback arm on PixArt runs at batch 1. The residual state is per sample and never mixes across a
    batch (on DiT, each CFG row keeps its own residual and the state resets at step 0 of every sampling batch).
  - **PixArt arms cut to four:** single table, full feedback, the best stateless arm from DiT (among arms 2–4, by DiT
    latent MSE), and the per-step reference. Reason: budget. A full PixArt stage 3 (8 arms × ≈ 0.27 h, more at
    batch 1) does not fit the remaining stage 1–4 budget; DiT runs all arms cheaply and selects the stateless arm.
    The other two stateless arms are cut on PixArt.
  - **If DiT's read 3 fails for feedback,** PixArt runs only the single table and the best stateless arm.
  - B′'s "beats golden-ratio" clause cannot be evaluated on PixArt unless golden-ratio is the selected stateless arm
    (it is not among arms 2–4); it is reported as not evaluated.
- Stage 2 read 2 on DiT: single-table A8 C/I 0.269 vs floor 0.177 → excess **+0.092 ≥ 0.03: PASS** (stage 3 opens).
  B′'s separate ρ condition fails on DiT: top-half ρ_adj under ±Δ/4 = +0.85, because the single layer carrying half the
  output error (b26.mlp.fc2) is 71% clipping error (21% of fc2 inputs clipped, from the α·[min, max] shrink cutting the
  dense post-GELU negative lobe). Calibration rule kept as pre-registered (the user did not pick the amendment).
- Feedback definition detail fixed before running: u_t = x_t + r_{t−1}, then ũ_t = clip(u_t) to the table range,
  q_t = Q(ũ_t), r_t = ũ_t − q_t (rounding residue only; the clip residue is not fed back). The identity checked
  numerically on one layer (b13.mlp.fc1, first batch) is Σ_t e_t = −r_T + Σ_t (ũ_t − u_t), which reduces to
  Σ e = r_0 − r_T when nothing clips.
- 2026-10-27 (user-approved amendment (b), before any stage-3 arm):
  - **Calibration:** every static quantizer (single table, per-step tables, A6), both beds, searches lo and hi
    independently on the same grid: lo = α_lo·min, hi = α_hi·max, with α_lo, α_hi ∈ {1.00, …, 0.20} (41 × 41).
    Errors come from the 65,536-bin histograms. Calibration and B1′ are re-run on both beds, and stage 2's read 2 is
    evaluated on the re-run.
  - **Superseded:** DiT B1′ under the old α·[min, max] rule. Files renamed `*_SUPERSEDED_alpha_shrink.json`; its
    numbers stay in the report as a finding about static A8. A single static range on a post-GELU input is
    clipping-limited: fc2 clip rate 21%, 71% of fc2's error energy. Clipping error is step-constant, and no dither can
    move it.
  - **fc2 check:** after recalibration, the clip rate of the post-GELU inputs (DiT mlp.fc2, PixArt ff.net.2) is
    reported. If it is still > 1% on a bed, that bed's post-GELU inputs use the standard one-sided split before
    stage 3: unsigned, zero point at the lobe's minimum (lo = min, only hi searched).
  - **Feedback rule:** the residual fed forward is the rounding residual only, clamped to [−Δ/2, +Δ/2]; clipping error
    is never fed back. The clamp rate is reported. (With ũ clipped to the table range before rounding, |ũ − q| ≤ Δ/2
    already holds, so the clamp is expected to be inactive. It is kept as a guard.)
  - **Smoke gate before the DiT arms:** a 2-seed W16A8 single-table run (seeds 0–1, no dither) must give latent rel.
    MSE < 0.068 (heig-8's W4 weight-only error). Otherwise stage 3 does not start.
- 2026-10-27, calibration detail found while testing (b), before any re-run: the integer zero point
  z = round(−lo/Δ) can move the range's lower edge above lo. A post-GELU input has a density spike at its minimum
  (−0.17, where GELU's slope is zero), so a fraction of a step there clips a large share of elements: 7.3% on a
  synthetic GELU test at α_lo = 1. The zero point is now nudged, z = ceil(−lo/Δ), so the range always covers lo (the
  top shrinks by < Δ). This applies to every static table of the re-run. The superseded runs used the rounded zero
  point; that likely contributed to their 21% fc2 clip rate.
- 2026-10-27 (user decision after the failed smoke gate; recorded before any of it runs):
  - **Smoke gate result:** single-table W16A8 latent rel. MSE 0.321 (seeds 0–1) and 0.336 (seeds 0–31). This is range,
    not clip: all clip rates < 0.015% after the one-sided split. Per-type ablation: fc2 alone 0.197, fc1 0.090,
    qkv 0.022, proj 0.0003. One static range across 50 steps forces Δ = 0.14 on b26.mlp.fc2.
  - **Smoothing:** static per-input-channel smoothing (SmoothQuant form) on the DiT mlp.fc1 and mlp.fc2 inputs:
    x̃ = x / s, W̃ = W·diag(s), s_c = sqrt(max|X_c|) / sqrt(max_i |W_ic|) (migration strength 0.5). max|X_c| is taken
    over the calibration trajectory (seeds 1000–1031, all steps, both CFG halves, all tokens).
    - Fold: for fc1, 1/s folds into the adaLN modulation output (shift and scale rows). For fc2, GELU blocks the fold
      into fc1, so 1/s is a static per-channel multiply before the quantizer. W·s folds into the fc2 weights in both
      cases. Encodings stay fixed per tensor, with no runtime statistics. Simulated as quantize(x/s)·s into the
      unchanged linear (algebraically identical).
  - **Recalibration:** independent lo/hi, nudged zero point, and the one-sided split for fc2 applied directly
    (post-GELU inputs stay one-sided after division by s > 0). Then B1′ and the smoke test. The smoke figure is now a
    sanity figure, **not a gate**.
  - **Feedback-induced clipping guard** (the feedback arm): x_c = clip(x) as in the baseline;
    r_fed = clamp(s_{t−1}, lo − x_c, hi_rep − x_c); u = x_c + r_fed (always inside the representable range); the
    discarded part d_t = s_{t−1} − r_fed has its energy logged per site. The clip rate therefore equals the baseline's
    by construction. Identity with the discarded term: Σ_t(q̂_t − x_t) = −s_last + Σ_t c_t + Σ_t η_t − Σ_t d_t, with
    c_t = x_c − x. The unguarded version is kept as a diagnostic row in the matched diagnostics, not as an arm.
  - **Stage 3:** DiT only, the registered arm list, on the smoothed baseline, within the remaining ≈ 2.0 h. PixArt's
    arms wait for a new allocation, requested only if DiT's read 3 passes.
  - **Known, not re-run:** the scorer cells over 1% (DiT b26.mlp.fc1 step 49, 1.24%; PixArt ff.net.2, 2.2%).
- 2026-10-27 12:40 UTC, accounting correction: in the amendment-(b) DiT and PixArt calibrations and the smoothed DiT
  calibration, the 41 × 41 table search ran after the run logger had exited, so ≈ 0.33 A30-h was not logged. It is now
  hand-logged from logger-end to log-file mtime (0.131 + 0.106 + 0.088 h). Logged total after the correction: 5.92 h.
- 2026-10-27 (user approval): +0.45 A30-h so all six registered DiT stage-3 arms run (each arm costs ≈ 0.22 h, not
  0.10 h as estimated). Stage 1–4 ceiling raised from a logged total of 7.54 h to 7.99 h. Arm order changed so that the
  key arm runs first: ±Δ/4 (already running), feedback, per-step reference, uniform, 4-phase (0,1,2,3), 4-phase (0,2,1,3).
