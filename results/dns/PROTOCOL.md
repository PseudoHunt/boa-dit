# DNS reproduction and a covariance-aware extension (protocol, 2026-10-28, committed before any DNS number)

Stage tag `f6`. Branch `dns-cov` (worktree `/home/boa-dit-dns`), from `followup-weights` at c0edc15, which carries the verified heig-8
baseline: `rebuild.py` reproduces the stored carry bit for bit, and E3 `check` confirmed the bitwise export and sampling.

**Budget.** The protocol paste set a 4.0 A30-h cap. Afterwards the user removed GPU caps ("there is no gpu budget anymore. just keep running
the experiments until they are over"), so the cap is recorded but not enforced. Every GPU job is logged under stage `f6`, failed jobs and
calibration included. CPU work is logged separately in `results/dns/cpu_log.md`. DNS completes before the extension's confirmation starts.

## 1. Sources (read in full before implementation)
- **Paper:** "Absorbing Quantization Error by Deformable Noise Scheduler for Diffusion Models", Yang, Huang, Sheng et al., ICML 2026, PMLR 306
  (https://proceedings.mlr.press/v306/yang26aa.html). The PDF was fetched 2026-10-28 from the PMLR asset
  `mlresearch/v306/main/assets/yang26aa/yang26aa.pdf` (31 pages, main text and Appendices A.1–A.5).
- **Official code:** https://github.com/ZephyrYoung-eYuan/DNS_AQE, commit `3f02e6932edb30f58c0471456501c34e85517e27`
  (2026-05-28, "Initial release: DNS-AQE"). Files read: `quant/dns_main.py`, `quant/dns_func.py`, `quant/dns_test.py`,
  `PTQD/quant_scripts/main_dns_ldm.py`. `quant/analyze_error_func.py` is not called by the entry point.

### Record (paper eq. numbers / code locations)
- **Error decomposition (Eq. 8; `calculate_residuals`).** Δε = ε′ − ε (student minus teacher), computed after CFG combination.
  - Per output channel, the code fits an OLS line Δε_c = k_c·ε_c + b_c (`scipy.stats.linregress`), pooled over samples and pixels.
  - The residual r = Δε − (k·ε + b) is pooled over all channels.
  - Eq. 8 has a scalar k and no intercept; the code uses per-channel k_c and b_c (Alg. 1 note: "d(t) is usually very close to 0").
- **Variance and kurtosis (`dns_slope_calcu_loop`).**
  - Raw std σ_raw. Residuals are clipped to ±6·1.29·σ_raw.
  - σ_o = IQR/1.349 of the clipped residuals (the code's default `std_method='iqr'`, and Table 7/14's choice).
  - κ_o = Fisher excess kurtosis (scipy, biased) of the clipped residuals.
- **DCNC (Eq. 13/23; same function).** If κ_o > 0: σ_u² = σ_o²·sqrt(5κ_o/6), the uniform half-width a = sqrt(3σ_u²)·W_u, and σ² = σ_o² + σ_u².
  - The code uses the analytic variance at W_u = 1 even when W_u < 1 (comment: "corresponding to uni_weight = 1").
  - Paper Alg. 1 instead takes the variance of the transformed residuals.
  - If κ_o ≤ 0: σ² = σ_o² and no noise.
  - Default W_u = 0.2 (`main_dns_ldm.py`, Fig. 17/18: "a single fixed default of W_u=0.2 for all models"). The default DCNC distribution is uniform.
- **Where DCNC noise enters (`model_forward_tea_stu`).** U(−a, a) is added to the raw student output of **both** CFG halves (all output channels), before
  CFG combination, independently per element. It uses a separate generator (seed 7). It is not added at the last step.
  This makes the deterministic sampler stochastic.
- **Correction of the prediction (Eq. 14; `aqe_dns_loop`).** ε̃ = (ε′_cfg − b)/(1 + k) per channel, applied to the CFG-combined output (with the DCNC noise).
- **Scalar variance used by the scheduler.** σ²_ε,t = mean_c[σ²/(1 + k_c)²].
- **State/noise parameterization.**
  - VP, x_t = sqrt(ā_t)·x0 + sqrt(1 − ā_t)·ε, with ε-prediction.
  - The ā table is the full 1000-step training schedule, indexed by the original timestep.
  - Deterministic update x̂ = ws·x_t + we·ε̃, with ws = sqrt(ā′/ā_t) and we = C1 = sqrt(1 − ā′) − sqrt(ā′(1 − ā_t)/ā_t) (eps-space coefficient).
- **Paper vs code on C1.** The paper's Eq. 16 writes C1 as the x0-space coefficient, while its variance term C1²σ²_ε,t uses the eps-space σ²; the code is self-consistent in eps space.
  We follow the code. With σ_t = 0 the two differ by the factor sqrt((1 − ā_t)/ā_t) only through that inconsistency.
- **Schedule modification (Eq. 20; `solve_alpha_prod_pai`).** Solve ā′ ∈ [ā_prev, 1] with ā′ = ā_prev·(1 + C1(ā′)²·σ²_ε,t).
  - Solver: `minimize_scalar(bounded)` on the squared residual. Because the acceptance tolerance is η² = 1e-40, the fallback almost always runs:
    `scipy.optimize.root` from 50 guesses in [ā_prev, min(1.1·ā_prev, 0.99)], keeping the smallest residual inside the bounds.
  - Only "if prev_timestep ≥ t_d" (stride). Otherwise ā′ = ā_prev; the final step uses ā_prev = final ā = 1.
- **Map Φ (Eq. 19; end of `aqe_dns_loop`).**
  - C2² = 1 + C1²σ²_ε,t. τ = the largest timestep index with ā_τ ≥ ā′/C2², searched over the full 1000-step table.
  - If none exists, x̂ is returned unscaled (a fallback).
  - x ← x̂ / sqrt(ā′/ā_τ). Not applied at the last step.
  - With an exact solve, ā′/C2² = ā_prev, so τ = prev up to the table search and float rounding.
- **Denoiser timestep (Alg. 2 note; code loop).** The next call uses the **original** next timestep (t_prev, not τ or t^q_prev).
- **Calibration distribution (`dns_cali`).**
  - Teacher-generated samples (3 per class × 1000 classes = 3000), produced with the bed's own sampler: DDIM 20, cfg 3.0 for LDM.
  - At each sampler timestep: one fresh noise draw per sample, forward-noised with q(x_t | x0).
  - Teacher and student are evaluated on the **same forward-noised FP-sample states**: not FP trajectory states, not quantized states.
- **CFG.** Statistics are taken on CFG-combined outputs (cond half first). The paper's DiT setting is 20 steps, cfg 1.5.
- **Assumptions behind the preservation claim (Eqs. 15–17, 32–41).**
  - (a) Substituting x0 for x̂0, i.e. the FP denoiser is treated as exact, so the FP DDIM step preserves q(x_t | x0).
  - (b) The residual is zero-mean, independent of x_t, and Gaussian after DCNC.
  - (c) Errors at different steps are independent, and earlier errors do not change later residual statistics (no accumulation or propagation term).
  - (d) Statistics measured on forward-noised FP samples apply at the states the quantized sampler actually visits.
  - (e) One isotropic scalar variance per step.
  - The law-of-total-variance step (Eq. 37–39) relies on (a)–(c).
- **Applicability to DiT-XL/2 + DDIM-50.** The formulation is generic VP/ε-prediction DDIM with σ_t = 0, and DiT uses the same VP schedule (linear β)
  with guided-diffusion respacing, so it applies as written. The paper itself evaluates DiT-XL at 20 steps.
  DiT's learned-variance channels are unused by DDIM. The only bed-specific item is the CFG convention.

### Port decisions (implementation port, not assumption changes)
1. **Bed.** heig-8 W4 + rank-8 carry, fp16 autocast forward, DDIM-50 (respaced t = 980, 960, …, 0), cfg 1.5.
   - Final step: ā_prev = 1 (guided-diffusion `alphas_cumprod_prev[0]`), which equals the code's `final_alpha_cumprod`.
   - Stride t_d = 20, so the "prev ≥ t_d" rule skips the solve on the step 20→0 and the final step 0→end, exactly as the code does for its last two steps.
2. **CFG.** The repository's convention is kept: CFG on eps channels 0–2, channel 3 (and the variance channels) from the cond half.
   - Calibration statistics are taken on these combined 4 eps channels; k_c and b_c are per channel, c = 0..3.
   - DCNC noise is added to the raw 8-channel output of both halves before this combination, as in the code.
3. **Schedule precision.** The ā table is the bed's float64 table. The code compares in diffusers' float32 table; the float64 comparison is recorded as a port difference.
4. **DCNC noise seeding.** One generator per image, seeded from its sample seed (`seed·7919 + 7`). The noise therefore does not depend on batch composition.
   The code instead uses one global generator seeded 7.
5. **Calibration set.**
   - 3 FP samples per class with labels [0,0,0,1,1,1,…,999,999,999] and noise seeds 40000–42999, generated with the bed's sampler (DDIM-50, cfg 1.5).
   - Forward noise at step index i comes from a generator seeded 45000 + i (order = sample index).
6. **Nothing else changes.** No change to weight codes, scales, carry, activation precision, CFG, or denoiser call count (50 calls, each one CFG-doubled batch).
   Q-Drift is not substituted anywhere.

### Correctness checks (before any quality metric; `results/dns/checks.json`)
1. Correction disabled → bitwise equal to `ddim_sample` (heig-8 + carry).
2. Zero statistics (k = b = σ = a = 0) → the original schedule/update (ā′ = ā_prev, τ = prev, C3 = 1) and an endpoint equal to the baseline within float tolerance.
   Any ulp-level τ jump is reported.
3. Coordinates: the explicit DDIM step in the DNS loop reproduces guided-diffusion `ddim_sample` per step.
4. The denoiser timestep equals `timestep_map[i]` (original t), and the sampler ā_t equals the table at that t, for all 50 steps.
5. The CFG convention matches `ddim_sample` (identical combined output at zero correction).
6. All ā′, τ and C3 are finite, with ā′ ∈ [ā_prev, 1], and boundary steps follow the code's rules.
7. Inference reads only frozen calibration statistics and the quantized model output.
8. Denoiser call count = 50 (counted by a forward hook).
9. Synthetic comparison of one update against the official `aqe_dns_loop` (imported from the commit above, with a mock model and a scheduler carrying the bed's table).

Per-step traces on design seeds: ā_t, ā_prev, ā′, τ, ws, we, C2², C3, the DCNC half-width, the correction magnitude ‖x_DNS − x_DDIM-step‖/‖x‖ at the same state,
solver-fallback use, and τ-not-found / rescale-skipped events.

## 2. Seeds (registry sweep 2026-10-28; every DiT seed < 10000 is used somewhere, 20000–39999 belong to `e3-confirm`)
| use | seeds |
| --- | --- |
| DNS calibration FP samples | 40000–42999 (labels: 3 per class, the official protocol) |
| design / stability checks, traces | 43000–43031 |
| covariance model: fit (paired FP / heig-8 free-running trajectories) | 44000–44063 |
| covariance model: held-out validation | 44100–44163 |
| covariance model: λ/selection, if any | 44200–44231 |
| **confirmation FID-2k** | **30000–31999**, shared with `e3-confirm` (see below) |
| confirmation paired diagnostics | 20000–20127, shared with `e3-confirm` |

The confirmation sets are untouched by every DNS and extension choice. `e3-confirm` also evaluates on them, but nothing there is tuned against DNS.
The heig-8 baseline on them (E3C arm **C0**) uses the same model artifacts (rebuild checkpoint + carry), sampler (`ddim_sample`, DDIM-50, cfg 1.5, fp16 autocast),
seeds, labels, VAE decode and metric code (pytorch-fid features vs `results/stage0/ref_stats.npz`; LPIPS-VGG; PSNR on uint8).
So its latents, FID and per-image features are **reused** as the baseline, not regenerated. If the E3C C0 run is not complete when the DNS confirmation
starts, the baseline is generated here with the same code.

## 3. DNS arms
- **Baseline:** frozen heig-8 (reused, as above).
- **DNS (faithful):** the code's defaults (W_u = 0.2, uniform DCNC, IQR), calibrated as in §1. No tuning. The design seeds are used only for stability and traces.
- **Q-Drift reference:** the frozen c_i from `results/temporal/sampler_corr/qdrift_calibration.json` (seeds 6000–6063), applied as before (eps hook ×(1 + c_i)).
  It is re-sampled on the confirmation seeds; its old FID (seeds 0–1999) is not comparable and is not reused.

**Confirmation.**
- FID-2k for baseline, DNS and Q-Drift (same reference statistics, labels and seeds), with per-image Inception features saved.
- Paired bootstrap over the generated images (same resampled indices for every arm, 1,000 resamples), against the fixed reference statistics.
  The result is reported as conditional on the reference and on n = 2000, not as a population estimate.
- Secondary: same-seed latent rel. MSE, PSNR and LPIPS vs FP on 20000–20127. DNS may move distributional and paired metrics in opposite directions; both are reported.
- Costs reported: calibration (GPU sampling + CPU statistics), inference overhead (ms/image), memory, and denoiser calls.

## 4. Covariance-aware extension
The mathematics (quantities, propagation model with the denoiser response, the explicit update and its limits, and the matched temporal-independence
ablation) is committed as a dated amendment **before its implementation and before any of its results**. Nothing is run under this heading until then.

---

## Amendments
(dated, appended before the affected arms run)

### 2026-10-28 17:30: §4 covariance-aware extension, mathematical specification (before implementation and before any of its results)

**Notation.** Deterministic DDIM step i (t_i → t_{i+1}), x_{i+1} = a_i x_i + c_i ε(x_i), with a_i = sqrt(ā_{i+1}/ā_i) and
c_i = sqrt(1 − ā_{i+1}) − sqrt(ā_{i+1}(1 − ā_i)/ā_i). These are exactly DNS's ws and we at ā′ = ā_{i+1}.
Bands b = 1..4 are the equal-area radial DFT bands (`objective/d2.band_masks`). Per-band quantities are averaged per element, and bands add exactly (Parseval).

**A. Quantities** (paired free-running FP / heig-8 trajectories, same noise and labels; fit 44000–44063, held-out 44100–44163).
1. Local prediction error at the quantized latent: ξ_i = ε_Q(x^Q_i) − ε_F(x^Q_i). Both networks run on the same state, so this needs one extra FP call per step in calibration only.
2. Accumulated free-running state error: d_i = x^Q_i − x^F_i.
3. Their means across seeds, m_ξ,i and m_d,i (per element). They are reported, and removed from every covariance below.
4. Temporal prediction-error covariance: S_b(i, j) = E⟨ξ_i,b − m, ξ_j,b − m⟩/D_b (all lags), and its lag-1 correlation.
5. State-error / current-error covariance: C_b(i) = E⟨d_i,b − m, ξ_i,b − m⟩/D_b. This is **not** the adjacent-error correlation S(i, i−1).
6. FP-state / accumulated-error covariance: X_b(i) = E⟨x^F_i,b − m, d_i,b − m⟩/D_b.
7. Uncentred second moments of ξ, d and the cross products, reported alongside the covariances (never substituted for them).

**B. Propagation model with the denoiser response** (per band b, scalar coefficients; a spatially isotropic within-band parameterization).
- Exact identity: d_{i+1} = a_i d_i + c_i Δ_i + c_i ξ_i, where Δ_i = ε_F(x^Q_i) − ε_F(x^F_i) is the FP denoiser's response to the state error (measured, not assumed).
- Linear band-diagonal closure: Δ_i,b ≈ J_i,b d_i,b, so

      d_{i+1,b} = A_i,b d_i,b + c_i ξ_i,b + r_i,b,       A_i,b = a_i + c_i J_i,b
      ξ_{i+1,b} = F_i,b ξ_i,b + ν_i,b

  - J_i,b is the least-squares regression of Δ_i,b on d_i,b (fit seeds, centred). F_i,b is the least-squares lag-1 regression of centred ξ. ν and r are residuals.
  - Reference gain: a noise-like perturbation of an exact VP marginal under an ideal denoiser propagates with g_i = sqrt((1 − ā_{i+1})/(1 − ā_i)).
    This is the eigenvalue of a_i + c_i J for J = 1/sqrt(1 − ā_i), the noise direction. Measured A_i,b is compared against it.
- **Joint covariance recursion** (V_d, C, V_ξ per band; V_ξ,i taken from data at every step, so the marginal variances are matched exactly):

      V_d,i+1 = A² V_d,i + 2 A c C_i + c² V_ξ,i            (+ Var r_i, reported)
      C_{i+1} = A F C_i + c F V_ξ,i
      V_d,0 = C_0 = 0

  - The C-recursion assumes the innovations ν_i are uncorrelated with d_i, ξ_i and r_i.
  - The V_d recursion assumes r_i is uncorrelated with d_i and ξ_i.
  - All four correlations are tested on held-out data (|corr| > 0.1 is flagged as a violation).
- **Matched temporal-independence ablation.**
  - Kept the same: m, V_ξ,i (marginals), A_i,b, the band parameterization, and the data and fitting procedure.
  - Removed: all cross-step dependence of ξ, consistently: F ≡ 0 with V_ν,i = V_ξ,i+1, so ξ_{i+1} is independent of (d, ξ)_{≤i}.
  - This implies C_i ≡ 0 for all i and V_d,i+1 = A² V_d,i + c² V_ξ,i. It is a self-consistent independent model, not one zeroed term inside a dependent recursion.

**C. Explicit sampler update ("C-DNS").** Everything is identical to faithful DNS (`port.py`): the same affine correction T, DCNC, solver, map Φ,
the original next timestep passed to the denoiser, one global schedule, and the same calls. The one difference is DNS's per-step variance σ²_ε,i, which is replaced by

      σ̃²_i = ρ_i · σ²_ε,i,      ρ_i = max(0, ΔV_i / (c_i² V_ξ,i)),      ΔV_i = Σ_b w_b [ (A_i,b² − g_i²) V_d,i,b + 2 A_i,b c_i C_i,b + c_i² V_ξ,i,b ],

with w_b = 1/4 (equal-area bands). V_ξ,i = Σ_b w_b V_ξ,i,b, and V_d, C come from the recursion above (full model) or the ablation (C ≡ 0, its own V_d).
- **Quantity corrected.** The same as DNS: the excess of the state's marginal second moment over the VP marginal at the next timestep.
- **Derivation of ΔV.** DNS's Φ renormalizes after every step, so variance it has already absorbed is treated as diffusion noise, and an ideal denoiser propagates noise with gain g_i.
  The excess that step i must newly absorb is the growth of the error variance beyond that reference propagation, V_d,i+1 − g_i² V_d,i.
  The recursion gives exactly ΔV_i. Its three terms are:
  - the denoiser-response deviation from noise-like propagation, (A² − g²) V_d;
  - the temporal cross-covariance, 2 A c C;
  - DNS's fresh-error term, c² V_ξ.
- **Ratio form.** The ratio transfers the model's relative effect onto DNS's own calibrated σ² (measured on forward-noised FP samples after T and DCNC).
  This keeps every other DNS quantity fixed.
- **Coordinates.** VP x_t; ε-prediction; the full 1000-step ā table; the denoiser gets the original next timestep (as DNS).
- **Approximations (labelled).**
  - (i) Linear band-diagonal propagation; cross-band coupling and nonlinearity go into r.
  - (ii) Open-loop closure: statistics come from uncorrected heig-8 trajectories but are used in the corrected (closed-loop) sampler. Valid when C3 − 1 is small; C3 is reported.
  - (iii) d is treated as independent of x^F for the marginal: the term 2·X_b(i) is **not** modelled. Its measured size is reported as a limitation.
  - (iv) One isotropic scalar per step after band pooling.
  - (v) The ρ clamp at 0, because DNS's map can only add timestep shift (ā′ ≥ ā_prev). Clamp events are counted.
- **No preservation guarantee is claimed.** DNS's own guarantee rests on assumptions (a)–(c) in §1, and C-DNS adds (i)–(v).
- **Not done.** No per-band timesteps. No separate addition of drift rescaling or noise subtraction: C-DNS changes only the variance fed to DNS's single schedule deformation, so there is no double counting.
  No replacement of local variance by accumulated variance: ΔV is the increment derived above, not V_d.
- **Limits (verified numerically before use).**
  - (L1) F ≡ 0, C ≡ 0 and A_i,b = g_i ⇒ ΔV_i = c_i² V_ξ,i ⇒ ρ ≡ 1 ⇒ C-DNS schedule == DNS schedule exactly.
  - (L2) σ²_ε,i = 0 ⇒ C-DNS == the uncorrected sampler (ā′ = ā_prev, C3 = 1), for any ρ.
  - (L3) ρ_i = 0 ⇒ no deformation at step i.

**§5 gate (held-out seeds 44100–44163, before any C-DNS sampling).**
- **Targets:**
  - (a) the band-wise trajectory V_d,i,b (open-loop accumulated state-error variance), predicted by recursion from i = 0;
  - (b) the one-step increment ΔV_i,b, computed with the empirical V_d,i and each model's C_i.
- **Score.** Mean over steps and bands of |log(predicted/empirical)| for (a), and normalized absolute error for (b), with steps where the empirical V_d is below 1e-8 excluded.
- **Pass** if the full model's score on (a) is lower than the ablation's by ≥ 10% relative, **and** the seed-bootstrap 95% CI (over held-out seeds, 1,000 resamples) of the score difference excludes 0.
  Score (b) is reported, not gated.
- **On pass:** freeze ρ (full and ablation) and run DNS, C-DNS-ablation and C-DNS-full on the confirmation sets with the same metrics as §3.
- **On fail:** report the specific failure (which target, which bands or steps, and which innovation assumptions are violated), keep the DNS result, and mark the C-DNS confirmation **unrun**.
  This is not a negative result for covariance-aware correction in general.
- **Activation feedback is not combined here.**

### 2026-10-28 17:45: check 2 criterion (amendment after the first check run, before any DNS quality number)
The first `checks` run passed every item except c2. Its zero-statistics endpoint differed from `ddim_sample` by 1.6e-3 relative error, against
the pre-set 1e-4 bound. At the same time the zero-statistics schedule was an exact identity (ā′ = ā_prev, τ = prev, C3 = 1) and the explicit
DDIM step agreed with guided-diffusion's to 4e-7 (c3). The port computes the same update as guided-diffusion in a different algebraic form
(ws·x + we·ε vs x0-then-ε′). Under fp16 autocast, last-bit differences in the state are amplified over 50 network calls; E3's check already
needed 1e-3 over 5 calls.

**Amended criterion.** c2 passes if the same comparison with fp32 forwards (amp off, both samplers) is below 1e-5. The fp16 number stays
reported. All other checks are unchanged. No DNS statistic or sample existed when this was decided.

### 2026-10-28 20:40: check results note
`checks.json` passes, with c2 under the amended fp32 criterion (3.2e-6; the fp16 value is 1.6e-3, as before).
Its c1 item (`base_equals_e3c_C0_latents`) is false because it sampled a batch of 4, and fp16 kernels differ with batch size.
Re-run with E3C's batch of 32 (seeds 20000–20031), the baseline is **bitwise identical** to E3C C0 (`check_c1_batch32.json`).
All DNS-side sampling uses the same aligned batches of 32, so the baseline reuse stands.
