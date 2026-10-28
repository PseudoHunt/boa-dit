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
