# Quantizer-informed error observability (protocol, 2026-10-30; frozen before any new result)

Stage tag `qo1`. Branch `quantizer-observer-pilot` from `dit-joint-feedback` @ 2c801728f75f9d855b47100d81bd05384ce170da.
Reviewed heads: weights-joint 1244c6fc, dns-response-pilot d9eab7f2, analytic-carry-pilot 3ce75311 (read-only, parked method).
Code `boa_dit/quantizer_observer/{run.py, analysis.py}`; outputs `results/quantizer_observer/`; caches `ckpt/quantizer_observer/` (not committed; hashes in manifest files).
The user's 2026-10-30 instruction is authoritative. This file records how it is implemented. A diagnostic and conditional pilot only, not a DNS extension.
Parked and not touched: closed-form carry fitting, new weight tables, scale modulation, PixArt, precision switching, FID.

## 0. Grounding, budget
- No AGENTS.md exists in the repository; `CLAUDE.md`, the joint protocol/report, the observability report and the DNS-response report were read.
- No other GPU job is running.
- **Hard ceiling 2.0 A30-h** (stage `qo1` rows in `results/gpu_hours.csv`; failures and profiling included). Planning envelopes:
  checks/profile 0.15, traces 0.65, regression GPU 0.10 (the regression runs on CPU), live 0.60, reserve 0.50.
  Each GPU stage refuses to start if spent + its profiled estimate exceeds the ceiling. If the cap binds, the live arm goes first, then the dev original-carry comparison.
- **Bed:** DiT-XL/2 256, DDIM-50, cfg 1.5, common smoothed W4 base, frozen single8 A8, DCT-4 feedback (`dit_joint.JointQuant` 'fb').
  Primary carry: **`ckpt/dit_joint/carry_E_L_u100.pt`**, sha256 c2b70026eee388586c7a6881128469be360b2d3c26061819cd996507849f3550. It is the development-selected E_L checkpoint of the joint pilot, frozen before this work.
  Secondary: the original common carry (configuration B), dev seeds only. The joint-training primary read did not pass; integration is not described as established.
- **Seeds** (swept 2026-10-30 over every branch's code/protocols and `handoff/SEED_REGISTRY.md`: unused; labels from seeds as usual; splits by whole image):
  fit 90000–90063 (64), development 90100–90131 (32), observer confirmation 90200–90263 (64), live confirmation 90300–90363 (64).
  Confirmation traces are generated in advance. Their target-dependent summaries (errors, headroom, metrics) are not opened until `observer_select.json` is committed.
- **Cache inventory:** joint eval latents and C/I aggregates are insufficient. The old observability traces are heig-8 W4A16, teacher-forced on FP states, with no carry or quantizer signals, so they are not reusable. No unpublished local observer test was found. Details in manifest.json.

## 1. Instrumented measurement (no change to deployed arithmetic)
`InstrQuant` subclasses JointQuant. The consumed value is returned by the **parent `_fb`** unchanged; afterwards the instrument recomputes the identical fp32 intermediates
(asserted: recomputed q == consumed, 0 mismatches). Terms, per site/step, in float64 from the actual fp32/fp16 values:
x (input after smoothing), p = U s_prev (fp32, as code), xcl, c0 = xcl − x, r_fed, g = r_fed − p, u (fp32 as code), **rho_u = u − (xcl + r_fed)** (fp32 rounding of the add,
an explicit extra term), uc, c1 = uc − u, q (consumed), r_raw = uc − q, r = clamp(r_raw, ±Δ/2), a = r_raw − r, P = U_d U_dᵀ (retained DCT-4), p_next = U_d s_next (fp16 stored), eta = p_next − P r.
**Identity:** q − x = p − p_next + c0 + g + rho_u + c1 − (I−P) r − a + eta (exact in float64 up to roundoff).
The full 9×9 Gram (energies and cross terms) is accumulated on the first fit batch (32 images × 2 CFG rows, all steps/sites); rates and energies on all images.
Rates: input clip (xcl ≠ x), state guard (r_fed ≠ p), second clip (u ≠ uc, the old counter), rounding clamp (|r_raw| > Δ/2), spare-code use (code == qmax).
Range, spare-code rule and guards are not changed. This is site-level accounting, not an attribution of endpoint damage.
**Checks (`checks.json`)**, before traces:
- c1: quantization off (JointQuant 'off') reproduces the W path, bitwise.
- c2: the instrumented trace (Gram, features, references) reproduces deployed WA sampling (`J.sample_config`), bitwise.
- c3: rows keep separate states (changing image 2's noise leaves images 1, 3, 4 bit-identical; state rows = 2n).
- c4: a repeated trace is bitwise.
- c5: identity relative residual energy ≤ 1e-20 (squared float64 roundoff) with denominator max(||q−x||², 1e-12·mean); recompute mismatches = 0.
- c6: e_total = e_A|W + e_W against an independent FP call on stored WA states, ≤ 1e-6 relative.
- c7: 50 calls per site.

## 2. Experiment A
**A1.** Per split, matched endpoints from identical noise/classes:
- F: FP, deployed autocast path;
- W: common W4 + E_L u100 carry, A16, feedback off;
- WA: the same + static A8 + DCT-4 (the trace pass's own trajectory).

On dev, W and WA with the original carry, reusing F. Terms normalized by ||x_F||² per image: ||d_W||², ||d_A||², 2⟨d_W, d_A⟩ (closure reported), intervention gain 1 − mean MSE_W / mean MSE_WA.
Paired 95% image bootstrap (2,000, rng 0). This is the effect of removing activation quantization **conditional on these weights/carry**.
**A2.** Inside the deployed sampler's eps hook at step t (after the WA call, with WA's real incoming state), the separate W and FP models run at the same latent/timestep/labels/CFG.
They have no quantizer hooks, so WA's state advances exactly once per WA call. Stored: e_A|W = f_WA − f_W, e_W = f_W − f_F (guided 4-channel sampler-facing ε), f_WA, x_t, final latents.
Separate conditional/unconditional diagnostics are **not** stored (cost).
Statistics per source (eW, eAW, etot) on fit+dev: energy, lag-1/lag-5 cosine, spatial DC fraction, first 4×4 spatial DCT fraction, radial FFT bands (`objective.d2.band_masks`),
orthonormal temporal decomposition {constant, period-2, remaining}, optional first four temporal DCT modes, and coherence c(e) = ||Σe_t||²/(T Σ||e_t||²) (iid ≈ 1/T).
**Activation prioritization (dev only):**
- gain ≥ 20%: proceed.
- point < 20% but CI spans 20%: uncertain. Cached regression is allowed, but no activation-feature live arm.
- upper CI < 20%: the activation-observer live arm is parked.

## 3. Experiments B / C (one trace pass collects all features)
- **C (carry):** the deployed carry hook is replaced by an identical-op hook that also stores the 2×2 spatial DCT (token grid, `token_basis('dct4')`) of its own bottleneck z = xV.
  That is [2 CFG rows × 112 sites × 4 × 8] per step. No new probe vectors; z is the carry's own matmul. The carry output D z is not used as a separate feature.
- **R:** the stored next DCT state, projected by a fixed seeded channel projection [C, 8] per site (seed 1000 + site index). The before-state is the previous after-state; the family is [R_t, R_t − R_{t−1}].
- **E:** per site/row energies of c0, g, c1, a, (I−P)r, eta (log10) and rates of input clip, guard, second clip, spare code.
- **S:** signed 2×2 spatial modes of c0 and g, projected the same way. Discarded modes are never projected onto the retained subspace.
- **H:** 4×4 spatial DCT of f_WA,t, f_WA,t−1, their difference, x_t; t/T; history-validity flag (zero history at t = 0).
- **Targets:** 4×4 spatial DCT (64 coefficients) of guided e_W (primary mechanism), e_A|W and e_total (secondary; the live corrector needs e_total).
- **Models:** M0 (per-step training mean), H, H+C, H+R, H+E, H+R+E+S, H+C+R+E+S (only if both families show dev value), Oracle = H + previous true e_total (diagnostic only).
- **Fitting:**
  - five fixed 10-step buckets;
  - per bucket, normalization/PCA fitted on training images only; each added family is reduced to ≤ 32 PCA components, standardized. The family C = [z_t, z_t − z_{t−1}] gets one PCA.
  - Target centered by the per-step training mean; ridge minimizes (1/N)Σ||y_c − Xw||² + λ||w||² on standardized X;
  - λ ∈ {1e-3, …, 100} by four-fold image-grouped CV on fit (fold = image index mod 4), per target and model.
  - Final models are trained on all fit images; dev and confirmation are prediction-only.
- **Shuffle controls:** 20 fixed seeds (1000+k). The added-feature trajectories are deranged across images, whole trajectory and same timestep, in training and evaluation; the CV λ of the real model is kept.
  PCA is invariant to the permutation. Class labels are mostly unique, so the shuffle does not separate class from image-specific information.
- **Metrics (low-band, per image):**
  - error-energy ratio (raw and centered);
  - relative change vs H with bootstrap CI;
  - median per-image improvement and wins;
  - temporal-mean error left ||Σ_t (y − ŷ)||² / ||Σ_t y||²;
  - prediction·error dot fraction and cosine.
- **Development value:**
  - carry: H+C ≥ 1% below H on e_W or e_total;
  - activation: any of H+R, H+E, H+R+E+S ≥ 1% below H on e_A|W or e_total.
- **Selection:** exactly one total-error observer by dev e_total MSE among H, H+C, the activation models (if allowed by the prioritization rule), and the combination (if triggered).
  Ties within 1% go to fewer features. It is frozen (`observer_select.json`, committed) before confirmation.

## 4. Experiment D: propagation audit (no GPU)
`propagation_audit.md`: an inventory of stored propagation/response data, signed vectors vs norm-only. A scalar schedule is not inferred from norm ratios. No gain sweep and no live schedule arm.

## 5. Observer gate and conditional live correction
**Gate (observer confirmation, e_total, low band):**
- (1) ≥ 10% lower MSE than H, paired CI excluding 0;
- (2) better than M0, and temporal-mean total error reduced (left fraction < 1);
- (3) shuffle loses the advantage: the real change vs H is below all 20 shuffles, and the shuffle mean is above half the real change;
- (4) finite predictions, and the measured cost fits the remaining budget.

These are prioritization thresholds. A null parks this observer design only.
**Live (only if the gate passes):**
- causal correction of the 4 guided ε channels, ε_corr = ε̂ − λ U ê (U = 4×4 DCT reconstruction), once per step, after CFG. Variance channels and feedback evolution are unchanged.
- λ ∈ {0, 0.25, 0.5, 1} by dev endpoint MSE; 0 wins → stop.
- Live confirmation (90300–90363) compares WA, corrected WA, and the H corrector with its own dev λ: paired latent/PSNR/LPIPS, cross terms, wall time, memory.
- Promising pilot: ≥ 5% endpoint MSE improvement, CI < 0, no clear LPIPS regression, and beating H. No DNS, no FID.

## 6. Stage commands (resumable)
`python -m boa_dit.quantizer_observer.run <stage>`, in order: inventory, checks, profile, trace_fit, trace_dev, dev_orig, headroom, accounting, errstats, observer_select → commit →
trace_conf (traces may also be generated earlier; summaries stay unopened) → observer_confirm → [live_dev, live_conf] → cost → report. Commit and push after each stage.

---
## Amendments

### 2026-10-30 01:25 UTC: features computed in fp32 (before any trace)
The profile (`profile.json`: 12.3 s per trace step at batch 32, against 1.7 s for plain WA sampling) showed that per-row float64 feature computation dominated.
From now on:
- The R/E/S features are computed in fp32 on whole tensors.
- The accounting identity and the 9×9 Gram stay float64 (first fit batch).
- The identity residual is evaluated on that batch.
The checks are rerun with the changed instrument before the traces: the old checks.json is renamed `checks_v1.json` and kept.

### 2026-10-30 02:00 UTC: incident note (synthetic dry-run outputs)
A CPU dry run of the analysis code on **synthetic random traces** (16/8/12 fake images) wrote headroom/site_accounting/errstats/observer_select/observer JSONs into this folder.
The queue's automatic commit after trace_fit (71c744b) swept them in. They are not results: n = 16 fit "images" of Gaussian noise. They are deleted in the next commit.
Real outputs are produced only by the stage CLI on the trace caches. In future, dry runs write only to the scratchpad.

### 2026-10-30 02:17 UTC: stop after development selection (gates do not justify more GPU)
- The development selection (`observer_select.json`, committed) chose **H**:
  - no added family reached the 1% development-value threshold;
  - all deployable observers left about 100% of the low-band error (ratio vs zero prediction 1.02–1.04);
  - only the non-deployable oracle helped.
- With H selected, the observer gate fails by definition (§5: no added internal features), and the live arm cannot start.
- Following the instruction ("stop earlier if the gates do not justify further GPU work"), the confirmation trace was stopped after about 1.7 min.
  Nothing was saved or opened, and its GPU time is recorded manually as an ABORTED row.
- **Observer confirmation (90200–90263) and live confirmation (90300–90363) remain unopened and unused.**
- The confirmation-set headroom was not computed. The cost and report stages follow.
