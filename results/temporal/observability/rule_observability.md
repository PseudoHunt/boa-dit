# Observability of persistent weight error, DiT heig-8 (consolidated plan §2; pre-registration)

Written 2026-10-28, before any run of this phase. Branch `weights-joint`, results in `results/temporal/observability/`.
Nothing above the marker changes afterwards; deviations are appended below it, dated.

## Budget
- Part of the consolidated **3.0 A30-hour** allocation, stage `c3`, counted across both branches' `gpu_hours.csv`.
- §2 is planned at ≈ 0.4 h. The FID arms in (a) and (c) run only if their gates pass and the remaining budget covers them.

## Setting and traces
- DiT-XL/2 256, **heig-8 W4** (high-precision activations), DDIM-50, cfg 1.5, guided 4 eps channels.
- **New seeds, untouched by any earlier phase:** calibration **7000–7063** (64) and evaluation **7100–7163** (64).
- For each seed, three things are recorded at all 50 steps:
  - the FP trajectory states x^FP_t;
  - the FP ε_t and heig-8 ε̂_t on those FP states (teacher-forced local error Δ_t = ε̂_t − ε_t);
  - the free-running heig-8 trajectory x^q_t, with d_t = x^q_t − x^FP_t the accumulated error.
- **Bands:** the 4 equal-area radial bands of `objective/d2.py`. **Buckets:** 10 buckets of 5 steps.

## (a) Audit
1. **Shared mean vs per-seed residual.**
   - The per-step across-seed mean μ_t of Δ_t (per element) and the centred residual Δ_t − μ_t.
   - The shared energy is estimated **without finite-sample bias**, as the inner product ⟨μ_t^A, μ_t^B⟩ of the means over two
     disjoint halves of the calibration seeds.
   - Reported: absolute energy and fraction per band and bucket, and the cross-step correlation of each part.
   - **If the shared mean is ≥ 10% of the energy** (summed over steps): add a per-step bias-table control arm
     ε̂_t − B_t, with B_t = μ_t from all 64 calibration seeds. It is evaluated by latent rel. MSE on evaluation seeds
     and by FID-2k (seeds 0–1999; base heig-8 27.47).
2. **Distributional-interpretation gate** (added at the user's request, 2026-10-28, before any run). Per band and per
   bucket, using across-seed covariance traces on the evaluation seeds (scalar: the sum of across-seed variances over the
   band's DFT coefficients, both pooled over the 4 channels and the bucket's steps):
   - ΔV = tr Cov(x^q) − tr Cov(x^FP);
   - the cross term X = tr Cov(x^FP, d), with the identity ΔV = tr Cov(d) + 2X;
   - **ρ = 2X / tr Cov(d)**, plus the correlation-normalized X / √(tr Cov(x^FP)·tr Cov(d)).
   - **Read:**
     - if |ρ| < 0.5, the accumulated error mostly **adds marginal variance** (the cross term is small);
     - if |ρ| ≥ 0.5, it mostly **rotates or contracts the signal** (the cross term is comparable to Cov(d)).
   - The **gate passes** if |ρ| < 0.5, evaluated as the tr Cov(d)-weighted mean over buckets, in band 1 **and** in the
     band carrying the most tr Cov(d).
   - The covariance-calibrated correction is **not built** unless this gate passes **together with** Read O. In this
     allocation it is not built in any case: nothing beyond the list is authorized, so a joint pass is reported as the
     condition being met.

## (b) Predictors of the low-band residual
- **Target:** r_t = the centred residual Δ_t − μ_t projected onto the **DCT-16 subspace**: the 4×4 lowest 2-D DCT-II
  modes of the 32×32 grid per channel, giving 64 coefficients per sample. The coefficients are shared over position by
  construction.
- **Models:** ridge regression, one per 5-step bucket, fitted on the calibration seeds only. λ is chosen per bucket by
  4-fold CV over calibration seeds from {1e-3, 1e-2, 1e-1, 1, 10, 100} × mean feature variance. Features are the DCT-16
  coefficients of:
  - **oracle history:** r_{t−1} (the previous true residual);
  - **deployable history:** ε̂_t, ε̂_{t−1}, ε̂_t − ε̂_{t−1}, and the step index t (within bucket);
  - **current-only control:** ε̂_t.
  The first step (t = 0) has no history and is excluded from all three equally.
- **Held-out metrics** (evaluation seeds 7100–7163):
  - residual MSE, as a fraction of the zero-prediction MSE;
  - absolute coherent error removed, C = T‖mean_t r‖² before vs after subtracting r̂ (per seed, summed);
  - error left in the temporal sum, ‖Σ_t (r_t − r̂_t)‖² / ‖Σ_t r_t‖².
  Each comes with a seed-bootstrap CI (2,000 resamples, seed 0).
- **Gate for (c):** deployable held-out MSE ≤ 0.8 × current-only held-out MSE (≥ 20% better), with the bootstrap CI
  of the ratio below 1.

## (c) Live correction (only if the (b) gate passes)
- ε̂_t ← ε̂_t − λ·U·r̂_t, with r̂_t from the deployable model computed on the live (quantized) trajectory and U the
  DCT-16 synthesis.
- λ ∈ {0.25, 0.5, 0.75, 1.0} is fitted on design seeds 7200–7231 by latent rel. MSE and then frozen.
- Rollout on untouched seeds: latent rel. MSE on 7300–7427, and FID-2k on seeds 0–1999, with confirmation on a second
  2k set, seeds 2000–3999 (base and corrected).
- **Read O:** viable iff (c) runs and improves FID-2k by ≥ 0.4 on seeds 0–1999 **and** the sign is confirmed on
  2000–3999. The expected and acceptable outcome is that the oracle works and the deployable predictor does not.
- The drift-aware schedule deformation is **not** built.

---

## Deviations and results
(appended below as they happen)
- 2026-10-28, results (`audit.json`, `predict.json`):
  - (a1) The shared mean is 0.22% of the local-error energy (split-half; naive 1.8%), so the **B_t arm is not
    triggered**.
  - (a2) **The gate fails**: tr Cov(d)-weighted ρ = −2.45 (band 1), −1.08, −0.90, −0.79. The normalized cross-correlation
    is ≤ 0.18 in magnitude; ρ is large because Cov(d) is small next to the cross term.
  - (b) The oracle removes 94% of the coherent low-band residual. Deployable and current-only predict nothing (CV picks
    the largest λ; held-out MSE 1.000). **The (c) gate fails**, so (c) is not run and **Read O = not viable**.
