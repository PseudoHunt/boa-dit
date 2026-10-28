# Sampler-side correction (Q-Drift) on heig-8 W4, and the conditional-residual temporal-covariance diagnostic (protocol)

Written 2026-10-28, before any run of this phase. Branch `weights-joint`, results in `results/temporal/sampler_corr/`.
Nothing above the marker changes afterwards; deviations are appended below it, dated.

## Budget
- **2.0 A30-hour hard stop,** stage `sampler_corr` (set by the assistant; the user asked not to be asked).
- Jobs are pilot-timed and unfinished arms reported as unrun.

## Setting
- **Model:** DiT-XL/2 256, **heig-8 W4** (W4 block linears + rank-8 carry, W8 adaLN), high-precision activations (fp16
  autocast).
- **Sampler:** DDIM-50 (η = 0), cfg 1.5 on the first 3 eps channels (4th from the cond half), the repo's eval protocol.
- **FP references:** stored and regenerated FP latents; the stored FP FID-2k is 23.89 (the same protocol).

## Q-Drift (arXiv 2603.18095, Eq. 14–15), as applied here
- **Update:** Euler in σ-coordinates, x_{i+1} = x_i + Δσ_i (1 + c_i) ε̂, with c_i = (|Δσ_i| / (2σ_i))·V_{σ_i}, where
  V_σ = E[Var(Δε | ε̂)] is a scalar per step. The derivation is for first-order deterministic Euler.
- **Mapping to DDIM:** with y = x/√ᾱ and σ = √((1 − ᾱ)/ᾱ), deterministic DDIM is exactly y_{i+1} = y_i + (σ_{i+1} − σ_i)·ε
  (linear in ε). Multiplying the guided ε (all 4 eps channels) by (1 + c_i) in the sampler's ε hook therefore
  implements Q-Drift's Euler update exactly. The learned-variance channels are unused at η = 0.
- **Check before use:** an explicit Euler-in-σ sampler reproduces the DDIM endpoint (fp32, relative difference
  < 1e-5) with and without the (1 + c_i) factor.
- **Calibration:**
  - Paired FP ε and heig-8 ε̂ on the **FP trajectory** states, all 50 steps, guided 4 eps channels.
  - Calibration seeds **6000–6063** (64; the paper reports K = 10 paired runs sufficient).
  - Per step, the isotropic joint-Gaussian conditional variance, pooled over seeds and elements:
    V_i = Var(Δ) − Cov(Δ, ε̂)² / Var(ε̂), with Δ = ε̂ − ε.
  - One scalar per step, as in the paper's main method. There are no other tunable parameters.
- **Arms:**
  - (1) heig-8, unmodified sampler;
  - (2) heig-8 + Q-Drift;
  - (FP reference).
- **Primary metric: FID-2k** (seeds 0–1999, eval_fid protocol, ADM reference batch), because Q-Drift targets the
  sampling distribution, not same-seed fidelity.
- **Secondary:** endpoint latent rel. MSE vs FP on eval seeds 0–511 (same-seed fidelity; reported, **not** the design
  objective) and per-step prediction-error energy, C and C/I.
- **Read:** Q-Drift helps iff FID-2k(2) < FID-2k(1). A single FID-2k per arm has no CI; the difference is reported
  with that caveat. If the latent MSE moves the other way, that is reported as the fidelity–distribution trade-off the
  paper itself shows.

## DNS (Deformable Noise Scheduler, PMLR v306)
**Not reproduced** in this phase: only its abstract was read (the method is in its paper and code,
github.com/ZephyrYoung-eYuan/DNS_AQE). It is listed as unrun.

## Conditional-residual temporal-covariance diagnostic
- **Question:** after removing what is predictable from the current quantized prediction (the statistic Q-Drift
  conditions on), does the remaining error stay correlated across steps?
- **Residual:** Δ_t = ε̂_t − ε_t on FP-trajectory states, and ξ_t = Δ_t − Ê[Δ_t | ε̂_t], with Ê the per-step scalar linear
  (joint-Gaussian) regression fitted on calibration seeds 6000–6063.
- **Evaluation:** on **held-out seeds 6064–6095** (32). For the raw Δ and the residual ξ, per radial band (the 4
  equal-area bands of `objective/d2.py`) and for the spatial mean (DC):
  - the cross-step correlation corr(ξ_t, ξ_s) pooled over seeds and band elements, at lags 1, 2, 5, 10 (averaged
    over t);
  - the coherent fraction ‖Σ_t ξ_t‖² / (T Σ_t ‖ξ_t‖²) (≈ 1/T for independent steps);
  - the share of residual energy per band.
- **Endpoint relevance:** the linear-response weighting via Stage-1-style injection is **not** run in this phase.
- **Read:** temporal dependence remains iff the lag-1 correlation of ξ in band 1 is ≥ 0.3 **and** its coherent
  fraction is ≥ 3/T. This is the stated basis for a covariance-aware correction, which is a research direction and not
  run here.

## Order
1. DDIM ↔ Euler-σ check.
2. Calibration + diagnostic (one pass).
3. Q-Drift latent eval.
4. FID-2k baseline.
5. FID-2k Q-Drift.
Commit and push after each step.

---

## Deviations and results
(appended below as they happen)
