## Interpretation (written 2026-10-30, after the results; protocol §6 rules)

**What ran.** Every stage ran to completion within budget: 1.26 of 4.0 GPU-h (the audit was CPU-only). The response-calibrated R arm was **blocked by its registered approximation check in both experiments**:
- The central-difference endpoint columns at h = 0.1 and h = 0.05 differ by 60–290%, and at h = 0.2 vs 0.1 by 53–110%. The tolerance was 25%.
- The mixed (h, −h, h, −h) superposition test misses by 66–101%.

So the full-sampler endpoint is not locally linear in these coefficients at a usable step size. As the protocol requires, R was not fitted or tuned, and the R contrasts (R vs S, R vs L, R vs B, R vs A) do not exist. **The prioritization signal cannot be evaluated.**
- Limitation of the check: the registered "above noise" condition used same-configuration repeat noise. That noise is exactly 0, because the sampler is bitwise deterministic, so the condition passed trivially.
- Context for scale: changing only the batch size moves the endpoint by 1.1% relative, and the L candidates (|a| ≤ 0.03) move development MSE by about ±1.5%. Small carry perturbations evidently push the fp16 W4A8 trajectory onto slightly different paths.
- A future response method needs a perturbation model that tolerates this. Options: averaging over several h, a smoothed or finite-step secant, or a lower-variance target. That would be a new protocol, not a retune of this one.

**L (local linear response) is near-identity in both experiments.**
- At the baseline, the four-direction span explains only 0.19% (E1) and 0.13% (E2) of the normalized local error, so the fitted coefficients are |a| ≤ 0.03.
- E1 L vs B: −1.07% [−1.66, −0.58] latent and a small LPIPS gain. That is real but tiny, about the fp16 sensitivity floor seen on development.
- E2 L vs B: −0.28% [−0.73, +0.19], not significant.
- The "predicted change" column for S and L is in normalized *local-objective* units. Its large positive values for a = 1 are quadratic extrapolations far outside the linear region, not endpoint predictions.

**E1 (calibrating the existing C_L u200 Adam update).**
- Scalar selection on development chose α = 1, i.e. A itself.
- On confirmation, A/S vs B is −22.7% [−29.0, −16.6] latent, +1.37 dB PSNR and −0.050 LPIPS, with 111/128 wins. This replicates the joint pilot's carry-refit gain on fresh seeds.
- Shrinking or calibrating the update per depth group gave no benefit on development: S at 0.5 had 0.102 vs 0.087 for α = 1, and L ≈ B.
- E1 therefore shows that the existing trained update is best used unscaled in this four-group family. It does not demonstrate any gain from response calibration.
- C_L's training cost (0.237 GPU-h) is separate from this calibration.

**E2 (analytic ridge directions, no Adam).**
- The local fixed-feature solve worked as designed: the regularized surrogate never increased, the fp16 export matches it to about 1e-6, and the maximum condition number is 2.9e3.
- The local headroom is concentrated in one site. Blocks 21–27 reduce the local surrogate by 69% on fit and 65% on dev, almost all of it from b27.mlp.fc2 (81% of all local error energy, 83% reduced; ΔD ≈ 14.5× ‖D0‖). Blocks 0–20 gain only 2.5–4% on fit and about 0–1% on dev.
- At the endpoint, the full analytic update (A_analytic) looked good on development (−9.5%; S chose α = 1) and modestly good on fit (−5%).
- **It did not hold up on confirmation:** A vs B is +3.9% [−2.4, +12.9] latent (n.s.), with **LPIPS worse by +0.020 [+0.010, +0.033]** and 62/128 wins.
- The selected E2 correction is therefore **not** better than the original carry. By the protocol rule ("if all methods worsen B, keep B"), **B is kept.**
- A large local-surrogate reduction at one late site did not translate into consistent endpoint fidelity. Development (n = 32) overstated the gain; this is the selection variance that the sealed set exists to catch.

**What this does and does not establish.**
- It does not test whether trajectory-response calibration helps, because the response approximation failed before any fit.
- It does not show that analytic or rank-8 corrections fail in general. It shows that one fixed-V, per-site ridge construction with a four-group scalar family does not reliably improve endpoint fidelity on this bed.
- E1 and E2 used different seeds and are not compared head-to-head.
- No-Adam vs Adam remains untested at matched data and cost.

Possible follow-ups, not run, each needing its own protocol:
- Exclude or separately handle b27.mlp.fc2, the outlier site.
- A ridge solve on student-trajectory features with a held-out-validated λ.
- A robust secant or multi-h response estimate.
- Larger development sets for selection.
