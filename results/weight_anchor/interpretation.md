## Interpretation (written 2026-10-30, after the confirmation)

**What was shown, on this bed only (heig-8 W4A16, DDIM-50, cfg 1.5, batch 32):**
- **The error persists.** The weight-induced guided-ε error persists over a few denoiser calls.
  - Calibrated reuse of a measured error leaves 14% of the next call's error energy, 32% at lag 2, 63% at lag 4 and 89% at lag 8.
  - The same holds on heig-8's own (uncorrected) trajectory.
  - Swapping in another image's error makes things worse (energy remaining > 1), so the reused information is image-specific, not a shared offset.
- **The persistence survives on live corrected trajectories.** With the measurement refreshed by a real FP call every 4 calls on the live latent,
  reuse lowered endpoint latent MSE vs the FP endpoint by **69% relative to FP anchors alone** on 64 fresh images [−79, −60], with 64/64 wins.
  LPIPS fell by 0.10 and PSNR rose by 6.9 dB. The pre-registered "promising" criterion is met.
- **Anchors alone also help.** FP at 13 of 50 calls without reuse already improves on plain W4 (−32% latent MSE). The main contrast above isolates the reuse.

**What the reuse is:**
- With β ≈ 1, ε_corrected(x_i) = ε_FP(x_a) + [ε_W4(x_i) − ε_W4(x_a)].
  It is the FP prediction at the last anchor, advanced by the W4 model's *change* in prediction since then.
- The W4 model is used as a difference engine between FP evaluations. Its own systematic error cancels to the extent that it is the same at x_a and x_i.
- This is a hybrid FP/W4 sampler, **not an improvement to the quantized model.**
- The scalar β was fitted on FP-trajectory traces. The fitted values are close to 1 except late in sampling, and none needed clipping.

**What was not shown (required before any method claim; deferred by protocol):**
- **Cost.** Reuse needs 50 W4 + 13 FP calls per trajectory, with both models resident.
  - In this harness, simulated W4 is *slower* than FP, so the reuse arm took longer than plain FP sampling (55 s vs 39 s per 64 images).
  - It also needs more memory than FP alone (≈ 20 bits/weight for FP16 + packed W4 before overhead).
  - Any speed benefit depends on real low-bit kernels making a W4 call much cheaper than an FP call. That is not measured here.
- **Controls.** The cost-matched controls are untested:
  - FP sampling at equal latency (e.g. FP with fewer DDIM steps);
  - higher-precision weights;
  - extra W4 steps;
  - **FP anchors with stale FP reuse and no W4 increments**, a feature-caching-style baseline that isolates the W4 increment's contribution.
- **Scope.**
  - One interval (K = 4) and one multiplier (s = 1) were selected on development data. K = 8 was not run live.
  - One weight-only model; no activation quantization, other models, FID, or distributional metrics.
  - The cached lag analysis used previously inspected seeds and is exploratory. The confirmation set (92100–92163) was fresh and opened once.

**Narrow conclusion:** occasional FP measurements identify weight-induced prediction error that stays useful for correcting the next few W4 steps
on this bed. The effect on endpoints is large relative to anchors alone. Whether this is a useful *method* depends on the deferred
cost-matched controls, which this pilot does not address.
