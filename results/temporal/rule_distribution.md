# Distributional metrics for the activation method, and memory engineering (consolidated §3 and §5; pre-registration)

Written 2026-10-28, before any §3/§5 run. Branch `temporal-oracles`, results in `results/temporal/dist/`.
Nothing above the marker changes afterwards; deviations are appended below it, dated.

## Budget and the cost problem
- Shared **3.0 A30-hour** allocation, stage `c3`. Spent before §3: §1 0.21 h, §2 ≈ 0.2 h. §4 is skipped (Read S failed).
  ≈ 0.3 h is reserved for §5.
- **§3 as written does not fit.** Measured costs on this A30:
  - DiT FID-2k: 0.67 h (single table), 1.31 h (full feedback, instrumented simulation).
  - PixArt 1024² at batch 1: 10–26 s per image per arm.
  - A second DiT 2k set for two arms costs ≈ 2.0 h. PixArt MJHQ-256 for four arms plus references is ≈ 4–5 h, and a
    PixArt FID-2k is ≈ 2,000 images per arm.
- **The items therefore run in this order** (information per hour), each only if the guard projects it inside the cap
  with the §5 reserve kept. Anything that does not fit is reported as **unrun with its cost**, never shrunk.
  - **D1.** ImageReward (v1.0) on the existing PixArt MJHQ-64 images (mjhq1k[0:64]) of the four arms (single table;
    best stateless = uniform dither; full feedback; DCT-4 feedback) and the bf16 batch-1 references. Scoring only.
    Per-prompt paired bootstrap vs single table (2,000 resamples, seed 0).
  - **D2. LPIPS settlement and a second PixArt set,** on **untouched prompts mjhq1k[64:96]** (32; no PixArt arm has
    used them). Arms: single table, full feedback, DCT-4 feedback, plus bf16 batch-1 references, all in one run.
    - Metrics: PSNR, LPIPS-VGG, latent rel. MSE vs the references, and ImageReward, each paired vs single table.
    - This set is also the second set for the PixArt ImageReward sign.
    - Uniform dither is not re-run here (the LPIPS question is feedback vs DCT-4 vs single).
  - **D3.** DiT FID-2k for **DCT-4 feedback** on seeds 0–1999 (completes feedback vs compressed feedback on set 1).
  - **D4.** DiT FID-2k for the best stateless arm (uniform dither), seeds 0–1999.
  - **D5.** DiT second-set confirmation (seeds 2000–3999) of single table vs full feedback.
- **PixArt FID-2k (MJHQ) is unrun by cost** in this allocation.
- **Deployed cost** is reported separately from simulation time:
  - persistent state bytes per image per model, from the allocated state tensors (full feedback: fp16 per quantized
    input element per CFG row; DCT-4: 4 coefficients per channel);
  - per element per step, one add before and one subtract after rounding;
  - the DCT-4 projection and synthesis cost in FLOPs per site.

## Read D (as instructed)
- "Better generation" iff full or DCT-4 feedback improves FID-2k on **both** beds **and** ImageReward on PixArt, with the
  sign confirmed on a second set.
- With PixArt FID unrun by cost, Read D **cannot pass** in this allocation. It is reported per component:
  - DiT FID set 1 (and set 2 if D5 runs);
  - PixArt ImageReward on two sets (D1, D2), sign agreement required.
- **LPIPS settlement read:** on mjhq1k[64:96], report whether full feedback's LPIPS is worse than single table (paired
  CI excluding 0) and whether DCT-4's is better. The screen-set pattern (feedback worse, DCT-4 better) counts as
  replicated iff both signs repeat.

## §5 Memory engineering (DiT, latent only)
- **Setting:** DiT static-A8 + feedback bed of Stage 3 (smoothed fc1/fc2; seeds and protocol as arm 5), evaluated on
  eval seeds 0–127 (latent rel. MSE vs FP, paired CI vs the single table and vs full fp16 feedback on the same seeds).
- **INT4 residual storage:** the fed-back residual is bounded to ±Δ/2. Store it as a signed 4-bit code in units of Δ/16:
  s = round(r/(Δ/16)) clamped to [−8, 7], read back as s·Δ/16.
  - Re-measure η (the feedback identity's residual-to-state energy ratio, as defined in `stage3/implementation.md`),
    computed on the same diagnostic seeds as the fp16 state.
  - State bytes: 0.5 per element.
- **Site budgeting:** full fp16 state on the top-k sites by coherent contribution, DCT-4 state elsewhere.
  - Ranking: the measured per-site coherent error C of the single table, from the Stage-3 diagnostics.
  - One budget: k chosen so the full-state sites hold ≤ 25% of the full-feedback state bytes.
  - Reported: state bytes and latent.

---

## Deviations and results
(appended below as they happen)
- 2026-10-28, operational: the D2 queue's wait (`pgrep -f` on the generator) matched its own launcher shell and stalled
  after generation finished; GPU idle, no budget spent. Relaunched with the wait removed (§3) and a PID wait (§5).
- 2026-10-28, D1/D2 results (`dist/pixart_scores.json`; paired bootstrap vs single table, 95% CI):
  - **LPIPS settlement: replicated.** On untouched mjhq1k[64:96], full feedback LPIPS +0.0213 [+0.0076, +0.0373] (worse,
    CI excludes 0); DCT-4 −0.0120 [−0.0231, +0.0008] (better sign; CI touches 0). Both signs repeat the screen set
    (+0.0155 / −0.0145), so the pattern counts as replicated under the pre-registered sign rule; DCT-4's CI alone is not
    significant on 32 prompts.
  - PSNR and latent replicate as well: full feedback +0.66 dB, latent −13.0%; DCT-4 +0.49 dB, latent −8.3% [−17.3, +1.5].
  - **ImageReward sign: not confirmed.** Screen: full +0.023 [−0.016, +0.066], DCT-4 +0.002; untouched: full −0.045
    [−0.107, +0.009], DCT-4 −0.042 [−0.103, +0.017]. Signs flip between sets and every CI includes 0. Quantized arms score
    at or above the bf16 reference on both sets, so ImageReward does not resolve differences at this scale.
  - The PixArt ImageReward component of Read D therefore fails independently of the unrun PixArt FID.
