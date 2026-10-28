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
- 2026-10-28, **user swap (pre-registered here before any run): §5 is replaced by one arm, DCT-16 feedback on PixArt.**
  - Arm: the static-A8 feedback of the DCT-4 arm with the state compressed to the 16 lowest 2-D DCT-II modes (u, v ∈
    {0..3}, orthonormal, 64×64 token grid; DCT-4 is its {0,1}² subset, checked to 0 difference). Same seeds (seed =
    prompt index, batch 1) and **both prompt sets**: screen mjhq1k[0:64] (`s3_single8_feedback_dct16`) and untouched
    mjhq1k[64:96] (`u_single8_feedback_dct16`). Metrics: PSNR, LPIPS-VGG, latent rel. MSE vs the bf16 batch-1
    references, paired bootstrap vs single table (2,000 resamples, seed 0). No ImageReward (not requested).
  - **Read (trade-off), on both sets, mean values:** full < DCT-16 < DCT-4 on latent **and** DCT-4 < DCT-16 < full on
    PSNR **and** DCT-16 LPIPS ≤ single table's. CIs are reported but are not part of the read.
    - If it holds: the method's recommended setting is band-limited feedback with k chosen for LPIPS, and the report
      gives the k-curve (single, k = 4, 16, full) on both sets.
  - Cost: ≈ 33 s/image × 96 ≈ 0.9 h (DCT-4 pilot).
  - **D3 is stopped to make room** (stopped at 128/2000 images, 0.099 h recorded as FAILED, partial kept). Remaining
    after DCT-16 is ≈ 0.95 h against ≈ 1.3 h to finish D3, so **D3 is reported unrun** (cost 1.38 h total) unless the
    measured spend leaves room. §5 runs afterwards with whatever budget is left (rank → η → INT4 → budget → DCT-4 ref,
    each behind the guard).
- 2026-10-28, **user instruction (before the DCT-16 read): §5 is dropped from this allocation; D3 is resumed instead.**
  - Order: DCT-16 finishes as set up → score → D3 resumes from its 128/2000-image checkpoint and completes.
  - Projected end ≈ 3.37 h. **A ≈ 0.3 h overrun of the 3.0 h cap is approved by the user** and is reported as such; the
    queue hard-stops D3 if the projection at launch exceeds 3.45 h.
  - §5 (INT4 state, η, site budget) is **unrun in this allocation**; `mem.py` is written but untested on GPU.
  - If DCT-16 passes its read, the report states that D3 is the **DCT-4** arm, so its FID is a lower bound on what the
    recommended band-limited setting would get on DiT.
- 2026-10-28, **correction (data provenance):** the D2 set mjhq1k[64:96] was untouched by every feedback/temporal arm,
  but these prompts belong to mjhq1k[0:256], which the earlier W4A4 base validation (MJHQ-256: bf16 and
  `hsvd_r4_b0.5`, `results/w4a4/eval/img256_*.json`) generated and reported. D2 is therefore a **second, previously
  unused-for-this-comparison set**, not untouched confirmation data in the strict sense. The results stand as
  reported; the label changes. New PixArt confirmation uses mjhq1k[256:1000], which no experiment has used.
- 2026-10-28, **DCT-16 result (`dist/pixart_scores.json`, `read_dct16`): the trade-off read FAILS on both sets.**
  - k-curve, mean latent rel. MSE / PSNR / LPIPS:
    - screen: single 0.1844 / 18.11 / 0.3778; DCT-4 0.1701 / 18.54 / 0.3633; **DCT-16 0.1793 / 18.29 / 0.3692**;
      full 0.1599 / 18.69 / 0.3933.
    - untouched (see the provenance note): single 0.1921 / 17.66 / 0.3965; DCT-4 0.1761 / 18.15 / 0.3845;
      **DCT-16 0.1851 / 17.89 / 0.3851**; full 0.1671 / 18.32 / 0.4178.
  - DCT-16 does **not** land between DCT-4 and full: it is worse than DCT-4 on latent and PSNR on both sets (between
    single and DCT-4). Its LPIPS stays at or below single's (screen −0.0086 [−0.0144, −0.0023]; untouched −0.0114
    [−0.0251, +0.0065]).
  - The k-curve is not monotone in k between 4 and full. The recommended-setting clause does not apply, so the D3
    lower-bound note is not needed.
