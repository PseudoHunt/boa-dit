# Answers to the handoff's five end-state questions (2026-10-31)

**1. Is BA's 5.5× raw-error increase real on common states, and does absolute corrected error improve?**
Yes, it is real. "5.5×" is squared-L2 energy of the per-step guided prediction error on common FP50 teacher-forced states (5.51×; 2.35× in relative
L2). On 16 common B0-trajectory states it is 5.1× vs B0 and 5.5× vs BL@330.
- **What the extra energy is:** about 77% of BA's raw error is a per-image, near-time-constant offset (vs about 19% for B0 and the local refits).
  Anchor reuse removes it.
- **What is left is smaller and better captured:** BA's interval deviations are 0.88× B0's, its held residual 0.90× B0's / 0.94× BL's, and its bridge
  residual 0.82×.
- **Absolute corrected error improves on live short-window replay from common states:** post-kick error 0.80× B0 and 0.83× BL@330 (16-image bootstrap
  CIs below 1). This holds in early, middle and late intervals, and is 0.86× for two-interval windows.
- So error shaping is supported for corrected windows.
- **Scope:** BA is hybrid-only. As a plain cheap sampler it is catastrophic (rel. MSE 1.36 vs B0's 0.137).

**2. Does the improvement repeat across the two new BA/BL training pairs?**
Only in direction, not decisively.
- On 256 fresh images, BA-sel/BL-sel = **0.949 [0.710, 1.288]** (seed 1) and **0.946 [0.757, 1.203]** (seed 2).
- Aggregate **0.947 [0.786, 1.150]** → **inconclusive** under the predeclared rule.
- LPIPS is lower for BA in both pairs (−0.0031 [−0.0064, −0.0000]; −0.0009 [−0.0040, +0.0025]) and PSNR is higher (+0.24, +0.41 dB). No LPIPS
  regression.
- BA does **not** beat the archived strong local refit (0.962 [0.733, 1.282]) or the original branch B0 (0.985 [0.745, 1.341]).
- The pilot's BA/B0 0.665 did not recur: the pilot BA on this cohort is 0.978 vs B0.

**3. What uncertainty remains over images and over training seeds?**
- **Over images:** conditional on these trained models, the aggregate CI spans about −21% to +15%. Per-pair CIs are about ±25–30%.
- **Heavy tails:** the largest image carries 11–14% of each BA arm's total error. The worst image (128228) is shared across BA and BL arms.
- N = 256 had about 20% projected power for a 10% effect, so a modest real gain (≈5%, as observed) is neither confirmed nor excluded.
- **Over training seeds:** three seeds give 0.949, 0.946 and 0.808 (pilot). The direction is consistent, but magnitudes differ, and the dev rule
  selected different checkpoints (BA@82 vs @330).
- The exploratory nested bootstrap over seeds and images, [0.730, 1.117], is fragile with three seeds. Population-level training-seed robustness is
  not established.

**4. Is there a supported packed implementation capable of beating W8-only, or only a hypothetical cost ratio?**
Only a hypothetical cost ratio. No installed or official prebuilt backend implements either side on the A30 (sm_80) as packed kernels:
- the cheap side is W4A8-static + rank-8;
- the anchor side is W8A16 + rank-8.
- PyTorch's W8A16 op is a naive kernel, 158–306× slower than fp16.
- INT8×INT8 `_int_mm` is packed but A8, a different model; with glue it is 1.65–1.84× fp16.
- tinygemm W4A16 is 4.9–9.1× slower than fp16 at these M.
- Nunchaku is W4A4 and model-specific.
- **Criteria:** beating W8-only needs C_Q/C_W8 < 0.5 − ΔO/(50 C_W8). Separately, K2-W8 vs K4-FP anchors needs C_W8/C_FP ≤ 13/25.
- No full-loop timing was possible; no speed claim. See `results/w8_kernel/REPORT.md` on branch `w8-kernel-feasibility`.

**5. What ran, what remains unopened, total spend, is any job running?**
- **What ran:**
  - 1A audit (CPU; no bug);
  - 1B frozen N = 256;
  - mechanism (0.256 h);
  - repchecks (0.043 h);
  - two training pairs + dev (≈0.90 h each);
  - fresh evaluation of 11 arms × 256 (1.500 h; one failed 0.011 h attempt with no outputs);
  - kernel inventory + partial microbenchmark (0.034 h).
- **Total spend: 3.593 of 4.0 A30-h**, failures included.
- **Unopened:** 128256–128511 (this protocol's reserve); cfb1 121000–121255; abn1 FID cohort 112000–113999; SANA mjhq1k 464–527; qo1 90200–90363;
  PixArt mjhq1k 288–351 / 400–463.
- **Nothing is running; the GPU is idle.**
- Not launched, per the handoff: FID, a new model, another optimizer/loss, a new anchor schedule, a larger N.
