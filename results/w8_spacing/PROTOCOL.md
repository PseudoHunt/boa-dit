# w8s1 — W8 anchors every two calls: an accuracy / anchor-spacing probe — pre-registration

Written 2026-10-31, **before any w8s1 measurement**. Specification: the user's handoff (2026-10-31), task 3. Branch `dit-anchor-refit` (shared with
dar1). Code `boa_dit/anchor_refit/w8.py`, using the abn1 harness unchanged. Tag **w8s1**; planning cap **0.4 A30-h** within the handoff total of 4.0.

## Configuration
The **original** common-base branch (b4a537b0…; not any dar1 checkpoint), W4A8 static cheap model, bridge λ = 1, DDIM-50, CFG 1.5, fp16 autocast. This
is the abn1 configuration.

The anchor model is the validated wa8 **H8** (`weight_anchor.w8.h8_model`):
- the heig-8 W4 state with its 112 block linears replaced by dequantized fixed-calibration W8 codes;
- its own carry δ8;
- A16 activations (fake-quantized weights in fp16 compute);
- h8.pt c574916e…, heig-8 W4 state facee154….

## Arms (abn1 `sample`)
| arm | anchors | anchor calls | corrections |
|---|---|---|---|
| FP-K4 (= abn1 CB) | FP at 0, 4, …, 48 | 13 | 12 |
| W8-K4 (= abn1 CB.w8) | H8 at 0, 4, …, 48 | 13 | 12 |
| **W8-K2** | H8 at 0, 2, …, 48 | 25 | 24 |

- Each arm makes 50 cheap calls. The final interval after 48 (step 49) is uncorrected in all arms.
- References: FP50 (fidelity target) and **H8_50** (W8's own error vs FP50, to check whether W8 error is negligible rather than assume it).

## Data
**126000–126063** (64 fresh seeds; audited unused; reserved now), batch 32, paired noise and FP50. If profiling shows this does not fit 0.4 h, a smaller
size is frozen before any quality result.

## Checks (before quality runs)
- K2 indices: 25 anchors with 24 corrections; bridge endpoints and coefficients for k = 2 (= ½·b_{a+1}).
- K1 limit with W8 anchors reproduces H8_50 bitwise, with zero corrections. The bridge coefficient vanishes for k = 1.
- Reproduce abn1 CB and CB.w8 bitwise on 110000–110031 (batch 32).
- Call counts are asserted per trajectory.

## Reading
- Primary: W8-K2 / FP-K4. Spacing isolated: W8-K2 / W8-K4. Also W8-K4 / FP-K4 and each arm vs H8_50.
- Paired ratio of mean endpoint latent rel. MSE (bootstrap 10,000, seed 0), LPIPS, PSNR, absolute errors, wins and call counts.
- This is a direct spacing probe, not a K search. The remaining error after correction combines the stale-error and anchor-model-error terms; no
  random-walk gain is predicted.
- **Cost:** reported as fake-quant simulation latency only. Equal anchor-call cost at K2 vs FP-K4 needs C_W8/C_FP ≤ 13/25 ≈ 0.52, ignoring overheads.
  The measured simulated ratio is not a kernel cost. No efficiency claim; a real-kernel W8 test is separate.

---

## Amendments and results
(appended below, dated)
