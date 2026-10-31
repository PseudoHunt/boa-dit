# w8s1 — W8 anchors every two calls: report

Protocol: `PROTOCOL.md` (pre-registered 2026-10-31; amendments dated). Spend **0.276 of 0.4 A30-h**: checks 0.050, the failed run attempt 0.135
(trajectories saved, timing crash), the rerun 0.092. **Complete.** 64 fresh seeds 126000–126063, batch 32.

## Observed
Endpoint latent rel. MSE vs paired FP50:

| arm | anchor calls | corrections | mean | median | PSNR | LPIPS |
|---|---|---|---|---|---|---|
| H8_50 (W8 every step; its own error) | 50 | 0 | 0.00226 | 0.00038 | 41.20 | 0.0114 |
| FP-K4 (abn1 CB) | 13 | 12 | 0.00698 | 0.00145 | 35.49 | 0.0317 |
| W8-K4 (abn1 CB.w8) | 13 | 12 | 0.00870 | 0.00144 | 34.96 | 0.0334 |
| **W8-K2** | 25 | 24 | **0.00238** | 0.00065 | 39.23 | 0.0153 |

Every anchored arm also makes 50 cheap (W4A8 static, original branch) calls.

| contrast | ratio [95% CI] | wins | ΔLPIPS [95% CI] | reading |
|---|---|---|---|---|
| **W8-K2 / FP-K4 (primary)** | **0.341 [0.142, 0.802]** | 52/64 | −0.0164 [−0.0265, −0.0074] | useful |
| W8-K2 / W8-K4 (spacing isolated) | 0.273 [0.101, 0.761] | 58/64 | −0.0181 [−0.0295, −0.0086] | useful |
| W8-K4 / FP-K4 | 1.247 [0.662, 2.048] | 27/64 | +0.0017 [−0.0062, +0.0096] | inconclusive |
| W8-K2 / H8_50 | 1.054 [0.410, 2.564] | 15/64 | +0.0039 [−0.0022, +0.0111] | (descriptive) |
| FP-K4 / H8_50 | 3.094 [1.778, 5.581] | 7/64 | +0.0203 | (descriptive) |

- **W8 error is not negligible**: H8_50's own error vs FP50 is 0.00226. At K2 the remaining error is close to that floor: W8-K2 is within an
  inconclusive 1.05× of H8_50, with H8_50 better on most seeds (W8-K2 better on only 15/64).
- Halving the anchor spacing cut the error about 3.7× (W8-K4 → W8-K2). This is measured, not predicted: the remaining error mixes stale cheap error
  and anchor-model error, and both change with spacing.
- **Checks:** K2 indices 0, 2, …, 48 (25 anchors, 24 corrections, coefficient ½·b_{a+1}); K1 with W8 anchors reproduces H8_50 bitwise with 0
  corrections; abn1 CB and CB.w8 reproduce bitwise.

## Cost accounting (simulation only; no efficiency claim)
- Calls per image: K2 = 50 cheap + 25 W8; FP-K4 = 50 cheap + 13 FP.
- Ignoring overhead, equal anchor cost requires C_W8/C_FP ≤ 13/25 = 0.52.
- Measured fake-quant per-call times (batch 32, CFG 64 rows, CUDA events, median of 20 interleaved calls): FP 389 ms, H8 456 ms (dequantized W8 in
  fp16 compute), cheap 1055 ms (fake-quant W4A8 with Python hooks). Simulated C_W8/C_FP = **1.17**.
- Simulated whole trajectories per 32 images: FP-K4 58.0 s, W8-K4 58.9 s, W8-K2 64.3 s, FP50 19.6 s.
- None of this is kernel cost. A real W8 kernel would need to run at ≤ ~0.52× the FP call cost for K2 to match FP-K4 anchor cost. Untested.

## Not established
Real-kernel W8 or W4A8 cost, generation quality, other spacings (not a K search), other models.
