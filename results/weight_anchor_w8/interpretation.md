# wa8 interpretation (kept narrow)

- **Primary (confirmation, 64 fresh images):** R8 vs A8 latent MSE −78.4% [−83.9, −72.7], 64/64 wins, ΔLPIPS −0.110 [−0.126, −0.096]
  → pre-registered reading **useful reuse**. Measured W4–W8 differences at H8 anchors carry sample-specific correction across the 3 following calls.
- **Anchor adequacy:** H8 is far more accurate than Q4 (−96.8% latent MSE, 63/64), so the anchor is adequate under the pre-registered definition.
- **Retained benefit vs FP anchors:** B = 0.983 [0.966, 0.997]. R8 is nonetheless measurably worse than R_FP (+10.9% [+1.6, +22.1] latent MSE,
  ΔLPIPS +0.004 [+0.001, +0.007]); being close to R_FP is not equivalence.
- **R8 is not close to H8.** Plain H8-50 has latent MSE 0.0020 vs R8 0.0091 (R8 +358% [+143, +886], 2/64 wins; ΔLPIPS +0.033). Any
  practical case for R8 therefore needs R8 to be cheaper than H8-50 by enough to matter, and to win against H8 (or H8 at fewer steps) at matched
  latency. This pilot measured no latency: W4 and W8 are both simulated with fp32-dequantized weights.
- **Persistence (dev traces):** the W4–W8 discrepancy decorrelates with lag (cos 0.95 / 0.89 / 0.75 / 0.50 at lags 1/2/4/8); the offline K = 4
  corrected FP-error ratio is 0.364 and K = 8 0.675, numerically the same as FP-anchor reuse on the same traces (0.361 / 0.675), because H8's own
  error is small relative to the W4 error. The K = 8 dev endpoint (−37.0%) agreed in sign with the frozen prediction — one preliminary check only.
- **Not shown:** speed, memory benefit in a real kernel, generation quality (no FID), any other model, bit-width, or K at confirmation scale.
- **Monday:** the necessary condition is C_Q4/C_H8 < 0.74 for K = 4 (zero overhead), measured on full trajectories, then quality vs H8 at matched latency.
