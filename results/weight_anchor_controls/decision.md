## Decision memo (written 2026-10-30, after the confirmation)

**Outcome rows of the pre-registered decision table that apply:**

| outcome | holds? | evidence (fresh confirmation 93000–93063, 64 images, vs FP50) |
|---|---|---|
| R/R1 clearly beat S | **yes** | R vs S: −75.0% [−83.8, −65.2] latent rel. MSE, 62/64 wins, ΔLPIPS −0.133 [−0.151, −0.116]. R1 vs S: −75.5% [−83.6, −66.6], 62/64 wins, ΔLPIPS −0.135 |
| S matches or beats R with adequate precision | no | — |
| R beats A but not S | no (R beats both) | R vs A −74.0% [−81.3, −66.6], 64/64 wins (replicates wa1's −69%) |
| FP-50 is faster than R | **yes** | complete trajectories, warm, 5 repetitions: batch 32 FP-50 19.28 s vs R 27.64 s (median; ranges ±0.03 s); batch 1 FP-50 1.44 s vs R 2.34 s |
| R beats a latency-matched FP sampler | not applicable | FP-50 is faster than R and is the reference, so no FP-k was built (as pre-registered) |

**Interpretation, kept narrow:**
- **The W4 prediction changes carry useful information beyond the stale FP output.**
  - Holding the last FP prediction (S) is no better than ordinary W4 between anchors (S vs A +3.9% [−15.7, +34.7], inconclusive on latent, worse on LPIPS).
  - Adding the W4 increments to the FP anchor (R1) cuts the remaining endpoint error by three quarters.
  - The fitted β adds nothing measurable over β = 1 (R vs R1 +2.2% [−4.8, +7.5], inconclusive). The construction ε_FP,a + (ε_W4,t − ε_W4,a) is what matters.
- **There is no measured efficiency advantage for this W4A16 implementation.**
  - In this harness R is 1.43× slower than FP-50 at batch 32 and 1.62× slower at batch 1.
  - It needs both models resident (5.40 GB in fp32 here, against 2.70 GB for FP alone).
  - FP-50 has zero discrepancy from itself.
  - So R has no advantage over FP-50 in the measured latency–memory–fidelity comparison.
  - This is about the implementation: the simulated W4 forward has no low-bit kernels and is itself slower than FP (Q 22.6 s vs FP-50 19.3 s).
  - It does not bear on whether the error persists over time, which wa1 and these controls both confirm.
- **The 13-call comparators are close to each other and far from R.**
  - S vs FP13: −4.0% [−8.8, +1.3] latent (inconclusive), with LPIPS slightly better for S (−0.007 [−0.014, −0.002]).
  - At the same 13 FP calls, the stale-hold sampler and a proper 13-step sampler land near each other, both well short of R.
- **Successful error reuse is not an efficient quantization method.** R uses more compute than FP-50 here.

**What remains untested (not run; proposed only):**
- **A sparser anchor schedule (K = 8)**, as a separately dated follow-up with its own fresh confirmation set. It would cut R to 7 FP calls plus 50 W4 calls.
  Even so, it can only beat FP-50 on latency if a W4 call costs much less than an FP call, which requires genuine low-bit kernels. This harness cannot measure that.
- **A real low-precision implementation** (packed W4 kernels) would be needed before any latency claim. Production-kernel timing and generation-quality validation (FID or similar) come after that.
- **This construction elsewhere:** no other models or activation-quantized beds were tested.

No further runs were made with the remaining budget, as instructed.
