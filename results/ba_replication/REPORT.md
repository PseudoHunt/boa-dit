# bar1 — audit and replication of anchor-aware branch calibration: report

Protocol: `PROTOCOL.md`, pre-registered 2026-10-31 14:06 UTC; dated amendments keep every failure. Kernel track: branch `w8-kernel-feasibility`,
`results/w8_kernel/REPORT.md`. Common ledger: `scripts/h1031b_budget.py`.

__STATUS__

## 1A. Audit of the existing dar1 result (CPU; `audit_1a.json`, `audit_1a.png`)
**No implementation or data bug**, by the predeclared criteria:
- seeds 124000–124127 are in batch order;
- all arms are finite, with shape 128×4×32×32;
- per-image rel. MSE is reproduced from the stored latents (max rel. diff 2.4e-7);
- call counts are 50 + 13 / 12, and 50 for FP50;
- checkpoint hashes match the manifest;
- stored ratios and CIs are reproduced exactly;
- re-decoded PSNR / LPIPS for 8 images per arm match the stored values within 1.4e-4 dB / 2.5e-6.

**"5.5×"** is the **squared-L2 energy** of the per-step 4-channel guided prediction error, Σ_t ‖Q(x_t) − F(x_t)‖², summed over 50 steps and 8 images
(60208–60215), on **FP50 teacher-forced states, which are common across arms**.
- BA 16,468 vs B0 2,987 = 5.51×. As a relative-L2 ratio this is 2.35×.
- On live trajectories: 5.51×.
- Replicated below on 16 images and B0 corrected-trajectory states: 5.11× vs B0, 5.46× vs BL@330.

| arm (124000–124127) | mean rel. MSE | median | PSNR | LPIPS | share of total error from largest 1 / 5 / 10 images |
|---|---|---|---|---|---|
| B0 | 0.00545 | 0.00140 | 36.08 | 0.0298 | 10% / 33% / 50% |
| BA@330 | 0.00362 | 0.00099 | 37.62 | 0.0201 | **28%** / 49% / 61% |
| BL@330 | 0.00434 | 0.00153 | 36.31 | 0.0264 | 9% / 32% / 48% |
| BL-tm | 0.00427 | 0.00113 | 36.86 | 0.0247 | 13% / 46% / 59% |
| BL-arch | 0.00393 | 0.00121 | 36.94 | 0.0241 | 10% / 41% / 58% |

| contrast | ratio of means [95% CI] | mean paired diff [95% CI] | median diff | wins | ΔLPIPS [CI] | ΔPSNR [CI] | LOO ratio range (diagnostic) |
|---|---|---|---|---|---|---|---|
| **BA@330 / BL@330 (pilot primary)** | 0.835 [0.489, 1.423] | −7.2e-4 [−2.6e-3, +1.7e-3] | −1.6e-4 | 79/128 | −0.0063 [−0.0118, −0.0005] | +1.31 | 0.620–0.891 |
| BA@330 / B0 | 0.665 [0.420, 0.964] | −1.8e-3 [−3.4e-3, −1.9e-4] | −3.3e-4 | 91/128 | −0.0097 | +1.54 | 0.535–0.700 |
| BA@330 / BL-tm | 0.848 [0.525, 1.234] | −6.5e-4 [−2.3e-3, +9.5e-4] | −1.0e-4 | 79/128 | −0.0046 | +0.75 | 0.709–0.945 |
| BA@330 / BL-arch | 0.923 [0.560, 1.394] | −3.0e-4 [−2.0e-3, +1.5e-3] | −7.1e-5 | 79/128 | −0.0040 | +0.68 | 0.734–1.015 |

- **Heavy tail.** One image, 124092, scores BA 0.128, BL@330 0.014, B0 0.070.
  - It is BA's largest error, at 28% of BA's total.
  - It is also the largest |paired difference|, against BA.
  - Dropping it would move the ratio to 0.620. It stays in every analysis; dropping it is shown only as a diagnostic.
- The median image favours BA in every contrast.
- The mean is driven by a few large failures in both directions.

## 1B. Frozen evaluation size (`power_1b.json`)
- **Planning alternative:** a true 10% reduction, modelled by a multiplicative shift of the observed pairs.
- **Projected precision:**

| | N = 128 | 256 | 512 |
|---|---|---|---|
| Single-pair median CI half-width | 0.47 | 0.34 | 0.25 |
| Delta-method SE | 0.26 | 0.19 | 0.13 |
| Approx. two-pair aggregate CI half-width | 0.38 | 0.28 | 0.20 |
| Probability of meeting the confirmation rule | 0.16–0.21 at every N | | |

- **Cost:** 512 does not fit the 4.0 h cap (projected 4.83 h); 256 does (3.70 h).
- **Frozen: N = 256** (128000–128255). 128256–128511 stay unopened.
- **This is a bounded replication, not a decisive test.**
- The limits are in the protocol amendment: tails are bounded by the 128 observed pairs, and the aggregate proxy shares BA outputs.

## 2. Mechanism on common states (60200–60215; `mech.json`, `mech_analysis.json`, `mech.png`)
**Setup.** One B0 corrected trajectory per image. At every state x_t (pre-anchor-operation at anchors), each arm's 4-channel guided cheap
prediction and one shared FP prediction were evaluated with the step-indexed static tables. The static quantizer is stateless.
- **Parity:** B0's traces equal the sampler's own diagnostic errors exactly.
- **Energies** are absolute sums over 16 images × 50 steps.
- **Common normalizer:** Σ‖H‖², identical across arms.

| arm | raw Σ‖e‖² | share in per-image time-mean | interval-mean energy | interval-deviation energy | held residual Σ‖e_t − e_a‖² | weighted held H (approx.) | bridge residual ‖H − v‖² (approx.) | cos(H, v) |
|---|---|---|---|---|---|---|---|---|
| B0 | 6,496 | 0.196 | 4,917 | 1,073 | 2,382 | 40.3 | 21.1 | 0.694 |
| BL@330 | 6,086 | 0.191 | 4,582 | 1,030 | 2,286 | 40.1 | 21.1 | 0.692 |
| BL-tm | 6,096 | 0.194 | 4,606 | 1,021 | 2,264 | 39.4 | 20.6 | 0.693 |
| BL-arch | 6,405 | 0.196 | 4,857 | 1,051 | 2,331 | 40.9 | 21.2 | 0.697 |
| **BA@330** | **33,200 (5.11× B0)** | **0.771** | 31,690 (6.4× B0) | **944 (0.88× B0)** | **2,152 (0.90× B0, 0.94× BL)** | 36.6 (0.91×) | **17.2 (0.82×)** | 0.730 |

**Lag structure** (correlation of e_t with e_{t+L}; increment energy Σ‖e_{t+L} − e_t‖²):
- BA's raw correlations are high (0.97 / 0.95 / 0.91 / 0.86 at L = 1 / 2 / 4 / 8) because of its large near-constant offset.
- After removing each image's time mean they are 0.85 / 0.76 / 0.57 / 0.27, against B0's 0.75 / 0.62 / 0.38 / 0.08.
- Increment energy is lower than B0 at L = 1 (2,196 vs 2,549) and L = 2 (3,444 vs 3,768), equal at L = 4, and higher at L = 8 (8,115 vs 7,101).

**Live controlled replay** (bridge accounting).
- **Procedure:**
  - Start from the common B0 latent x_a, before a's anchor operations.
  - Measure d_a; step a with the FP prediction; steps a+1…a+3 use cheap − d_a.
  - x⁻ is the latent at a+4 before the kick; the kick k = −c(d_{a+4} − d_a) gives x⁺.
  - Reference: the FP DDIM continuation from x_a.
  - All 12 intervals, a = 0…44.
- **Values:** mean ‖x − x^F‖² / ‖x^F‖²; image-bootstrap CIs over the 16 images, descriptive only.

| arm | one interval: pre-kick | post-kick | post / pre | cos(k, x^F − x⁻) | ‖k‖ / ‖x^F − x⁻‖ | two intervals (a → a+8, after kick) |
|---|---|---|---|---|---|---|
| B0 | 5.36e-5 | 2.83e-5 | 0.535 | 0.697 | 0.84 | 6.77e-5 |
| BL@330 | 5.10e-5 | 2.73e-5 | 0.545 | 0.692 | 0.85 | 6.31e-5 |
| BL-tm | 5.06e-5 | 2.70e-5 | 0.543 | 0.693 | 0.85 | 6.32e-5 |
| BL-arch | 5.24e-5 | 2.80e-5 | 0.544 | 0.692 | 0.84 | 7.11e-5 |
| **BA@330** | **4.89e-5** | **2.27e-5** | **0.471** | **0.739** | 0.87 | **5.40e-5** |

- **Post-kick error ratios (one interval, all 12 intervals):**
  - BA / B0 0.80 [0.79, 0.81];
  - BA / BL@330 0.83 [0.82, 0.85];
  - BA / BL-tm 0.84 [0.82, 0.86];
  - BA / BL-arch 0.81 [0.79, 0.83].
- **Two intervals:** BA / BL@330 0.86 [0.81, 0.91].
- **Pre-kick:** BA / BL@330 0.96 [0.94, 0.98].
- **By group (one interval, post-kick, ×1e-5):**
  - early: BA 1.05 vs BL@330 1.43;
  - middle: 1.56 vs 1.99;
  - late: 4.20 vs 4.79.
  - BA is lower in every group, and the smallest relative gain is late.
- The exact scalar-coefficient replay agrees with the live kick arithmetic (max |diff| 0). It is an arithmetic check only.

**Standalone vs anchored** (60200–60215, batch 16; endpoint rel. MSE / PSNR / LPIPS):

| branch | plain 50 cheap calls | anchored + bridge |
|---|---|---|
| B0 | 0.137 / 20.1 / 0.307 | 0.0036 / 36.4 / 0.021 |
| BL@330 | 0.133 / 20.9 / 0.285 | 0.0014 / 38.5 / 0.013 |
| BL-tm | 0.143 / 20.5 / 0.298 | 0.0032 / 36.1 / 0.020 |
| BL-arch | 0.119 / 21.3 / 0.271 | 0.0043 / 37.0 / 0.023 |
| **BA@330** | **1.36 / 11.7 / 0.661** | 0.0017 / 38.2 / 0.015 |

**Reading (predeclared rules).**
- **The 5.5× raw increase is real on common states** (5.1× here). About 77% of BA's raw energy is a per-image, near-time-constant offset; the
  local refits have about 19%.
- Anchor reuse removes such an offset: d_a is held across each interval, and the bridge uses differences.
- **Absolute corrected error improves on common-state replay.**
  - Post-kick error is 17–20% lower than every comparator, in early, middle and late intervals.
  - BA's error is less variable within each anchor interval (deviation energy 0.88×, held residual 0.90–0.94×).
  - The bridge captures more of what remains (residual 0.82×, cos 0.74 vs 0.69).
  - By the protocol's rule this is **"error shaping supported" for one-interval and two-interval corrected windows**. It is not established at the full
    trajectory.
- **BA depends entirely on the correction.** As a plain cheap sampler it is catastrophic (rel. MSE 1.36, 10× B0's plain 0.137). It is a hybrid-only
  branch: a scope result, not a failure.
- **Caveats:**
  - 16 inspected images and one training seed;
  - interval CIs are over images only;
  - the linearized coefficient summaries are approximations;
  - none of this substitutes for the fresh full-trajectory evaluation below, where the heavy-tail failures (e.g. 124092) live.

__REPLICATION__

## 5. Real-kernel W8 feasibility (branch `w8-kernel-feasibility`; `results/w8_kernel/REPORT.md`)
**No usable packed pair exists on sm_80** in the installed environments, neither for the cheap side (W4A8-static + rank-8) nor for the anchor side
(W8A16 + rank-8).
- `torch._weight_int8pack_mm` (W8A16) is a naive scalar kernel, 158–306× slower than fp16 cuBLAS at DiT shapes.
- Only INT8×INT8 `_int_mm` (A8, a different precision) is a packed tensor-core GEMM. With unfused quantization glue it is already 1.65–1.84× fp16.
- W4A16 tinygemm is 4.9–9.1× slower than fp16 at these M.
- An unfused rank-8 branch alone costs 0.86× the fp16 GEMMs at batch 1.
- **Beating W8-only therefore remains a hypothetical cost ratio:**
  - required: C_Q/C_W8 < 0.5 − ΔO/(50 C_W8);
  - separately, C_W8/C_FP ≤ 13/25 for K2-W8 vs K4-FP anchors.
- No full-loop timing was possible and no speed claim is made. 0.034 A30-h.
