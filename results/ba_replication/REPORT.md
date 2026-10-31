# bar1 — audit and replication of anchor-aware branch calibration: report

Protocol: `PROTOCOL.md`, pre-registered 2026-10-31 14:06 UTC; dated amendments keep every failure. Kernel track: branch `w8-kernel-feasibility`,
`results/w8_kernel/REPORT.md`. Common ledger: `scripts/h1031b_budget.py`.

**Complete.**
- Spend: **3.593 of 4.0 A30-h** (bar1 3.560 incl. 0.011 failed eval attempt, 0.043 repchecks; wkf1 0.034). Nothing is running.
- Fresh cohort 128000–128255 is evaluated; 128256–128511 stays unopened.
- **Headline:**
  - The 5.5× raw-error increase is real on common states. It is mostly a removable per-image offset.
  - BA's absolute post-correction error is lower on short-window replay.
  - On fresh full trajectories the BA/BL improvement **did not replicate decisively**. The aggregate over the two new pairs is 0.947
    [0.786, 1.150]: **inconclusive**. Both point estimates are < 1.
  - BA was no better than the original branch B0 on this cohort (aggregate 0.985 [0.745, 1.341]).

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

## 3. Two new training-seed replications (`repchecks.json`, `train_s{1,2}.json`, `dev_s{1,2}.json`)
**The completed dar1 procedure was replicated unchanged.**
- Same code path (`anchor_refit.run.train`), only the draw-list generator seed changed: s = 1, 2; the pilot is s = 0.
- STE through the static quantizer: a surrogate gradient, labelled as such.
- Teacher input-state gradient retained.
- Anchor discrepancies recomputed with the current student at the student's latent.
- Windows start from the cached B0 corrected-trajectory anchor states (before anchor operations) with no bridge at a.
- N = 330, checkpoints 82 / 165 / 330. BL-tm time cap = min(BA time, 1260 s).

**Checks:**
- s = 0 reproduces the pilot's draw sha and its first 3 BA and BL (loss, grad-norm) pairs exactly;
- resume bitwise; trained export ↔ install bitwise;
- teacher gets no gradient; W4 codes and tables unchanged; window counts 9/3/2.
- The first attempt's `pass = false` came from a hash-prefix comparison bug. It was re-scored from the recorded provenance; see the amendment.

| seed | BA updates / time | BL-tm updates / time | finite | dev BA@82 / 165 / 330 | dev BL@82 / 165 / 330 / tm | selected |
|---|---|---|---|---|---|---|
| 0 (pilot) | 330 / 1378 s | 565 / 1261 s | yes | 0.00325 / 0.00305 / 0.00208 | 0.00288 / 0.00279 / 0.00264 / 0.00298 | BA@330, BL@330 |
| 1 | 330 / 1389 s | 561 / 1260 s | yes | **0.00222** / 0.00300 / 0.00263 | **0.00260** / 0.00264 / 0.00292 / 0.00314 | BA@82, BL@82 |
| 2 | 330 / 1387 s | 557 / 1261 s | yes | 0.00310 / 0.00241 / **0.00204** | **0.00348** / 0.00456 / 0.00380 / 0.00394 | BA@330, BL@82 |

(dev B0 0.00337, BL-arch 0.00374)

Cost per new pair: 0.774 h training + 0.128 h dev ≈ 0.90 A30-h.

## 4. Fresh evaluation (128000–128255, N = 256 frozen; `eval.json`, `analysis.json`, `eval.png`)
**Setup:** 11 arms, same FP50 and noise, batch 32, counts asserted (50 + 13 / 12). The pilot arms were included by the predeclared budget rule.
1.500 A30-h.

| arm | mean rel. MSE | median | PSNR | LPIPS | share of total error from largest 1 / 5 / 10 |
|---|---|---|---|---|---|
| B0 | 0.00399 | 0.00133 | 36.31 | 0.0235 | 8% / 28% / 38% |
| BL-arch | 0.00408 | 0.00132 | 36.17 | 0.0251 | 7% / 25% / 36% |
| BA_s1 (@82) | 0.00366 | 0.00118 | 36.53 | 0.0215 | 14% / 30% / 42% |
| BL_s1 (@82) | 0.00386 | 0.00134 | 36.29 | 0.0246 | 5% / 22% / 33% |
| BL-tm_s1 | 0.00459 | 0.00149 | 36.31 | 0.0235 | 12% / 37% / 49% |
| BA_s2 (@330) | 0.00420 | 0.00117 | 36.60 | 0.0236 | 11% / 30% / 43% |
| BL_s2 (@82) | 0.00444 | 0.00130 | 36.18 | 0.0245 | 11% / 33% / 44% |
| BL-tm_s2 | 0.00495 | 0.00136 | 36.20 | 0.0255 | 10% / 34% / 48% |
| BA_s0 (pilot @330) | 0.00390 | 0.00125 | 36.63 | 0.0225 | 13% / 32% / 41% |
| BL_s0 (pilot @330) | 0.00483 | 0.00121 | 36.21 | 0.0252 | 10% / 27% / 41% |
| BL-tm_s0 (pilot) | 0.00444 | 0.00133 | 36.27 | 0.0248 | 5% / 22% / 38% |

**Primary (BA-sel / BL-sel per seed):**

| seed | ratio of means [95% CI] | mean paired diff [95% CI] | median diff | wins | ΔLPIPS [95% CI] | ΔPSNR [95% CI] |
|---|---|---|---|---|---|---|
| **1** | **0.949 [0.710, 1.288]** | −2.0e-4 [−1.2e-3, +1.1e-3] | −3.1e-5 | 135/256 | −0.0031 [−0.0064, −0.0000] | +0.24 [−0.19, +0.68] |
| **2** | **0.946 [0.757, 1.203]** | −2.4e-4 [−1.3e-3, +7.5e-4] | −7.0e-5 | 141/256 | −0.0009 [−0.0040, +0.0025] | +0.41 [+0.01, +0.83] |
| 0 (pilot, descriptive) | 0.808 [0.626, 1.051] | −9.3e-4 [−2.0e-3, +2.1e-4] | −1.0e-4 | 147/256 | −0.0027 [−0.0060, +0.0007] | +0.42 [−0.01, +0.85] |
| **aggregate R̄ (two new pairs, joint image bootstrap)** | **0.947 [0.786, 1.150]** | | | | | |

**Secondary (named in advance; aggregates over the two new pairs):**
- BA / BL-tm 0.823 [0.653, 1.025];
- BA / BL-arch 0.962 [0.733, 1.282];
- BA / B0 0.985 [0.745, 1.341].

Per seed:

| | BA/BL-tm | BA/BL-arch | BA/B0 | BL-sel/B0 |
|---|---|---|---|---|
| s1 | 0.797 [0.597, 1.044] | 0.895 [0.658, 1.232] | 0.917 [0.653, 1.300] | 0.967 |
| s2 | 0.848 [0.658, 1.103] | 1.028 [0.766, 1.373] | 1.053 [0.797, 1.411] | 1.113 |
| s0 | 0.879 [0.635, 1.213] | 0.955 [0.700, 1.296] | 0.978 [0.693, 1.386] | 1.211 |

BL-arch / B0 = 1.024.

**Reading (fixed before the results).**
- **Primary latent: inconclusive.** The aggregate CI crosses 1 (upper bound 1.150), even though both new pairs' point estimates are < 1 (0.949, 0.946).
- **This is not a confirmed conditional mean-fidelity improvement.**
- **LPIPS** is secondary.
  - BA is lower than BL-sel in every pair (−0.0031, −0.0009, −0.0027), and only seed 1's CI just reaches 0.
  - **No significant LPIPS regression** in any BA contrast.
  - PSNR is higher for BA in every pair (+0.24 to +0.42 dB).
- **No broad superiority:**
  - BA is not distinguishable from the archived strong local refit (0.962) or from the original branch B0 (0.985).
  - The pilot's BA/B0 0.665 [0.420, 0.964] on 124000–124127 did not recur on this cohort: s0 gives 0.978 [0.693, 1.386].
  - On this cohort the update-matched local refits were themselves no better than B0 (BL-sel/B0 0.97–1.21). So BA's edge over BL-sel partly reflects
    weak local comparators, not BA beating the original branch.
- **Training-seed variability:**
  - The point ratios over three seeds are 0.949, 0.946 and 0.808: consistent in direction, smaller than the pilot's 0.835 for the new seeds.
  - The selected checkpoint differed between seeds (BA@82 vs @330). The dev rule picks noisily at 32 images.
  - The exploratory nested bootstrap over seeds {1, 2, 0} and images gives [0.730, 1.117]. It is fragile with three seeds and does not establish
    population-level seed robustness.
- **Tails:**
  - One image (128228) is the worst for BA_s0, BA_s2, BL_s0 and BL_s2 (rel. MSE about 0.12–0.13), so the failure is shared, not BA-specific.
  - The largest image carries 11–14% of each BA arm's total error.
- As projected in §1B (power ≈ 0.2 for a 10% effect), N = 256 was **not decisive**. The observed effect is about 5% vs BL-sel. This is a bounded
  replication: it neither confirms nor rules out a modest gain.
- No FID, perceptual-equivalence, speed or readiness claim follows. No further sample-size increase or objective change.


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
