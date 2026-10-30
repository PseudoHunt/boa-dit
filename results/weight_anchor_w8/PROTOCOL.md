# W8 anchors for cross-precision error reuse (wa8): accuracy pilot — pre-registration

Written 2026-10-30, **before any build, check or generation of this pilot**. Branch `weight-anchor-w8` (from weight-anchor-controls
83f657c323c7490f898ff0d18a94c30bd91a67b0), results in `results/weight_anchor_w8/`, code `boa_dit/weight_anchor/w8.py`, stage tag `wa8`,
checkpoints in `ckpt/weight_anchor_w8/` (not committed). Nothing above the marker changes; amendments are appended below it, dated.
The user-supplied protocol text (2026-10-30) is the specification; this file fixes the concrete choices it leaves open.

## Question and scope
Can W8A16 anchors replace FP anchors while retaining useful sample-specific correction? This pilot measures **accuracy and discrepancy
persistence only**. No speedup, generation-quality, equal-cost or novelty claim can follow from it. Any time measured here is simulation-only.
Out of scope: retraining, new kernels, alternate weight tables, DNS, activation feedback, block skipping, precision sweeps, K = 8 confirmation.

## Budget
- **Hard cap 1.0 A30-hour**, counted in `results/gpu_hours.csv` rows tagged `wa8` (build, checks, profiling, failed/aborted jobs, evaluation).
  wa1/wac1 guards do not apply.
- **Pre-launch guard** per stage (spent + estimate ≤ cap, logged in `budget.json`), plus a **runtime stop**: generation loops check
  spent + elapsed before each batch and stop with a partial checkpoint if the next batch would exceed the cap.
- **Reservation:** confirmation (estimate 0.15 h) and the report are reserved before any optional work. Cut order if short: optional K = 8 dev
  first, then trace diagnostics (reduced to the first 8 dev images, labelled). The confirmation set and primary controls are never shrunk.
- Estimates (from wac1 timings, batch 32: FP-50 ≈ 19 s, W4 50-call ≈ 23 s, R ≈ 28 s per trajectory): build ≈ 0.10–0.20 h; checks ≈ 0.10 h;
  dev + traces ≈ 0.08 h; conf ≈ 0.15 h; optional K = 8 ≈ 0.03 h. The build's first layer is profiled and projected before the full build continues.

## Fixed bed (unchanged from wa1/wac1)
DiT-XL/2 256; original heig-8 W4A16 (`ckpt/ckpt_deflate_heig_r8_w4/model.pt` + `ckpt/temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt`, loaded by
`sampler_corr.run.heig`, W4 bit-exactly unchanged); DDIM-50 eta 0; CFG 1.5 on ε channels 0–2, channel 3 from the conditional half, channels 4–7
(learned variance, conditional half) passed through and unused by DDIM; fp16 autocast forward, fp32 sampler; **batch 32 for every arm**
(guided output via `weight_anchor.run.guided`). No refitted carry checkpoints.

## H8: fixed-calibration W8 + rank-8 carry
Built from the saved layer cache `ckpt/temporal/heig8_layers.pt` (written by `temporal.rebuild`, which captured the exact inputs of the heig-8
solver: projected metric H = P H P, `replace` = 1/256, rank-8 basis V (fp32), and the chosen W4 grid).
For each of the 112 quantized block linears (qkv, proj, fc1, fc2 × 28):
1. target = the **original FP weight** (never the dequantized W4 weight);
2. call the audited solver path **`qutils.boa_gptq_rows` unchanged** (the exact call `deflate.quantize_dc` makes for heig-8: one row group, all rows,
   BoA preprocessing = dead-column replacement + 1% damping, Hessian-weighted grid search `qparam_comput="Hessian"` over the min-max range,
   per-output-channel asymmetric grid, natural column order, GPTQ compensation) with `bits=8` (codes 0..255) and the saved H;
3. the grid (scale, zero) the solver chose is captured from its quantizer; codes c = round(Q/scale) + zero are recovered and must dequantize
   to Q bitwise (scale · (c − zero), the solver's own fake-quantize formula), with 0 ≤ c ≤ 255;
4. carry δ8 = (W_FP − Q8) V with the existing `deflate.carry.make_carry` (fp32 product, stored fp16, V fp16), V shared with W4.
The H8 model = the heig-8 W4 state dict with those 112 weights replaced by Q8 (fp32 dequantized, as the W4 bed stores W4) and the carry
replaced by δ8. Biases, non-block tensors and the auxiliary quantization (adaLN W8 RTN) are those of the W4 bed, unchanged.
**Label: H8 = fixed-calibration W8 + rank-8 carry.** It is not a sequentially recalibrated W8 optimum (H was accumulated through the W4 prefix).

**Replay audit (before W8 is used):** the same procedure with `bits=4` must reproduce, for all 112 layers, the stored heig-8 weights (bitwise),
the stored grid (bitwise) and the stored carry δ4 and V (bitwise), and a forward pass of the rebuilt W4 model must equal the original heig-8
model's output bitwise on one batch. Any difference is explained (e.g. stored dtype) before continuing; an unexplained mismatch stops the pilot.
The artifact `ckpt/weight_anchor_w8/h8.pt` stores codes (uint8), scale, zero, V, δ8, construction settings and source hashes; `build.json`
records per-layer weight error ‖W−Q‖/‖W‖ and H-weighted error for W4 and W8, code ranges, hashes, and bytes:
- theoretical packed bytes (whole model): quantized linears at 4 or 8 bits + per-row fp16 scale and 8-bit zero; carry (fp16 V + δ);
  adaLN at 8 bits + per-row fp16 scale/zero; all other tensors at fp16 — each category reported separately;
- actual simulation resident bytes (fp32 dequantized weights + fp16 carry) reported separately.
If the cache were missing it would be regenerated with `temporal.rebuild` (not RTN); it is present (hash recorded in provenance.json).

## Splits (verified 2026-10-30)
- Development **94000–94031** (32), confirmation **94100–94163** (64).
- Verified absent from `handoff/SEED_REGISTRY.md`, from `*.py/*.md/*.csv` in all 14 remote branches (fetched 2026-10-30) and all local worktrees.
- Labels and noise from `model.labels_for_seeds` / `model.noise_for_seeds`. Correctness check C1 uses the already-evaluated wac1/wa1 dev seeds
  92000–92031 (cached latents), not the new splits.
- Development is for correctness and diagnosis only: nothing (β, K, arms) is selected from it. Confirmation stays unopened until all
  configurations, contrasts and the report generator are committed.

## Prediction rule
At anchor a (calls i ≡ 0 mod K): Q4 and H8 run on the **same** current latent, timestep and conditioning of this arm; d_a = ε_Q4,a − ε_H8,a in fp32 on
consumed channels 0–3; the step uses the H8 guided output **directly**. At a non-anchor t: ε = ε_Q4,t − d_a on channels 0–3 (channels 4–7 from
Q4 at t, unused). β = 1 fixed, no calibration, decay, clipping or schedule. State resets per batch; every arm follows its own trajectory.
Primary K = 4 (anchors at calls 0, 4, …, 48: 13).

## Arms (mandatory; development, then fresh confirmation)
| arm | meaning | Q4 calls | H8 calls | FP calls |
|---|---|---|---|---|
| FP50 | FP reference | 0 | 0 | 50 |
| Q4 | plain heig-8 W4 | 50 | 0 | 0 |
| H8 | plain H8 | 0 | 50 | 0 |
| A8 | H8 at anchors, Q4 otherwise, no reuse | 37 | 13 | 0 |
| R8 | H8 anchors + measured difference reuse, β = 1 | 50 | 13 | 0 |
| S8 | H8 anchors, hold the last H8 guided output (all 8 channels) otherwise, no call | 0 | 13 | 0 |
| R_FP | FP anchors + Q4 error reuse, β = 1 (= wac1's R1) | 50 | 0 | 13 |

## Mandatory checks (stage `checks`, before development; stop on unexplained failure)
- **C1:** on 92000–92031, FP50, Q4 and R_FP equal the cached wac1 latents FP50, Q, R1 (`ckpt/weight_anchor_controls/dev_latents.pt`) bitwise.
- **C2:** R8 with K = 1 equals H8-50 bitwise (anchors return H8 directly).
- **C3:** with the anchor model set to Q4 (Q = H), every recorded correction is exactly zero and R-type sampling equals Q4 bitwise.
- **C4:** in an instrumented R8 run, at every anchor the Q4 and H8 calls receive the identical latent tensor, timestep and labels (asserted by
  equality of the recorded inputs); the state's anchor index equals the last anchor ≤ t.
- **C5:** call counts equal the table for all arms; state is per-trajectory (each call to the sampler starts with empty state; batch rows are
  independent because the model is row-independent — verified by C6b).
- **C6:** code ranges 0..255 (W8) / 0..15 (W4 replay), dequantization reconstruction bitwise, carry shapes [d_out, 8] / [d_in, 8] fp16;
  all outputs finite; **C6b:** H8-50 on images 0–7 alone equals rows 0–7 of the batch-32 run (latent max-abs difference recorded; exact equality is
  not required because GEMM tiling may differ with batch size — recorded, not a stop condition); **C6c:** FP50 with channels 4–7 zeroed equals FP50.
- C1, C2, C3, C4, C5, C6, C6c are stop conditions.

## Development (exploratory) and discrepancy traces
- All mandatory arms on 94000–94031, same metrics as confirmation.
- **Traces:** along the common uncorrected Q4 trajectory (Q4-50) of the 32 dev images, record at all 50 calls the same-state guided outputs of
  Q4, H8 and FP (channels 0–3, fp32). d_t = ε_Q4,t − ε_H8,t, e_t = ε_Q4,t − ε_FP,t.
  - Lags L ∈ {1, 2, 4, 8}, pairs (a, t = a + L): cosine(d_t, d_a), norm ratio ‖d_t‖/‖d_a‖ (pairs with ‖d_a‖ = 0 excluded and counted);
    Σ‖d_t − d_a‖², Σ‖d_t‖², ratio of sums with paired image-bootstrap 95% CI; breakdown by stage of t (early calls 0–16, middle 17–33, late 34–49);
    shuffled control: d_a from a different image (cyclic shift by one image) with the same pairs.
  - Corrected local FP error: Σ‖e_t − d_a‖² vs Σ‖e_t‖² (and vs FP-anchor reuse Σ‖e_t − e_a‖²), same lags.
  - Offline fixed-period K = 2, 4, 8 on the same traces: for every non-anchor t, a = largest anchor ≤ t: ratio Σ‖d_t − d_a‖²/Σ‖d_t‖² and
    Σ‖e_t − d_a‖²/Σ‖e_t‖². These are not endpoint or speed metrics.
  - **Frozen prediction** (written to `drift.json` before any K = 8 live endpoint): the sign of the R8-vs-A8 endpoint change at K = 8 is predicted
    negative (reuse helps) iff the offline K = 8 corrected-error ratio Σ‖e_t − d_a‖²/Σ‖e_t‖² < 1.
- **Optional (only if spent + conf reservation leaves room):** development-only R8 and A8 at K = 8 (7 anchors), compared with the frozen
  prediction. Preliminary check only.

## Confirmation (fresh, 94100–94163)
All mandatory arms; FP50 once as reference. Opened after the `freeze` commit.

## Metrics and pre-registered reading
- Per image vs FP50: endpoint latent rel. MSE ‖x − x_FP50‖²/‖x_FP50‖², PSNR and LPIPS (VGG, sd-vae-ft-ema decode) — same versions and code as wac1.
- Means, medians, paired differences, win counts, paired image-bootstrap 95% CIs (2000 resamples, numpy RNG seed 0); relative changes are ratios of
  cohort means recomputed in each resample.
- **Primary contrast: R8 vs A8.** Secondary: R8 vs S8, Q4, H8, R_FP; also H8 vs Q4 (is the anchor more accurate?), A8 vs Q4.
- **Reading of R8 vs A8:**
  - **useful reuse:** latent MSE change ≤ −10%, CI upper bound < 0, and the LPIPS-difference CI not entirely above 0;
  - **smaller positive:** CI upper bound < 0 but point estimate > −10% (or LPIPS clearly worse — reported as such);
  - **harmful:** CI lower bound > 0;
  - **inconclusive:** otherwise. A CI containing 0 is never read as "no difference".
- **Anchor adequacy:** if H8's mean latent MSE is not below Q4's (H8 vs Q4 CI upper bound not < 0), H8 is reported as an inadequate anchor; nothing is tuned.
- **Retained benefit** B = (E_Q4 − E_R8)/(E_Q4 − E_R_FP), E = cohort mean latent rel. MSE, with paired bootstrap CI and both unnormalized gains.
  Not interpreted if the denominator's CI includes 0. B ≥ 0.8 is an exploratory practical target, not a noninferiority margin.
  Distance to R_FP and distance to H8 are both reported. All absolute errors appear.
- Passing "useful reuse" motivates hardware work only.

## Cost record for Monday
Exact call counts; resident and saved bytes; any time measured here is labelled simulation-only. First-order real-implementation comparison against
H8-50: 50·C_Q4 + 13·C_H8 + overhead vs 50·C_H8, so C_Q4/C_H8 < 0.74 is necessary (zero overhead, K = 4).

## Stages (`python -m boa_dit.weight_anchor.w8 <stage>`)
`build` → `checks` → `dev` (arms + traces + frozen prediction) → [`dev_k8`, optional] → `freeze` (commit) → `conf` → `report`
(`scripts/make_weight_anchor_w8.py`). Commit and push after every stage. At the cap: stop safely and report completed / partial / unrun stages.

---

## Amendments and results
(appended below, dated)
- 2026-10-30, build (0.07 h):
  - Replay audit at bits = 4: weights, grid (scale, zero), carry δ4 and V bitwise on all 112 layers; W4 codes 0..15.
  - H8: codes 0..255, dequantization bitwise, finite. Mean rel. weight error W4 0.202 → W8 0.0146.
  - Note on `H_quad_after_carry`: the cached H is the projected metric P H P, which is blind to errors inside span(V), so the
    carry barely changes that number by construction. It is not a measure of the carry's effect; no decision uses it.
  - Hashes and restoration instructions in `provenance.json`; `h8.pt` is not committed.
- 2026-10-30, checks (0.127 h): C1 (FP50, Q4, R_FP = wac1 FP50, Q, R1 bitwise), C2, C3, C4 (13 anchors), C5, C6, C6c pass; rebuilt-W4 forward bitwise.
  C6b (recorded only): H8 at batch 8 vs batch 32 differs by max-abs 0.0187 in the endpoint latent (GEMM tiling); all arms use batch 32 throughout.
- 2026-10-30, dev + traces (32 images, all mandatory arms): written to dev.json / drift.json. The K = 8 drift prediction in drift.json is frozen
  by this commit, before any K = 8 endpoint is generated.
- 2026-10-30, optional K = 8 dev check run after the prediction commit (dev_k8.json). Freeze record `frozen.json`: all confirmation settings as
  pre-registered; nothing selected from development. Confirmation 94100–94163 opens after this commit.
- 2026-10-30, confirmation (opened once): R8 vs A8 −78.4% [−83.9, −72.7] → useful reuse; H8 adequate; B = 0.983 [0.966, 0.997];
  R8 vs H8 +358%. See REPORT.md / interpretation.md. Seeds 94000–94163 marked evaluated. No further runs.
