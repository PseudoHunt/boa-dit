# Temporal decorrelation of quantization error: two oracles (DiT weight pairs; PixArt-Σ static-A8 dither) (pre-registration)

Written 2026-10-27, before any run of this phase. One A30 (24 GB) on a fresh machine: every job runs alone, one at a
time, on `cuda:0`, and each logs its GPU-hours to `gpu_hours.csv` (stage `temporal`). Results go to `REPORT.md`,
generated from the JSONs. Nothing above the marker changes afterwards. Deviations are appended below the marker as they
happen.

**Principle under test.** A many-step sampler heals per-step error that changes from step to step and compounds error
that repeats. Weight-quantization error repeats exactly. Static-scale activation rounding error repeats partly, because
neighbouring steps see nearly the same activations on the same grid. Each oracle makes the error change across steps
without changing its per-step size, then asks whether the sampler removes it.

## Conventions
- **Bootstraps and "beats by ≥ x%":** "beats by ≥ x% (CI excluding 0)" means:
  - the point-estimate reduction 1 − mean(arm)/mean(ref) is ≥ x;
  - and the lower bound of its paired-bootstrap 95% CI is > 0.
  Bootstraps resample seeds or prompts: 2,000 resamples, seed 0.
- **Design and evaluation data are kept apart** (`results/w4a4/rule_addendum_design.md`).
  - DiT calibration-time choices use calibration seeds 1000–1031. DiT evaluation uses eval seeds 0–511, with A0's
    healing check on 0–31.
  - PixArt design choices (static scales, dither offsets, B1's measurements) use the design set: deepcompressor's
    calibration prompts 32–63, seed = index.
  - MJHQ-64 (`mjhq1k[:64]`, seed = index) is used only for evaluation.
- **Diagnostics reported for every arm:**
  - **DiT.** The guided error δ_t = arm − FP is measured on the 4 eps channels of the CFG-combined output, at all 50
    steps of the FP trajectory, for eval seeds 0–31. Each forward uses the arm's own step-indexed tables. Reported:
    - the per-step mean ‖δ_t‖²;
    - C = Σ_seed T·‖mean_t δ_t‖² and I = Σ_seed Σ_t ‖δ_t − mean_t δ_t‖², with T = 50 (the
      `results/w4a4/coherence.py` definition), and C/I;
    - the 4-band radial spectrum (`objective/d2.py` bands) of δ_t, averaged over steps 20–25 inclusive, and its band-1
      share.
  - **PixArt.** δ_g = 4.5·δ_c − 3.5·δ_u. It is measured on a new design cache with all 20 steps: 32 design prompts × 20
    steps × 2 branches, bf16 model, with the transformer inputs and the bf16 output. Reported:
    - the per-step mean ‖δ_g‖²;
    - C, I, C/I over the 20 steps;
    - the band spectrum averaged over steps 8–10 inclusive (the window matching DiT's 20–25 of 50).
- **Compute cap.**
  - 4 A30-hours, stage `temporal`. This covers setup, the core, and the tail in the order below.
  - The cap rises to 5 A30-hours only if all of the following hold:
    - A0 passes its read;
    - B1 does not hit its stop line;
    - the A2 core read is positive: (b) beats (a) by ≥ 15% with the CI excluding 0;
    - the B2 core read is positive: the best dither beats no-dither by ≥ 10% latent MSE (CI excluding 0) and by
      ≥ 0.3 dB.
  - The extra hour pays for B3 and the (a)/(b) FID-2k. Spending stops at 4.0 and results are reported before anything
    past 4 is spent.

## Setup (counted against the cap)
1. DiT FP references:
   - FP eval latents for seeds 0–511, the stored eval protocol: DDIM-50, cfg 1.5, fp16 autocast, batch 32;
   - the FP trajectory (x_t at every step) for eval seeds 0–31.
2. **heig-8 rebuild** (`deflate.quantize_dc --mode heig --r 8 --bits 4`), unchanged except that it also saves, for
   every layer:
   - the deflated Hessian PHP handed to the solver;
   - V;
   - the per-row grid (scale, zero) the solver used.
   **Bit-exactness check:** latent MSE of the rebuilt W_q¹ on eval seeds 0–31, against the first 32 per-image values of
   the stored `results/gate1/eval/latent_deflate_heig_r8_w4.json`. Bit-exact means every per-image value is equal.
   - If bit-exact, arm (a)'s 512-seed numbers are the stored ones.
   - If not, arm (a) is re-run on 512 seeds with the rebuilt checkpoint, every comparison uses only the new numbers,
     and the report says so. Old and new numbers are never mixed.
3. PixArt caches:
   - T5 embeddings for the calib, design, screen and mjhq1k sets;
   - the block-0 calibration cache, needed only to build W4-nobranch;
   - the 20-step design cache above.
4. bf16 MJHQ-64 references: images and final latents, regenerated with the earlier seeds (seed = prompt index) and
   scheduler (DPM-Solver++ 20, guidance 4.5). Then the fp16-vs-bf16 control (`w4a4/dtype_control.py`).
   **Reproduction check:** the control must land within 0.2 dB of the stored 21.27 dB. If it does not, B2 does not
   start; the discrepancy is reported and B waits for a decision.

## Part A: complementary weight roundings, DiT-XL/2
**A0. Healing check for period-2 error.**
- FP model, eval sampler, eval seeds 0–31, from noise, the same batch layout as the FP reference.
- At every step i, an error e_i is added to the guided prediction's 4 eps channels. Its energy per sample is heig-8's
  measured mean ‖δ_g,i‖² (`objective/d1.json`, `dg4_energy_per_sample_by_step`), interpolated log-linearly between the
  measured steps 0, 5, …, 45 and held at step 45's value for steps 46–49.
- The error is white in space: i.i.d. Gaussian per sample, channel and pixel, rescaled to that energy per sample.
- Three time patterns:
  - (i) the same unit vector u at every step: e_i = √E_i · u;
  - (ii) the same u with alternating sign, (−1)^i (step 0 is +);
  - (iii) an independent unit vector at each step.
  The vector seed is 7000 + seed (and + 100·i for pattern (iii)).
- Reported: mean final-latent error ‖x0 − x0_FP‖² per pattern, with bootstrap CIs on the ratios (ii)/(iii) and
  (ii)/(i).

**A1. The pair.** Every quantized layer (the 112 block linears) starts from the rebuilt heig-8 state of that layer:
W_q¹, its grid, PHP and V.
- **W_q²:** GPTQ (the same BoA GPTQ path, natural column order, the same damping) on the target T = 2W − W_q¹, clamped
  to the grid range, using W_q¹'s grid (not re-searched) and the same PHP. Its carry is δ² = (W − W_q²)V.
- **W_q³:** GPTQ of W on the same grid and PHP, in reverse column order. Its carry is δ³ = (W − W_q³)V.
- All three are solved from the same saved per-layer Hessians. W_q² and W_q³ do not re-collect Grams along their own
  prefix; that would be a different Hessian.
- **Solver check:** the fixed-grid solver run on W in natural order must reproduce W_q¹. The fraction of mismatched
  weights is reported. If it is above 0.1%, W_q² and W_q³ are re-defined relative to that solver's own reproduction of
  W_q¹, which then serves as arm (a), and this is recorded as a deviation.
- Recorded per layer:
  - the fraction of weights where W_q² ≠ W_q¹ (and W_q³ ≠ W_q¹);
  - ρ_pair = ‖(E₁ + E₂)X‖²/‖E₁X‖², with E_k = (W − W_q^k)(I − VVᵀ), evaluated as tr(E·PHP·Eᵀ) with the layer's saved
    PHP (calibration statistics);
  - the same quantity for the control pair, (E₁ + E₃).

**A2. Arms.** The step parity is the sampling index i (i = 0 is the first, noisiest step). A weight table and its carry
always switch together.
- Arms:
  - (a) W_q¹ at every step (= heig-8);
  - (b) W_q¹ on even i, W_q² on odd i;
  - (c) W_q¹ / W_q³ alternating (different, not complementary);
  - (d) W_q² at every step (sanity check);
  - (e) (W_q¹ + W_q²)/2 at every step with carry (δ¹ + δ²)/2. This is not 4-bit; it isolates the mean-error reduction;
  - (f) W_q² on even i, W_q¹ on odd i (phase check).
- **Seeds (amended 2026-10-27):** (a), (b), (c), (e) on eval seeds 0–511; **(d) and (f) on seeds 0–127**. Paired CIs
  vs (a) use (a)'s per-image values on the same seeds.
- Every arm also reports the diagnostics above.
- **FID-2k** (seeds 0–1999, the `eval_fid.py` protocol), in tail priority:
  - (a) and (b), only if the A2 core read is positive;
  - **(e) only if (e) and (b) are within ±3% latent MSE of each other.** That is the only case where FID decides the
    mechanism read.

**A3. Storage-efficient probe** (tail, only if read A passes).
- Starting from W_q¹, in each row flip the k weights with the largest |E₁[r, j]|·‖V[j, :]‖ to the adjacent grid level
  on the other side of W[r, j], staying inside the grid. k = 8 and k = 16 give W_q²ₛ. The carry is recomputed as
  (W − W_q²ₛ)V.
- Alternate W_q¹ and W_q²ₛ as in (b). Latent MSE on seeds 0–511.
- The extra storage is k·(⌈log₂ d_in⌉ + 1)/d_in bits per weight.

## Part B: temporal dither of static A8 activations, PixArt-Σ
**Quantizer.**
- **Layers:** every block linear in `w4a4.common.LAYERS` except cross-attention K/V, which stay 16-bit (the corrected
  layer list).
- **Activation quantizer:** static per-tensor asymmetric uint8, one (Δ_t, z_t) per layer per step, shared by both CFG
  branches. No per-token scales and no low-rank branch.
- **Calibration** uses all 32 design prompts on the bf16 trajectory:
  - per (layer, step), the range [α·min, α·max] over all design tokens and both branches;
  - α from {1.00, 0.98, …, 0.60}, chosen to minimize the summed squared rounding-plus-clipping error over the same
    tokens;
  - Δ_t = (hi − lo)/254, z_t = round(−lo/Δ_t).
  For W4 the quantizer sees X̃ = X/d, so calibration is done on X̃ with that model's d.
- **Dither:**
  - x̂ = (clamp(round((x + o_t)/Δ_t) + z_t, 0, 255) − z_t)·Δ_t − o_t, with x first clipped to the calibrated range
    [−z_tΔ_t, (254 − z_t)Δ_t], and o_t ∈ [0, Δ_t).
  - Calibrating 254 intervals leaves code 255 spare. The shifted grid therefore always covers the calibrated range,
    and the offset never adds clipping.
  - The no-dither arm uses the identical (Δ_t, z_t) and clip range, so every arm has the same per-step step size.
  - Deployment note, recorded in the code: o_t folds into the preceding affine's per-step β table (adaLN shift) and
    the next layer's per-step bias table (−W·o_t).
- **Weight settings:**
  - **W16:** bf16 weights; isolates the activation side.
  - **W4-nobranch:** the corrected-base recipe at rank 0: `w4a4.quantize --method hsvd --r 0 --kv16 --range_search`,
    i.e. the closed-form smoothing d plus GPTQ, with no branch. Its build replays earlier in-block layers with the
    builder's own A4 quantizer, unchanged. The A4 activation quantizer is then replaced by the static-A8 quantizer
    above.

**B1. Is static-A8 error coherent?** (W16A8-static, design set)
- Measurements:
  - **Layer level.** On the bf16 trajectory of design prompts 0–7 × 20 steps × 2 branches, take 256 fixed random image
    tokens per layer (seed = layer index; every quantized layer sees image tokens, since cross-attention K/V are
    excluded). Per layer:
    - ρ_adj(t) = the Pearson correlation of e_t = Q_t(x_t) − x_t and e_{t+1} over (prompt, branch, token, channel),
      for t = 0..18, with its mean over t;
    - the coherent fraction ‖Σ_t e_t‖²/(T·Σ_t‖e_t‖²), summed over (prompt, branch, token), with T = 20. Fully
      independent errors give ≈ 1/T = 0.05;
    - the output error energy share, Σ‖W e_t‖² as a fraction of the total over layers.
  - **Network level:** C/I of δ_g on the 20-step design cache (all 32 prompts).
  - **Reference:** the same numbers for dynamic per-token asymmetric uint8 (min/max per token).
- Reported by layer type and per step pair.
- **Stop line:** take the layers carrying ≥ half the output error energy (sorted by share, the smallest such set). If
  network-level C/I < 0.05 **and** their energy-weighted mean ρ_adj < 0.3, Part B stops here and is reported.

**B2. Arms at W16A8-static.** Each arm reports:
- paired relative latent MSE of the MJHQ-64 final latents vs the bf16 latents;
- MJHQ-64 PSNR / LPIPS-VGG vs the bf16 images;
- the diagnostics above.

The arms:
- **no dither;**
- **golden-ratio dither:** o_t = frac(t·0.6180339)·Δ_t per layer, with t the step index 0..19;
- **random dither:** o_t ~ U[0, Δ_t) i.i.d. per layer per step, generator seed 0 (control);
- **calibrated dither:**
  - per layer, o_t is chosen from 16 offsets {k·Δ_t/16} to minimize J = Σ_p ‖W·Σ_t ē_{p,t}(o_t)‖², i.e. the norm
    ‖Σ_t ē‖²_H with H = WᵀW, so J is the output energy of the step-summed mean error;
  - ē_{p,t}(o) is the token-mean rounding error at step t for design-set (prompt, branch) p, prompts 0–15;
  - solved by coordinate descent over steps from o ≡ 0, 2 sweeps;
  - reported: the distribution of chosen offsets and J's decrease.

**B3. NPU-shaped config** (tail).
- W4-nobranch + A8-static: no dither vs the winning B2 dither.
- If the winner is the calibrated dither, it is re-solved for W4: J with W = the dequantized W4 weight, on X̃.
- Reported: MJHQ-64 PSNR / LPIPS, latent MSE, and the diagnostics.

## Order (single GPU)
- **Setup:** DiT references, the heig-8 rebuild and its check, PixArt caches, the bf16 references and fp16 control.
- **Core, in order** (A0 and B1 reported as soon as they land):
  1. A0;
  2. B1;
  3. A1;
  4. A2, all six arms;
  5. B2, if B1 passes.
- **Tail, in priority order, stopping at the cap:**
  1. FID-2k for (a) and (b), if the A2 core read is positive;
  2. B3, if B2 passes;
  3. A3, if read A passes;
  4. FID-2k for (e), only in the ±3% case.
  **If the cap binds, A3 is dropped before B3 or the (a)/(b) FID.**

## Reads (pre-registered)
**A.**
- **Period-2 error is healed** iff A0's (ii) is within 20% of (iii) and ≤ 50% of (i).
- **The mechanism is live** iff all three hold:
  - (b) beats (a) by ≥ 15% latent MSE (CI excluding 0);
  - (b) beats (c) by ≥ 5%;
  - **(d) is within 5% of (a), with its CI reported** (amended 2026-10-27: 128 seeds cannot resolve 3%).
- **The alternation itself is doing the work** iff (b) ≥ (e) − 3%, i.e. (e) does not beat (b) by more than 3%. If (e)
  beats (b) by more than that, the gain is mean-error reduction, and the storage-efficient path must target the mean,
  not the alternation.
- FID-2k must move in the same direction as latent MSE for (b) vs (a).
- (f) is reported as a phase check, with no threshold.

**B.**
- **The lever exists** iff B1 does not hit its stop line.
- **Dither earns its place** iff all three hold:
  - the best dither beats no-dither by ≥ 10% latent MSE (CI excluding 0) at W16A8-static;
  - and by ≥ 0.3 dB on MJHQ-64 at W16A8-static;
  - the gain persists at W4A8-static in B3 (≥ 0.2 dB).
- **The calibrated solver earns its place** iff it beats golden-ratio by ≥ 2% and both beat random. If random matches
  calibrated, the claim is "any decorrelation works", which is cleaner and drops the solver from the method.

**Paper.** One principle with two legs.
- **A and B both pass:** the principle, the two algorithms, and the healing measurements.
- **Only A passes:** the GPU INT4 method with the storage curve; B is a negative result in the analysis.
- **Only B passes:** the NPU method on 20-step SD 1.5 / SDXL (the port is the next step), with A as analysis.
- **Neither passes:** the principle is wrong in the form stated, and the line closes.

---

## Deviations and results
(appended below as they happen; results in `REPORT.md`)
- 2026-10-27 05:31 UTC, setup 2: the heig-8 rebuild is bit-exact (all 112 carries equal to the stored `carry.pt`; the
  32-seed per-image latent MSE equals the stored JSON's first 32 values). Arm (a)'s latent numbers are the stored
  512-seed `results/gate1/eval/latent_deflate_heig_r8_w4.json`; only its diagnostics are run fresh (`diag_only`).
- 2026-10-27, Part B estimators: the α clip search uses 256 fixed random tokens per row (min/max over all tokens), and
  the calibrated dither's token-mean errors ē use 1,024 fixed random tokens per row. Full-token versions would cost
  ≈ 2 A30-hours.
- 2026-10-27 05:24 UTC, A0: read FAILS as written. (ii)/(iii) = 0.020 [0.008, 0.048], so (ii) is not within 20% of
  (iii); it is 50× smaller. (ii)/(i) = 0.0015. Whether A0 counts as passing for the cap-rise rule is referred to the
  user; the read text is unchanged.
- 2026-10-27 06:40 UTC, A1/A2: the pre-registered pair is not complementary. Mean over layers: ρ_pair(1,2) = 3.46,
  ‖E₂‖²_H/‖E₁‖²_H = 3.63 (control W_q³: 2.32 and 1.00). Cause (checked on 3 layers, exploratory, not a result): the
  pre-loop clamp of T = 2W − W_q¹ to the grid range. 1–2% of entries are clamped, and they are exactly the clipped
  outliers whose clip error W_q¹'s GPTQ compensated in other columns. Clamping hides that error from the error
  feedback, so the mirrored compensation and the unmirrored clip error add instead of cancel. Without the clamp
  (4 layers): ρ_pair 1.5–2.1, ‖E₂‖²/‖E₁‖² ≈ 1.85. Pre-registered reads are evaluated on the arms as built.
- 2026-10-27 07:57 UTC, B2: read FAILS. Best dither = golden-ratio: −6.3% latent MSE [CI −11.6, −0.2], +0.31 dB
  [+0.01, +0.60] (needs ≥ 10% and ≥ 0.3 dB). Random: +0.9% worse latent, +0.01 dB. Calibrated: −1.6% worse latent
  (CI ±10%), −0.01 dB, despite its design-set objective falling 98%. All three dithers lower LPIPS by ≈ 0.008–0.009.
  B3 is not run (conditional on B2). The cap stays at 4 h. The exploratory DiT queue (W5/W6, k = 3/4, b′) starts at
  2.55 h, gated on the cap.
- 2026-10-27, exploratory (user-requested after B2; not pre-registered; full table at the top of `REPORT.md`):
  - W5 and W6 single tables (heig-8 recipe, 5.15 / 6.15 bits per weight);
  - k = 2/3/4 cycling of independent GPTQ orderings, storage entropy-coded as code differences from W_q¹
    (`diffcode.json`);
  - the unclamped mirror pair b′;
  - period-2 dither o_t = (t mod 2)·Δ_t/2 at W16A8-static.
  Phase stopped at 3.54 A30-hours.

---

## B′: single-table static activations and symmetric period-2 dither (pre-registration, written 2026-10-27, before any B′ run)
Replaces the earlier Part B stop. Everything in this section is fixed before its first run; deviations are appended
below it, dated.

**Premise.**
- The deployed configuration on a fixed-encoding NPU is one static encoding set per layer for all denoising steps.
  - Qualcomm's on-device SD 1.5 (NeurIPS 2023 expo demo): "one static encoding set independent of the denoising
    stage".
  - ONNX Runtime QNN EP documentation: static, calibration-fixed encodings on the HTP QDQ path; QNN context binaries
    bake encodings into the graph.
- A per-step offset as a runtime graph input, added before quantization and subtracted in the next bias, is the user's
  inference and **unverified**.
- Per-step scale tables already de-phase the rounding error every step, so the B2 baseline already contained a dither.
  Here they become a reference arm (not shippable on such an NPU), and the single table is the baseline.

**Quantizer.** As in Part B (static per-tensor asymmetric, the same layers, cross-attention K/V 16-bit, W16, dither
x̂ = Q(clip(x) + o) − o with one code spare, so the offset never adds clipping), with these changes:
- **Single table:** one (Δ, z) per layer for all steps. The range is [α·min, α·max], min/max over all steps and all
  calibration tokens, α minimizing the summed squared error over all steps' subsampled tokens.
- **Per-step reference:** recalibrated with the same α grid.
- **α grid:** 1.00, 0.98, …, 0.20 (41 values), for every B′ arm.
- **Bit-widths:** A8 (uint8: 254 calibrated intervals, code 255 spare). A6 (uint6: 62 intervals, code 63 spare) is
  a mechanism probe only.
- **Calibration data:**
  - PixArt: the design set (32 prompts, bf16 trajectory), 256 fixed random tokens per row for the α search.
  - DiT: calibration seeds 1000–1031 (FP trajectory, 50 steps, cfg 1.5, both CFG halves), all 256 tokens. DiT
    activation quantization covers all 112 block linears (qkv, proj, fc1, fc2); weights fp32 under the eval
    protocol's fp16 autocast.
- **Offsets** o_t per layer, with t the step index:
  - **symmetric period-2 (headline):** o_t = −Δ/4 on even t, +Δ/4 on odd t;
  - **golden-ratio:** o_t = frac(t·0.6180339)·Δ;
  - **biased period-2 (0, Δ/2):** kept in the record as the variant that creates a period-2 bias on inputs that sit
    exactly on a grid level.
  Zero-mean-per-element rule: a dither sequence must have zero mean error at every input value. (0, Δ/2) violates
  it, giving error mean −Δ/4 for on-grid inputs. ±Δ/4 and golden-ratio do not.

**B1′: gate (commit 0.6 A30-hours: calibration + B1′).** Both beds, measured on design data (PixArt design prompts
0–7, 20 steps; DiT calibration seeds 1000–1031, 50 steps).
- **Layer level,** at A8, on the FP/bf16 trajectory (PixArt 256 fixed tokens per layer; DiT all tokens). Per layer
  and per layer type:
  - the fraction of input elements exactly at zero, and exactly on a grid level of the single table;
  - ρ_adj (mean over t) and the coherent fraction under: single table with no offset, ±Δ/4, golden-ratio and
    (0, Δ/2); the per-step reference; and dynamic per-token uint8;
  - each layer's share of the single-table output error energy Σ‖W e‖².
  - At A6: the same for single table / no offset and single table / ±Δ/4.
- **Network level:** C/I of the guided step error on design data. PixArt uses the 20-step design cache (all 32
  prompts); DiT uses the FP trajectory of seeds 1000–1031, all 50 steps. Arms:
  - single table A8;
  - per-step reference A8;
  - single table A6;
  - dynamic per-token A8.
- **Floor:** the measured dynamic per-token A8 C/I on the same bed (PixArt 0.054 in B1, re-measured here; DiT measured
  here).
- **Gate (core runs iff both hold on at least one bed):**
  1. single-table A8 C/I − floor ≥ 0.03;
  2. under ±Δ/4, the energy-weighted ρ_adj is ≤ −0.3 in the smallest set of layers carrying ≥ half the single-table A8
     output-error energy.
  If the gate fails, the dither line ends here.

**Core (+1.2 A30-hours, only if the gate passes).** A8, W16, on both beds:
- **Arms:**
  - single table;
  - single table + ±Δ/4;
  - single table + golden-ratio;
  - per-step reference.
- **Measurements:** PixArt MJHQ-64 PSNR / LPIPS-VGG and latent rel. MSE vs bf16; DiT latent rel. MSE on seeds 0–511.
  Diagnostics as in Part B.
- **Read: the period-2 dither earns its place iff, on PixArt (the deployed bed):**
  - ±Δ/4 beats the single table by ≥ 10% latent MSE (CI excluding 0) and by ≥ 0.3 dB;
  - ±Δ/4 beats golden-ratio on latent MSE;
  - ±Δ/4 recovers ≥ 50% of the per-step reference's latent gain: (single − ±Δ/4)/(single − per-step) ≥ 0.5. If the
    per-step reference's gain CI over the single table includes 0, this criterion is reported but not applied.
  DiT results are reported alongside.

**Tail (+1.2 A30-hours, only if the core read passes),** labelled as mechanism probes:
- A6 single vs single + ±Δ/4 on both beds;
- DiT FID-2k for single and single + ±Δ/4.
- 2026-10-27, B′ prediction recorded before any B′ run (it does not change the gate or the read). On a synthetic
  slowly varying input, 6 steps, the coherent fraction ‖Σ_t e_t‖²/(T Σ‖e_t‖²) is: no offset 1.00; ±Δ/4 0.22; (0, Δ/2)
  0.30; golden-ratio 0.05. The ρ_adj values are −0.55 / −0.42 / −0.41.
  - Period-2 half-step reaches at most the quarter-variance pair mean, so ≈ 25% of a static input's coherent error
    survives at any T.
  - Golden-ratio offsets equidistribute over the cell, so the step-summed error keeps shrinking with T.
  - Prediction: where slowly varying inputs dominate, golden-ratio removes more coherent error than ±Δ/4, while ±Δ/4
    has the more negative ρ_adj. The read's "±Δ/4 beats golden-ratio" clause is the falsifiable point.
- 2026-10-27 10:35 UTC, calibration: the α search's direct re-quantization of tokens (123 range candidates per
  (layer, step)) ran ≈ 1–2 s per (layer, step), ≈ 1–1.5 h per bed. The first DiT calibration was killed after
  0.53 A30-h (hand-logged in `gpu_hours.csv`; no tables produced).
  - Replaced by a 65,536-bin histogram per (layer, step) over **all** tokens, with errors evaluated at the bin centres.
    On a heavy-tailed test it picks the identical α for all three table kinds, its errors agree to 3e-5, and it is
    ≈ 300× faster.
  - For PixArt this also replaces the 256-token subsample with all tokens.
  - Stage 2's rounding/clipping split (clip energy share, clip rate) and DiT's prediction-error bands are added to the
    B1′ run.
