# Weight-space cancellation oracle and joint two-table GPTQ pilot (protocol)

Written 2026-10-28, before any sampling for this pilot. Branch `weights-joint`, split from `temporal-oracles` at
543b4d1; the activation experiments stay on `temporal-oracles`. Results go to `results/temporal/weights_joint/`. Nothing
above the marker changes afterwards; deviations are appended below it, dated.

## Budget
- **2.0 A30-hours, hard stop,** for everything in this pilot: oracle, audit, constructions and sampling. Logged under
  stage `weights_joint` in `results/gpu_hours.csv`; a guard counts only that stage.
- Every job is pilot-timed on its first batch or layer, its cost projected, and it is launched only if the projection
  fits the remaining budget.
- Unfinished arms are marked **unrun**. Confirmation sample counts are never reduced silently, and the budget is
  never expanded.

## Fixed setting (DiT-XL/2 256, the repo's eval protocol)
- **Checkpoint:** `DiT-XL-2-256x256.pt`. Sampler: DDIM, 50 denoiser calls (one batched CFG forward per call), cfg 1.5
  on the first 3 eps channels.
- **Precision:** fp32 weights, fp16 autocast forward. Activations stay at this high-precision reference setting: no
  activation quantization, dither or feedback.
- **Quantized layer set:** the 112 block linears (attn.qkv, attn.proj, mlp.fc1, mlp.fc2 × 28) at W4, plus the adaLN
  modulation linears at W8 RTN, unchanged from heig-8.
- **Baseline: heig-8** (`deflate_heig_r8_w4`):
  - BoA/GPTQ with a per-output-channel asymmetric uint4 grid (codes 0..15, zero point = integer code, scale per row);
  - grid chosen by BoA's Hessian-weighted grid search;
  - deflated Hessian PHP (P = I − VVᵀ, V = top-8 eigenvectors of the cfg-weighted calibration Gram);
  - a rank-8 fp16 carry δ = (W − Q)V applied as y = Qx + δ(Vᵀx);
  - natural column order; damping 1% of the mean diagonal (BoA preprocess); dead-column replacement 1/256;
  - calibration seeds 1000–1031, 10 buckets.
- **Shared frozen grids:** every table in this pilot reuses heig-8's per-row (scale, zero) and its saved PHP and V
  (`ckpt/temporal/heig8_layers.pt`). Codes are stored as uint8 and dequantized as scale·(code − zero) in fp32 (BoA's
  fake_quantize arithmetic). Grid clipping: codes clamp to [0, 15]; targets are never clamped.
- **Seeds:**
  - design / screen: **4000–4031** (32);
  - oracle: the first 16 (4000–4015);
  - **reserved untouched for later confirmation: 4100–4227** (128; not used in this pilot);
  - calibration seeds 1000–1031 are never used for evaluation.
  - Every compared arm runs on the same seeds, one batch of 32 in seed order (16 for the oracle).
- The revision, configuration, dtype, seeds, order, group size (per output row), scales, zero points and clipping are
  written into each result JSON.

## Part 0: weight-space cancellation oracle (first; within the 2.0 h)
- **Stage-1 metadata, verified:** the Stage-1 source is heig-8, W4 weights with high-precision activations. δ_t was
  measured on the 4 eps channels of the CFG-combined output along heig-8's own trajectory, and injected into the same
  channels of the FP sampler. It is therefore described as a **weight-induced prediction-error replay**. Reference:
  sign-alternated replay / recorded = 0.0023 [0.0008, 0.0047] (seeds 0–31, fp16 autocast).
- **Construction:**
  - W_plus = the effective heig-8 weight W_q + δVᵀ (carry folded in) for the 112 block linears, and the W8 adaLN
    weights;
  - W_minus = 2W − W_plus, unquantized and unclamped. This is an oracle, not a deployable W4 method.
  - Computed in fp64, stored in fp32. The symmetry discrepancy ‖(W_plus − W) + (W_minus − W)‖ / ‖W_plus − W‖ is reported.
- **Precision:** the oracle runs in **fp32 (no autocast)**, so that fp16 casting cannot break the symmetry. Its FP
  reference trajectory is fp32 too.
- **Trajectories** (seeds 4000–4015):
  - static W_plus;
  - static W_minus;
  - alternating W_plus/W_minus;
  - alternating W_minus/W_plus.
  Alternation is by denoiser call index, with the same table for both CFG branches. Endpoint latent rel. MSE vs the
  fp32 FP reference, with paired bootstrap CIs.
- **Same-state diagnostic** on the fp32 FP trajectory states x_t (seeds 4000–4015, all 50 steps):
  - e_±,t = f(W_±, x_t) − f(W, x_t) on the guided prediction (4 eps channels);
  - reported: both energies, their cosine, the energy of the mean ‖(e_+ + e_−)/2‖², and that energy relative to the
    mean individual energy, per step and overall.
- **Interpretation** (no overclaiming):
  - large endpoint gain from alternation → proceed to joint GPTQ;
  - poor opposition on identical states → weight symmetry does not give prediction symmetry, and joint calibration
    should target layer or prediction errors;
  - good opposition but little live gain → temporal alignment / state sensitivity.
  - A failed oracle rejects only this route.

## Part 1: GPTQ column-cost audit
- **Method:** re-run heig-8's GPTQ per layer (fixed grid, same PHP and damping; reproducing W_q exactly is checked)
  with the cost of each input column logged as Σ_rows (w_j − q_j)² / U_jj², where U = upper Cholesky of H_damped⁻¹
  (BoA's convention) and U_jj² = [H_F⁻¹]_jj.
- **Check:** on small matrices, Σ_j cost_j = tr((W − Q) H_damped (W − Q)ᵀ).
- **Reported:** per layer and overall, the cost fraction of the final column, the final 16 and 64 columns (processing
  order: natural; no permutation), and the highest-cost 16 and 64 columns. This is a headroom diagnostic, not an
  endpoint predictor.

## Part 2: joint two-table GPTQ
- **Surrogate:** L = ‖(E_A + E_B)/2‖²_H + β‖(E_A − E_B)/2‖²_H, with E = (Q − W) on the damped PHP metric. **β = 0.1**,
  fixed, with no tuning.
- **Algorithm:**
  - two compensated working copies;
  - at column j, per row, enumerate all 16 × 16 legal code pairs and score
    [(e_A + e_B)²/4 + β(e_A − e_B)²/4] / U_jj², with e = w_work − q. Ties are broken by the lowest flattened index;
  - each table then gets the usual GPTQ update W_X[:, j:] −= (e_X/U_jj) ⊗ U[j, j:];
  - targets are not clamped or mirrored.
- **Selective variation:** columns outside S are forced to q_A = q_B (the best shared code under the same score).
  Shared codes may differ from the original static GPTQ codes.
- **Carries:** each table gets its own carry δ_X = (W − Q_X)V; V is shared.

## Part 3: validation (synthetic, CPU; the sampling queue stops on any failure)
1. Pair score and compensation vs a direct solve of the remaining continuous quadratic.
2. Telescoping: Σ logged pair costs = L(Q_A, Q_B).
3. β = 1 gives the independent objective (‖E_A‖² + ‖E_B‖²)/2.
4. With identical init, deterministic ties and β = 1, the result equals ordinary GPTQ (BoA's loop) exactly.
5. Table-swap symmetry of the objective and of the alternating sampler (A/B vs B/A swap phases only).
6. Identical tables in the alternating sampler give latents bit-identical to the static model.
7. Grid bounds and dtype round-trips (codes in [0, 15], scale·(code − zero) exact).

## Part 4: arms (screen, seeds 4000–4031, exploratory)
- **A.** static heig-8.
- **B.** joint pair, final column varying.
- **C.** joint pair, final 16 columns.
- **D.** joint pair, highest-cost 16 columns (from the Part-1 audit of static GPTQ).
- **E.** joint pair, 16 random columns per layer (frozen selection seed 0).
- **F.** joint pair, all columns.
- **Matched control:** the existing independently constructed pair (W_q1 / W_q3 alternating) on these seeds.
- **Selection** is made independently in each layer.
- **Alternation** is by denoiser call index; the same table serves both CFG branches.
- **If the cap binds:** keep A, one selective joint pair, and its controls before more constructions.

## Part 5: controls for the most promising pair
"Most promising" means the lowest screen latent among B–E (F is the reference). Its controls:
- Q_A static;
- Q_B static;
- alternating Q_A/Q_B;
- static average Q_mean = (Q_A + Q_B)/2 in fp32, an exact half-step value with no requantization; on varied columns
  this is a 5-bit code;
- **static extra precision on the same selected columns,** budget permitting: GPTQ with a half-step (5-bit) grid on
  the columns in S and the W4 grid elsewhere.

## Measurements (each sampled arm)
- **Endpoint:** latent rel. MSE vs the FP reference (fp16-autocast FP latents of the same seeds), per seed, with paired
  bootstrap CIs vs A (2,000 resamples, seed 0).
- **Prediction error:** per-step energy, coherent energy C, I and C/I, on the FP trajectory of the design seeds.
- **Reconstruction:** individual-table error tr(E H Eᵀ) per layer; the shared and alternating components
  ‖(E_A ± E_B)/2‖²_H; the fraction of differing codes.
- **Storage:**
  - actual serialized bytes;
  - packed-INT4 bytes (computed);
  - scale/zero overhead;
  - selection metadata (|S|·⌈log₂ d_in⌉ bits per layer);
  - carries;
  - runtime weight-buffer memory (the simulation holds dequantized fp32 tables);
  - entropy of the code differences, labelled as an **estimate**, never as achieved compression.
- **Timing:** sampling wall time per arm. This is simulation runtime, not a packed INT4 implementation.

## Report
REPORT.md answers, separately:
- whether joint design beats the independent pair;
- whether selective variation beats random selection at equal size;
- whether alternation beats each constituent table;
- whether alternation beats or approaches the static average;
- whether there is a plausible quality–storage–runtime advantage.
A lower endpoint error alone does not show that temporal shaping beats extra precision.

## Excluded
Stochastic-rounding schedules, four-table constructions, β searches, and sampler-aware Hessians.

---

## Deviations and results
(appended below as they happen)
- 2026-10-28, audit check failure (found in the build run, before any arm result was read):
  - The static logger used nearest-level search where BoA uses round(w/scale) + clamp. At fp boundaries they disagree:
    24 of 112 layers differ from heig-8's W_q on ≤ 0.012% of weights (mean 0.0006%).
  - The damped quadratic used the pre-retry H, but BoA's Cholesky step re-damps H when factorization fails. Telescoping
    error was 79% on b27.mlp.fc2 and ≤ 1e-4 elsewhere.
  - **Fix:** the static logger now uses BoA's exact rounding, and every quadratic (audit and reconstruction components)
    uses H_eff = (UᵀU)⁻¹, the H actually used.
  - The audit is re-run (`audit_rerun.json`, reported as the authoritative Part-1 numbers).
  - The joint constructions are unaffected: they compensate with the same U and their own rounding.
  - Construction D's highest-cost-16 selection was made from the pre-fix costs; this is recorded, and its overlap with
    the re-run selection is reported.
  - The `build.json` reconstruction totals use the pre-fix H on retried layers (b27.mlp.fc2 materially); they are
    recomputed from the stored codes in the report.
