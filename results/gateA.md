# Gate A — time-indexed low-rank residual correction (pre-registration)

Pre-registered 2026-10-25T08:50Z, before any Gate A correction was built or evaluated (the FID anchors of item 1
were already running; they do not enter the rule). The results section is appended below the marker by
`scripts/make_gateA.py`; nothing above the marker changes after this time.

## Question
Does a low-rank correction of the rounding residual that is **indexed by timestep regime** beat a single static
correction **at equal extra bits**, and is it worth its bits relative to plain `uniform` at all?

## Method (as specified; no backbone recalibration)
- Start points: the existing `uniform_w4_K1_lam1_brcfg` and `uniform_w3_K1_lam1_brcfg` checkpoints (adaLN W8),
  loaded via CPU and verified tensor by tensor. Everything for Gate A runs on **cuda:1** (Grams, corrections,
  evaluation).
- For each quantized linear (qkv, proj, fc1, fc2) in all 28 blocks: E = W − Q(W), with W the FP weight and Q(W) the
  checkpoint weight.
- Grams: recomputed on the **FP model** from the cached calibration inputs (seeds 1000–1031, both CFG branches, 50
  steps), per bucket (B = 10, 5 steps each), branch-weighted s² : (s−1)² = 2.25 : 0.25 (normalized), mean over tokens.
  A regime's Gram is the equal-weight mean of its buckets' Grams. GPTQ-style damping: H + 0.01 · mean(diag H) · I.
- H = V diag(s) Vᵀ; M = E H½; rank-k truncated SVD M ≈ U_k Σ_k V_kᵀ; L = U_k Σ_k; R = V_kᵀ H^−½. Minimizes
  ‖(E − L R) H½‖_F. L and R stored and applied in fp16.
- Inference: y = Q(W) x + L(R x), with (L, R) chosen by the current step's regime (bucket = step // 5).
- Regimes: `static` = one Gram over all 10 buckets; `time2` = buckets 0–4 | 5–9; `time3` = buckets {0,1,2} | {3,4,5} |
  {6,7,8,9} (steps 0–14 | 15–29 | 30–49), chosen from the Stage 1 dissimilarity matrices by classical MDS + k-means
  (K = 3), recorded in `configs/time3_split.json`.

## Arms (each at W4 and W3; relative latent MSE on eval seeds 0–511, same layout as all earlier evals)

| arm | regimes × rank | extra bits/param (fp16 L, R; all regimes stored) |
| --- | --- | ---: |
| static-16 | 1 × 16 | 0.296 |
| static-32 | 1 × 32 | 0.593 |
| time2-16 | 2 × 16 | 0.593 |
| time2-32 | 2 × 32 | 1.185 |
| time3-16 | 3 × 16 | 0.889 |
| static-48 (if budget allows) | 1 × 48 | 0.889 |

bits/param = 16 · Σ_layers n_regimes · k · (d_out + d_in) / Σ_layers d_out · d_in, over the 112 quantized linears
(445,906,944 weights; 18,432 · k extra values per block per regime). Equal-budget pairs: time2-16 vs static-32,
time3-16 vs static-48.

## Decision rule (fixed now)
Reduction r(A vs B) = 1 − relMSE(A) / relMSE(B) on the 512 eval seeds; 95% CI by paired bootstrap over images
(2,000 resamples, seed 0).

**Pivot GO** iff, at some bit-width b ∈ {W4, W3}:
- (b = W4) r(time2-16 vs static-32) ≥ 5% with the CI excluding 0, **or** (b = W3) r(time2-16 vs static-32) ≥ 10%
  with the CI excluding 0; **and**
- at that same bit-width, r(time2-16 vs plain `uniform`, uncorrected) ≥ 15%.

Otherwise the pivot is dead: final STOP note, heavy processes stopped. On GO: nothing further runs; this file gets
the full table, the extra-bits accounting and the FID anchors, and the work waits for the next handoff.

## Budget (cap 3.5 GPU-hours for items 1–3 together)
Estimates from measured costs: FID anchors ≈ 0.9, block/layer attribution ≈ 0.4, guard arms ≈ 0.35, Gram pass ≈ 0.15,
Gate A corrections + 10 evals ≈ 0.9, static-48 × 2 ≈ 0.17; total ≈ 2.9. Gate A arms run in this order so the rule's
arms finish first: time2-16 and static-32 (W4, W3), then static-16, time2-32, time3-16, then static-48. If the
running total would pass 3.5, remaining non-rule arms are skipped and recorded as "not run (cap)".
