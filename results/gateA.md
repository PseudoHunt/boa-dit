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

---

## Results (appended after the runs by `scripts/make_gateA.py`)

### W4 (base `uniform_w4_K1_lam1_brcfg`, uncorrected rel. latent MSE 0.09087)

| arm | extra bits/param | rel. latent MSE | vs uniform (95% CI) | images improved |
| --- | ---: | ---: | --- | ---: |
| static-16 | 0.296 | 0.07046 ± 0.00286 | **+22.5%** [+19.9, +25.0] | 89% |
| static-32 | 0.593 | 0.06506 ± 0.00279 | **+28.4%** [+25.4, +31.5] | 88% |
| time2-16 | 0.593 | 0.06934 ± 0.00285 | **+23.7%** [+21.1, +26.4] | 88% |
| time2-32 | 1.185 | 0.06548 ± 0.00282 | **+27.9%** [+24.9, +30.9] | 88% |
| time3-16 | 0.889 | 0.06993 ± 0.00293 | **+23.0%** [+20.3, +25.8] | 88% |
| static-48 | 0.889 | 0.06130 ± 0.00270 | **+32.5%** [+29.4, +35.7] | 90% |

Equal-budget and rank comparisons (reduction of the first arm vs the second):

| comparison | reduction (95% CI) | images improved |
| --- | --- | ---: |
| time2-16 vs static-32 | **-6.6%** [-9.3, -4.0] | 33% |
| time3-16 vs static-48 | **-14.1%** [-17.8, -10.7] | 28% |
| time2-32 vs static-32 | **-0.7%** [-2.5, +1.2] | 51% |
| static-32 vs static-16 | **+7.7%** [+5.4, +10.0] | 70% |

### W3 (base `uniform_w3_K1_lam1_brcfg`, uncorrected rel. latent MSE 0.39457)

| arm | extra bits/param | rel. latent MSE | vs uniform (95% CI) | images improved |
| --- | ---: | ---: | --- | ---: |
| static-16 | 0.296 | 0.16496 ± 0.00408 | **+58.2%** [+56.8, +59.6] | 100% |
| static-32 | 0.593 | 0.12509 ± 0.00353 | **+68.3%** [+66.9, +69.7] | 100% |
| time2-16 | 0.593 | 0.14897 ± 0.00392 | **+62.2%** [+60.8, +63.7] | 100% |
| time2-32 | 1.185 | 0.12547 ± 0.00352 | **+68.2%** [+66.8, +69.6] | 100% |
| time3-16 | 0.889 | 0.14629 ± 0.00389 | **+62.9%** [+61.5, +64.4] | 100% |
| static-48 | 0.889 | 0.11886 ± 0.00351 | **+69.9%** [+68.5, +71.3] | 100% |

Equal-budget and rank comparisons (reduction of the first arm vs the second):

| comparison | reduction (95% CI) | images improved |
| --- | --- | ---: |
| time2-16 vs static-32 | **-19.1%** [-21.9, -16.6] | 15% |
| time3-16 vs static-48 | **-23.1%** [-26.0, -20.3] | 13% |
| time2-32 vs static-32 | **-0.3%** [-2.1, +1.1] | 48% |
| static-32 vs static-16 | **+24.2%** [+22.2, +26.1] | 90% |

### Captured weighted residual energy (mean over the 112 linears; fraction of ‖E H½‖²_F)

| bits | regime | rank 16 | rank 32 | rank 48 |
| --- | --- | ---: | ---: | ---: |
| W4 | static.0 | 0.184 | 0.255 | 0.308 |
| W4 | time2.0 | 0.221 | 0.309 | 0.373 |
| W4 | time2.1 | 0.190 | 0.263 | 0.317 |
| W4 | time3.0 | 0.292 | 0.407 | 0.485 |
| W4 | time3.1 | 0.203 | 0.284 | 0.345 |
| W4 | time3.2 | 0.194 | 0.267 | 0.322 |
| W3 | static.0 | 0.203 | 0.276 | 0.329 |
| W3 | time2.0 | 0.249 | 0.339 | 0.403 |
| W3 | time2.1 | 0.207 | 0.282 | 0.336 |
| W3 | time3.0 | 0.327 | 0.441 | 0.517 |
| W3 | time3.1 | 0.226 | 0.310 | 0.371 |
| W3 | time3.2 | 0.210 | 0.286 | 0.340 |

### Item 2a — attribution (uniform W4; one block / layer swapped back to FP; seeds 0–63)

Base rel. latent MSE on these 64 seeds 0.0992. Share = (base − swapped) / base; the 28 block shares sum to 0.595 (errors do not add exactly).

| block | 2 | 10 | 11 | 7 | 12 | 9 | 14 | 8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| share | +0.234 | +0.062 | +0.055 | +0.040 | +0.039 | +0.038 | +0.028 | +0.027 |

| layer (top-3 blocks) | share |
| --- | ---: |
| b2.attn.qkv | +0.206 |
| b2.attn.proj | +0.011 |
| b2.mlp.fc1 | +0.036 |
| b2.mlp.fc2 | -0.001 |
| b10.attn.qkv | -0.003 |
| b10.attn.proj | -0.001 |
| b10.mlp.fc1 | +0.024 |
| b10.mlp.fc2 | +0.043 |
| b11.attn.qkv | -0.001 |
| b11.attn.proj | +0.010 |
| b11.mlp.fc1 | +0.018 |
| b11.mlp.fc2 | +0.020 |

### Item 2b — guard arms (top-100 rows kept at W8 RTN on top of `uniform`; seeds 0–511)

Row sets: `gate` = top-100 by RTN-W4 row output error × mean gate energy; `err` = top-100 by the error alone; overlap 41 rows.

| set | bits | extra bits/param | rel. latent MSE | vs uniform (95% CI) | images improved |
| --- | --- | ---: | ---: | --- | ---: |
| gate | W4 | 0.0036 | 0.09493 | **-4.5%** [-5.3, -3.7] | 15% |
| err | W4 | 0.0038 | 0.09494 | **-4.5%** [-5.4, -3.7] | 14% |
| gate | W3 | 0.0045 | 0.42930 | **-8.8%** [-9.1, -8.5] | 0% |
| err | W3 | 0.0047 | 0.42822 | **-8.5%** [-8.8, -8.2] | 0% |

### FID-2k anchors (item 1, `results/stage2/fid_anchors.json`)

| model | FID-2k | rel. latent MSE (512) |
| --- | ---: | ---: |
| FP | 23.89 | 0.0000 |
| uniform W4 | 29.11 | 0.0909 |
| uniform W3 | 130.91 | 0.3946 |

### GPU-hours (items 1–3, since the pivot started)

- item 1 (FID anchors): 0.95
- item 2: 0.66
- item 3 (Gate A): 1.30
- **total: 2.91** (cap 3.5)

### Decision (pre-registered rule)

- W4: time2-16 vs static-32 -6.6% [-9.3, -4.0] (needs ≥ 5%, CI > 0): **False**; time2-16 vs uniform +23.7% (needs ≥ 15%): **True** → fail
- W3: time2-16 vs static-32 -19.1% [-21.9, -16.6] (needs ≥ 10%, CI > 0): **False**; time2-16 vs uniform +62.2% (needs ≥ 15%): **True** → fail

**Gate A: STOP**

