# Gate 2b — pre-registered W4 diagnostic set (written before any arm ran)

Pre-registered 2026-10-25T07:29:44Z, before any Gate 2b arm was quantized or evaluated. The results section below is
appended afterwards by `scripts/make_gate2b.py`; this pre-registration section is not edited after that time.

## Why
Gate 2 failed: at W4, `full` was 11.7% worse than `uniform` (paired 95% CI −17.0% to −6.9%). The headline
diagnostic is how peaked the weight was: the fp32 median gain puts **81.0%** of its bucket weight on steps 0–14
and **3.5%** on steps 25–49 (row-weighted over all layers, the Gate 2 `full` arm: 78.9% / 5.2%; uniform 30% / 50%).

## Arms (all W4; everything else as in the Gate 2 `full` run: branch weights s²/(s−1)², fp32 median gain,
## adaLN W8 RTN, BoA layer-output path, 32 calibration images, B = 10)

Effective weight, computed before running from `boa_dit/weights.py` by `scripts/arm_weights.py` (row-weighted mean
p over all blocks, layers, groups; dead RTN rows excluded):

| arm | definition | steps 0–14 | steps 25–49 | max/min over buckets |
| --- | --- | ---: | ---: | ---: |
| (ref) uniform | p = 1/B, K = 1 | 0.300 | 0.500 | 1.0 |
| (Gate 2) full | gain × gates, K = 4, λ = 1 | 0.789 | 0.052 | 59.3 |
| a | gates-only, K = 4 | 0.238 | 0.600 | 2.4 |
| b | gain-only, λ = 0.5, K = 1 | 0.555 | 0.268 | 4.4 |
| c | full, λ = 0.5, K = 4 | 0.544 | 0.276 | 4.2 |
| d | gain-only, tempered p ∝ (g_t²)^τ, K = 1 | τ = 0.25: 0.470; 0.5: 0.622; 0.75: 0.734 | 0.302; 0.159; 0.076 | 3.4; 10.6; 34.8 |
| e | full, tempered (same τ), K = 4 | τ = 0.5: 0.566 (others computed once τ is picked) | 0.225 | 6.2 |
| f | gain-only, p clipped to max/min ≤ 10, renormalized, K = 1 | 0.710 | 0.155 | 10.0 |

Implementation notes: τ is applied per step before bucket averaging; clipping raises the floor to max(p)/10 on the
normalized bucket weights. Both are new flags (`--tau`, `--clip_ratio`) in `boa_dit/quantize.py`; λ already existed.

## τ selection (closed form, no gradients)
For τ ∈ {0.25, 0.5, 0.75}: quantize gain-only (arm d's family, K = 1) at W4 and evaluate relative latent MSE on
**64 held-out calibration seeds 1032–1095** (never the 512 eval seeds; FP reference generated for those seeds).
Pick the τ with the lowest held-out MSE. Arm d on the eval seeds then reuses that τ's quantized model (identical
config); arm e uses the same τ with K = 4. The held-out curve is reported.

## Metrics per arm
Effective weight on steps 0–14 and 25–49; relative latent MSE on eval seeds 0–511 (mean ± s.e.m.); reduction vs
`uniform` (`uniform_w4_K1_lam1_brcfg`, 0.09087) with a paired bootstrap 95% CI (2,000 resamples, seed 0); fraction
of images improved.

## Decision rule (fixed now)
Proceed to Stage 3 **only if** at least one arm a–f reduces relative latent MSE vs `uniform` by ≥ 5% at W4 (point
estimate) **and** the best such W4 arm (largest point reduction), rerun once at W3, reduces it by ≥ 5% vs
`uniform_w3_K1_lam1_brcfg` at W3. Otherwise: final STOP note. No further arms after this set.

## Budget: 1.5 GPU-hours cap
Measured costs (from `results/gpu_hours.csv`): quantize K = 1 ≈ 0.10–0.11 h, K = 4 ≈ 0.14 h (expected lower now
that the k-means row groups are cached), latent eval on 512 seeds ≈ 0.076 h, held-out eval on 64 seeds ≈ 0.015 h.
Projected total for everything ≈ 1.65 h, **over the cap**. Pre-registered order and stop rule:

1. τ selection (3 quantizations + held-out evals, + one-time held-out FP reference)
2. d (eval only; quantization reused from τ selection)
3. e
4. a
5. c
6. b
7. f (lowest priority: at max/min 10 it is nearly the same weight profile as τ = 0.5, max/min 10.6)

Two arms run at a time, one per GPU. 0.21 GPU-h is reserved for the W3 rerun. Before launching any arm, if
(GPU-hours spent on this set) + (that arm's projected cost) + (the W3 reserve, if not yet used) > 1.5, the arm is
not run and is recorded as "not run (cap)". GPU-hours for the set are summed from `results/gpu_hours.csv` rows that
start at or after the pre-registration time and belong to these arms.

## Relation to the earlier RESPONSE (guard arms)
The earlier RESPONSE asked that, on a Gate 2 fail, guard arms (top-0.1% rows at W8) be run at W4 before any STOP.
This later RESPONSE pre-registers a closed set and says "no further arms after this set". This document follows the
later one. Guard arms and the row-attribution diagnostic are **not** run here; they are flagged to the human in
`REPORT.md` for a decision (they would apply in Stage 3 if an arm passes, per item 4 of this RESPONSE).

---

## Results (appended after the runs by `scripts/make_gate2b.py`)

**Headline diagnostic (peakedness):** the fp32 median gain alone puts 81.0% of its weight on steps 0–14 and 3.5% on steps 25–49 (max/min over buckets 114).

### τ selection on held-out calibration seeds 1032–1095 (64 images, W4, gain-only K=1)

| τ | held-out rel. latent MSE | steps 0–14 | steps 25–49 |
| ---: | ---: | ---: | ---: |
| 0.25 | 0.09091 ± 0.01136 | 0.470 | 0.302 |
| 0.5 | 0.10613 ± 0.01268 | 0.622 | 0.159 |
| 0.75 | 0.11462 ± 0.01318 | 0.734 | 0.076 |

τ* = **0.25**

### W4 arms vs `uniform` (`uniform_w4_K1_lam1_brcfg`, rel. latent MSE 0.09087), eval seeds 0–511

| arm | definition | name | steps 0–14 | steps 25–49 | rel. latent MSE | reduction vs uniform | 95% CI | images improved |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| a | gates-only, K=4 | `gates_w4_K4_lam1_brcfg` | 0.238 | 0.600 | 0.09226 ± 0.00327 | **-1.5%** | [-5.9%, +2.6%] | 52% |
| b | gain-only, λ=0.5, K=1 | `gain_w4_K1_lam0.5_brcfg_gain_iso_rho0.003_fp32_gmedian` | 0.555 | 0.268 | 0.09480 ± 0.00309 | **-4.3%** | [-9.0%, +0.2%] | 45% |
| c | full, λ=0.5, K=4 | `full_w4_K4_lam0.5_brcfg_gain_iso_rho0.003_fp32_gmedian` | 0.544 | 0.276 | 0.08997 ± 0.00287 | **+1.0%** | [-3.2%, +5.2%] | 49% |
| d | gain-only, tempered τ*, K=1 | `gain_w4_K1_lam1_brcfg_gain_iso_rho0.003_fp32_gmedian_tau0.25` | 0.470 | 0.302 | 0.00000 ± 0.00000 | **+100.0%** | [+100.0%, +100.0%] | 100% |
| e | full, tempered τ*, K=4 | `full_w4_K4_lam1_brcfg_gain_iso_rho0.003_fp32_gmedian_tau0.25` | 0.402 | 0.397 | 0.09928 ± 0.00344 | **-9.2%** | [-13.9%, -4.8%] | 42% |
| f | gain-only, clipped max/min ≤ 10, K=1 | `gain_w4_K1_lam1_brcfg_gain_iso_rho0.003_fp32_gmedian_clip10` | 0.710 | 0.155 | not run (cap) or pending | — | — | — |

GPU-hours spent on the set: **1.35** (cap 1.5).

### Decision (pre-registered rule)

- Best W4 arm: **d** (gain-only, tempered τ*, K=1), reduction +100.0% (needs ≥ 5%): **True**.
- W3 rerun reduction: pending

**Gate 2b: PENDING**

