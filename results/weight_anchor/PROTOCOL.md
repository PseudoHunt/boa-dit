# Measured weight-error anchoring pilot (wa1): pre-registration

Written 2026-10-30, **before any result of this pilot**. Branch `weight-anchor-pilot` (from quantizer-observer-pilot 7352686),
results in `results/weight_anchor/`, code `boa_dit/weight_anchor/run.py`, stage tag `wa1`. Nothing above the marker changes;
deviations are appended below it, dated.

## Question
Can an occasional FP measurement identify weight-induced prediction error that remains useful for correcting later W4 steps?
This tests **scalar reuse of measured output error on one weight-only model**. Other threads stay parked. No training, no new weight
quantizers, no DNS changes, no FID.

## Budget
- **1.0 A30-hour hard cap** in `gpu_hours.csv` rows tagged `wa1`, including checks, failed and aborted runs.
- The cached analysis runs on CPU and is not counted.
- Each GPU stage first checks spent + estimate ≤ 1.0 and stops if not.

## 1. Frozen bed
| item | value |
|---|---|
| model | DiT-XL/2 256 (`ckpt/DiT-XL-2-256x256.pt`, sha256 9ec1876e…) |
| quantized arm | heig-8 W4A16 = `ckpt/ckpt_deflate_heig_r8_w4/model.pt` (sha256 facee154…) + carry `ckpt/temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt` (sha256 f40ea01d…), loaded by `sampler_corr.run.heig` (simulated quantization: dequantized weights in fp32 storage) |
| sampler | DDIM-50 (eta 0), cfg 1.5 on the first 3 ε channels; the 4th ε channel comes from the cond half (official `forward_with_cfg`) |
| precision | fp16 autocast forward, fp32 output and sampler; TF32 at library defaults, as in all earlier caches |
| batch size | 32 (as in every cache used here) |

The **sampler-consumed ε** is channels 0–3 of the CFG-combined output. DDIM with eta 0 ignores channels 4–7 (the variance). Every error,
correction and replacement below acts on channels 0–3 only. Channels 4–7 are passed through unchanged.

### Cache provenance (inspected 2026-10-30)
| cache | seeds | latent at which W4 and FP are evaluated | same latent? | trajectory | use here |
|---|---|---|---|---|---|
| `ckpt/observability/traces_cal.pt` (sha256 89b645a0…) | 7000–7063 | `xfp` = FP DDIM states; `eps` = FP guided ε, `epsq` = heig-8 guided ε, both via `weights_joint.common.step_outputs` on `xfp[i]` | **yes** | **FP** | calibration |
| `ckpt/observability/traces_eval.pt` (sha256 cf46a1d3…) | 7100–7163 | same as above | **yes** | **FP** | held-out cached evaluation (gate) |
| (both files) `xq`, `x0q` | same | free-running heig-8 states; **no ε recorded at them** | n/a | W4 | not used for errors; `x0q`/`x0fp` give the live bit-exactness check |
| `ckpt/temporal/stage1_delta_heig8.pt` (sha256 5261783…; `temporal.stage1.record`) | 0–31 | `delta` = heig-8 guided ε − FP guided ε, both at heig-8's **own** state x^q_i (FP called inside heig-8's eps hook) | **yes** | **W4** | diagnostic only: lag curves along an (uncorrected) W4 trajectory, with β from calibration |
| Stage-1 replay code `boa_dit/temporal/stage1.py` | 0–31 | replays `delta` into the FP sampler | — | — | not rerun |

- Seeds 7000–7163 and 0–31 have all been inspected before, so **every cached result here is exploratory**, including the held-out
  set 7100–7163. It is called "held-out cached" only in the sense that the coefficients are never fitted on it.
- FP-trajectory traces cannot establish persistence along corrected live trajectories; only the live pilot tests that.

## 2. Offline lag experiment (CPU)
- e_i = epsq_i − eps_i (channels 0–3, float64), calls i = 0…49 in chronological order.
- **Lags:** k ∈ {1, 2, 4, 8}. **Anchor indices:** every a with a + k ≤ 49.
- **Buckets:** b(a) = ⌊a/10⌋, five buckets of ten steps: early = 0, middle = 2, late = 4; all five are reported.
- **Coefficient:** β_{b,k} = Σ⟨e_a, e_{a+k}⟩ / Σ‖e_a‖², summed over calibration images and the anchors in bucket b.
  - The unconstrained value is reported as a diagnostic.
  - The deployed value is clipped to [0, 1.5].
- **Arms:**
  - zero: ê = 0;
  - unit: ê = e_a;
  - calibrated: ê = β_{b(a),k} e_a, deployed;
  - shuffled anchor: calibrated reuse, but with e_a taken from image π(s) at the same a.
- **Shuffle:** 20 fixed permutations π of the evaluation images (numpy seed 1000 + j, redrawn until no fixed point).
  Whole image histories move together and timestep alignment is kept.
- **Metrics,** on held-out images:
  - energy remaining R = Σ‖e_{a+k} − ê‖² / Σ‖e_{a+k}‖², pooled, per (lag), per (bucket, lag), and per radial band;
  - the band results use the same whole-output β; band-specific unconstrained β are diagnostic only;
  - bands: the 4 equal-area radial FFT bands of `objective/d2.py`, band 1 lowest;
  - mean per-image cosine cos(e_a, e_{a+k}) and norm ratio ‖e_{a+k}‖/‖e_a‖ against lag;
  - per-image wins of calibrated vs zero and vs unit;
  - paired image-bootstrap 95% CIs: 2000 resamples, numpy seed 0, ratio of pooled sums.
- **Harmful interval:** a (bucket, lag) with held-out R > 1 for the calibrated or the unit arm.
- **Diagnostic:** the same metrics on the Stage-1 W4-trajectory errors, seeds 0–31, using the calibration β.
- The historical "94% of the coherent low-band residual removed" is an oracle on a DCT-16 centred target. It is **not**
  total-error removal and not an endpoint improvement, and is not described as either.

## 3. Periodic-anchor simulation (CPU, cached traces)
- **Intervals:** K ∈ {4, 8}. Anchors at calls 0, K, 2K, … ≤ 49.
- **Per-call error:**
  - anchor arms: anchor calls have error 0 (FP used);
  - a non-anchor call i with last anchor a and age j = i − a ∈ 1…K−1 has error e_i (anchors-only) or e_i − β^K_{b(a),j} e_a (reuse).
- **Coefficients:** β^K_{b,j} is fitted on the calibration images with the same formula, using only that schedule's anchors (a ≡ 0 mod K),
  then clipped to [0, 1.5].
- **Arms:** plain W4 (all e_i); FP anchors only; anchors + reuse; anchors + shuffled reuse (20 permutations).
- **Metrics,** held-out:
  - (i) non-anchor error energy reduction of reuse vs anchors-only: 1 − Σ_{non-anchor}‖e_i − β e_a‖² / Σ_{non-anchor}‖e_i‖²;
  - (ii) temporal-sum error energy ‖Σ_i r_i‖² (summed over images), where r_i is the arm's per-call error;
  - paired image-bootstrap CIs for both;
  - neither is an endpoint-quality metric.
- **Live gate:** at least one K must satisfy all of:
  - reduction (i) ≥ 20%;
  - the paired 95% CI of (i) has lower bound > 0;
  - reuse reduces (ii) relative to anchors-only (point estimate).
- **If neither K passes:** stop and report the lag curves. This is a spending threshold, not an impossibility claim.
- **Choice of K** (on the held-out cached images, before any live image is generated):
  - among the passing intervals, take the larger reduction (i);
  - if the two differ by under 2 percentage points, take K = 8.

## 4. Live pilot (only if the gate passes)
- **Fresh seeds** (checked against `handoff/SEED_REGISTRY.md` and a grep of every worktree, 2026-10-30):
  - development 92000–92031 (32);
  - confirmation 92100–92163 (64).
  - Labels: `labels_for_seeds`; batches of 32.
- **At an anchor call** (i ≡ 0 mod K), at the current live latent:
  - eps_q = W4(x_i, i), eps_f = FP(x_i, i), both through the sampler's own CFG path;
  - anchor_error = eps_q − eps_f (channels 0–3);
  - the sampler consumes eps_f in channels 0–3.
- **At a non-anchor call:** eps = eps_q − s·β^K_{b(a),i−a}·anchor_error in channels 0–3.
- **Rules:** the measurement refreshes at every anchor. No cached FP-trajectory error is ever used live.
- **Arms:**
  1. plain W4A16;
  2. FP anchors only (same schedule, no reuse);
  3. anchors + reuse.
  FP endpoints are the reference. All arms are generated in this run with identical seeds, batches and numerics.
- **Development screen** (32 images): global multiplier s ∈ {0, 0.5, 1}, where s = 0 is the anchors-only arm.
  - Select by the lowest mean per-image latent rel. MSE ‖x − x_FP‖²/‖x_FP‖².
  - **If s = 0 wins, stop:** the measured persistence did not translate into endpoint benefit. Confirmation is not opened.
- **Confirmation** (64 images, run once): freeze K, β and s, then run FP, plain W4, anchors only and reuse.
  - Report paired latent rel. MSE, PSNR and LPIPS (VGG) against the FP endpoint, decoded with sd-vae-ft-ema.
  - CIs: image bootstrap, 2000 resamples, numpy seed 0.
- **Main contrast:** reuse vs anchors only. Reuse vs plain W4 is secondary.
- **Promising** requires all of:
  - reuse latent MSE ≥ 5% below anchors-only;
  - the CI's upper bound below 0% change;
  - no clear LPIPS regression, defined as the ΔLPIPS (reuse − anchors-only) 95% CI **not** lying entirely above 0.
  Smaller effects are reported as measured.
- **Live correctness checks** (run before development, counted in the budget), on eval seeds 7100–7131 (one batch of 32, already inspected):
  - C1: the live harness with no anchors and no reuse reproduces the cached `x0q` **bitwise**;
  - C2: the harness with K = 1 (FP at every call) reproduces the cached `x0fp` **bitwise**;
  - C3: the per-call anchor error measured at FP states equals the cached e_i at the anchor calls of the first two steps.
  - If C1 or C2 fails, the live pilot does not proceed until the cause is found and documented.

## 5. Cost
Record:
- FP and W4 forward latency separately, from CUDA events at batch 32 with the CFG-doubled 64-sample forward;
- wall time per arm;
- peak allocated memory;
- resident model bytes (parameters + buffers) per model;
- residual-buffer bytes (4·32·32 fp32 per image);
- per-arm counts of W4 and FP calls, including the extra W4 evaluations at anchors.

The heig-8 harness uses **simulated** quantization (dequantized weights at full storage precision), so its timing cannot
establish a deployment speedup. FP16 + packed W4 copies together are ≈ 20 bits per weight before overhead. **No memory saving
over FP is claimed.**

Equal-latency FP sampling, higher-precision quantization and additional-W4-step controls are needed before a method claim. They are
deferred until reuse shows a benefit.

## 6. Replay consistency audit
- **Premise correction:** uniform subtractive dither **improved** the static W16A8 DiT bed (latent rel. MSE 0.1066 → 0.0928, Stage 3).
  The dynamic-A4 failure is a different setting.
- **Availability:** no aligned dither traces exist. Stage 3 saved endpoints only, and `ckpt/temporal/pixart/dither_w16.pt` is an offset
  table, not per-step errors. The audit is therefore **deferred** and uses none of this budget.
- **Existing evidence:** the one existing replay, Stage-1 heig-8, reproduced its endpoint as recorded (153.4 vs 153.3), as documented in
  `results/quantizer_observer/propagation_audit.md`.

## 7. Deliverables
In `results/weight_anchor/`:
- PROTOCOL.md;
- provenance.json (hashes);
- lag.json and plots (`lag_*.png`);
- sim.json;
- coefficients.json (frozen β tables, selection decisions);
- checks.json;
- live_dev.json and live_conf.json, or the reason they were skipped;
- cost.json;
- REPORT.md (generated by `scripts/make_weight_anchor.py`).

The GPU-time ledger is `results/gpu_hours.csv`. Commit and push after every job.

---

## Deviations and results
(appended below, dated)
