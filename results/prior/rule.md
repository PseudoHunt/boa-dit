# Premise check — does W4 break the shared positional prior first, and is that error shared? (pre-registration)

Written 2026-10-25, before any arm of this handoff ran. Results are appended below the marker by
`scripts/make_prior.py`; nothing above the marker changes afterwards.

## §0 Rebuild: not needed (stated deviation)
The handoff assumed `ckpt/` was lost. It was not: this is the same instance, and `ckpt/ckpt_uniform_w4_K1_lam1_brcfg/model.pt`,
`ckpt/fp_eval_latents.pt` and the DiT weights are on disk.
- **Integrity check, instead of rebuilding:** Part A's base arm re-evaluates that checkpoint on seeds 0–127. Its per-image
  rel. latent MSE must equal the stored values for the same seeds (`results/stage2/eval/latent_uniform_w4_K1_lam1_brcfg.json`)
  exactly. The sampler is deterministic, and the batch layout is the same.
- **Base FID-2k:** reused (29.11, same checkpoint, seeds 0–1999). It is not regenerated.
- **Base label accuracy:** measured from a 512-image generation.
- **Calibration inputs (seeds 1000–1031):** recomputed on the fly for Part B, as FP-propagated and quantized-prefix
  propagated block inputs, from the same FP-trajectory capture that `quantize.py` uses.

## Conventions
As in the handoff:
- step 0 is the noisiest step;
- onset = 14;
- buckets are 5 steps each;
- logits L = Q Kᵀ / √72 per head;
- "row-centered" means each row's mean is subtracted.

Paired bootstrap: 2,000 resamples, seed 0, over the evaluated images.

## Part A
Arms as in the handoff. The FP Q,K / V rows of `attn.qkv` (weight and bias) are swapped in per step by a forward
pre-hook that reads CTX.step.
- **Latent MSE:** relative latent MSE vs FP on seeds 0–127, with reductions vs base. "Share of base error" = 1 −
  relMSE(arm) / relMSE(base).
- **FID-2k and label accuracy:** FID-2k (seeds 0–1999) for A1, A4 and A6, with base reused. Label accuracy
  (ResNet-50 top-1 vs the conditioning label, seeds 0–511) for base, A1, A4 and A6.

## Part B
**B1, samples:** the shared fraction is computed per (block, head, bucket, branch). The samples are all (image, step)
pairs in the bucket: 32 images × 5 steps.
- D̄ = the mean over those 160 samples of rowcenter(L_FP − L_q·).
- Shared fraction = ‖D̄‖² / mean ‖D_s‖².
- A per-step version (D̄ per step, then averaged over the bucket) is reported as a secondary number.
- The cond branch decides; uncond is reported.

**Inputs:**
- *FP-input definition:* L_q uses the W4 qkv on the FP-propagated input of the block.
- *Deployed definition:* L_q′ uses the W4 qkv on the quantized-prefix input (blocks 0…l−1 quantized, W4 adaLN W8), from
  the same FP trajectory x_t. L_FP is always FP weights on FP inputs.

**B2:** in the W4 model, for blocks 0–4 and steps < 14, the logits become L_q′ + D̄[block, head, bucket(step), branch]
(the deployed-definition D̄).
- *Variants:* the FP-input D̄; and the deployed D̄ truncated to rank 16 (SVD per block, head, bucket and branch).
- *Recovered fraction* = (relMSE(base) − relMSE(B2)) / (relMSE(base) − relMSE(A7)), on seeds 0–127.

## Part C
Runs only if criteria 1 and 2 both pass. Done as in the handoff.

## Verdict (handoff §4)
1. **Routing error is large:** A1's share of base error is ≥ 0.30, with the lower bound of its 95% CI > 0.20, **and**
   FID-2k(base) − FID-2k(A1) ≥ 0.5 × (FID-2k(base) − 23.89).
2. **The early routing error is shared, and repairing it works.** First: at least 3 of blocks 0–4 have a median over
   heads of the shared fraction ≥ 0.5 in **each** of buckets 0, 1 and 2 (deployed definition, cond branch). **And**
   B2's recovered fraction is ≥ 0.50.
3. **It holds on PixArt-Σ:** `attn1` Q,K of blocks 0–6 account for ≥ 20% of the PixArt W4 error, **or** a single
   early block's Q,K accounts for ≥ 8%.

If 1 or 2 fails: STOP before Part C, and `SUMMARY.md` states that the shared-prior quantization mechanism is not
supported on DiT-XL/2. If 3 fails: the mechanism is specific to the class-conditional model. If all pass: write the
summary with the three numbers up front, then stop. Each number is entered in REPORT.md as it lands.

---

## Results (appended by `scripts/make_prior.py`)

Integrity: base reproduces the stored W4 per-image errors on seeds 0–127 (max abs diff 0.0e+00). Base rel. latent MSE (0–127) 0.09226; base FID-2k 29.11 (anchor, reused); FP 23.89.

### Part A (share of base error = 1 − relMSE(arm)/relMSE(base); paired bootstrap 95% CI, seeds 0–127)

| arm | what is FP | rel latent MSE | share of base error removed (95% CI) | FID-2k | label acc | changed class |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| base | nothing | 0.09226 | — | 29.11 | pending | pending |
| A1 | Q,K FP, all blocks, all steps | 0.08625 | +6.5% [+3.0, +11.1] | 28.08 | 77.7% | 19.7% |
| A2 | Q,K FP, all blocks, steps < 14 | 0.09262 | -0.4% [-1.3, +0.5] | — | — | — |
| A3 | Q,K FP, all blocks, steps ≥ 14 | 0.08580 | +7.0% [+3.2, +11.7] | — | — | — |
| A4 | V FP, all blocks, all steps | 0.06917 | +25.0% [+18.5, +31.6] | 26.32 | 81.1% | 17.6% |
| A5 | Q,K FP, block 2, all steps | 0.09052 | +1.9% [+0.0, +3.9] | — | — | — |
| A6 | Q,K FP, blocks 0–4, all steps | 0.08948 | +3.0% [+1.2, +5.1] | — | — | — |
| A7 | Q,K FP, blocks 0–4, steps < 14 | 0.09255 | -0.3% [-0.9, +0.3] | — | — | — |
| | *Part B arms below; same base* | | | | | |
| B2 | W4 + deployed D̄, blocks 0–4, steps < 14 | 0.09211 | +0.2% [+0.1, +0.3] | 29.07 | 76.6% | 21.5% |
| B2_fpin | W4 + FP-input D̄ | 0.09210 | +0.2% [+0.1, +0.3] | — | — | — |
| B2_r16 | W4 + deployed D̄ at rank 16 | 0.09211 | +0.2% [+0.1, +0.3] | — | — | — |

Shares need not add up: routing (Q,K) and value errors interact.
- Q,K vs V: A1 removes 6.5%, A4 removes 25.0% → **V (values) dominates**.
- Early vs late routing: A2 (steps < 14) -0.4%, A3 (steps ≥ 14) 7.0%.

### Part B1 (median over the 16 heads; cond branch; shared = ‖D̄‖² / mean ‖D‖² over the bucket's 160 (image, step) samples; per-step version in brackets)

| block | def | bucket 0 | bucket 1 | bucket 2 | bucket 7 | bucket 8 | bucket 9 | branch cos (b0) | rank-16 energy (b0) | kernel cos (b0) |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | deployed | 0.07 [0.07] | 0.06 [0.06] | 0.06 [0.06] | 0.09 [0.10] | 0.11 [0.11] | 0.11 [0.13] | 0.74 | 0.94 | 0.26 |
| 0 | fpin | 0.07 [0.07] | 0.07 [0.07] | 0.06 [0.06] | 0.10 [0.10] | 0.12 [0.12] | 0.12 [0.14] | 0.70 | 0.95 | 0.24 |
| 1 | deployed | 0.18 [0.18] | 0.18 [0.18] | 0.17 [0.18] | 0.12 [0.12] | 0.13 [0.13] | 0.12 [0.14] | 0.85 | 0.97 | 0.28 |
| 1 | fpin | 0.17 [0.17] | 0.15 [0.15] | 0.12 [0.13] | 0.14 [0.14] | 0.16 [0.16] | 0.12 [0.15] | 0.86 | 0.96 | 0.33 |
| 2 | deployed | 0.04 [0.04] | 0.04 [0.04] | 0.04 [0.04] | 0.08 [0.08] | 0.07 [0.07] | 0.06 [0.07] | 0.97 | 0.75 | 0.26 |
| 2 | fpin | 0.04 [0.04] | 0.04 [0.04] | 0.04 [0.04] | 0.08 [0.08] | 0.07 [0.07] | 0.06 [0.07] | 0.97 | 0.75 | 0.27 |
| 3 | deployed | 0.08 [0.09] | 0.09 [0.09] | 0.10 [0.10] | 0.12 [0.12] | 0.10 [0.10] | 0.10 [0.11] | 0.91 | 0.98 | 0.44 |
| 3 | fpin | 0.10 [0.10] | 0.10 [0.10] | 0.10 [0.11] | 0.11 [0.11] | 0.15 [0.15] | 0.14 [0.16] | 0.74 | 0.96 | 0.40 |
| 4 | deployed | 0.12 [0.12] | 0.11 [0.11] | 0.17 [0.18] | 0.11 [0.11] | 0.10 [0.11] | 0.10 [0.11] | 0.88 | 0.95 | 0.42 |
| 4 | fpin | 0.14 [0.14] | 0.12 [0.12] | 0.12 [0.12] | 0.20 [0.20] | 0.24 [0.25] | 0.24 [0.24] | 0.82 | 0.97 | 0.41 |
| 13 | deployed | 0.20 [0.21] | 0.13 [0.14] | 0.07 [0.08] | 0.04 [0.04] | 0.04 [0.05] | 0.04 [0.05] | 0.82 | 1.00 | 0.53 |
| 13 | fpin | 0.20 [0.20] | 0.15 [0.15] | 0.08 [0.08] | 0.05 [0.05] | 0.06 [0.06] | 0.06 [0.06] | 0.87 | 1.00 | 0.47 |

Rank energy of D̄ (bucket 0, deployed, cond; median over heads), rank 4 / 8 / 16 / 32: b0 0.90 / 0.92 / 0.94 / 0.97; b1 0.85 / 0.93 / 0.97 / 0.99; b2 0.54 / 0.62 / 0.75 / 0.88; b3 0.88 / 0.94 / 0.98 / 0.99; b4 0.77 / 0.89 / 0.95 / 0.97; b13 0.94 / 0.99 / 1.00 / 1.00.

### Verdict

1. Routing error large: A1 removes 6.5% [3.0, 11.1] (needs ≥ 30%, CI > 20%): **False**; FID-2k gain +1.03 (needs ≥ 2.61): **False** → **criterion 1 False**.
2. Shared and repairable: blocks 0–4 with median shared ≥ 0.5 in each of buckets 0–2: [] (needs ≥ 3): **False**; B2 recovered fraction -0.52 of A7's gain (needs ≥ 0.50): **False** (FP-input D̄: -0.55; rank 16: -0.52) → **criterion 2 False**.

**STOP before Part C**: the shared-prior quantization mechanism is not supported on DiT-XL/2.

