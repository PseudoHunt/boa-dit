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
