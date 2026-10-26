# W4A4 PixArt-Σ phase — H-SVDQuant base, subspace / allocation / asymmetric target / network-level (pre-registration)

Written 2026-10-26, before any arm of this phase ran. This includes the revisions agreed with the user: no image
generation in the dev loop, a step-error screening proxy, 32 calibration prompts, and images only for base
validation and the final check. Prior art is in `prior_art.md`. Results go to `REPORT.md`, regenerated from the JSONs.
Nothing above the marker changes afterwards.

## Test bed
- **Model:** PixArt-Σ XL-2 1024-MS, from the local copy `/home/jl_fs/pixart_sigma_1024`.
  - DPM-Solver++ multistep, 20 steps, guidance 4.5, 1024², the model's defaults (identical to deepcompressor's PixArt
    `eval` config).
  - Reference precision: bf16.
  - T5-XXL runs once per prompt set (bf16, no-grad), and its embeddings are cached; T5 is then freed. Max sequence
    length 300, negative prompt "".
- **Quantized layers:** every nn.Linear in the 28 transformer blocks, 10 per block, 280 in total: attn1 to_q / to_k /
  to_v / to_out.0, attn2 to_q / to_k / to_v / to_out.0, ff.net.0.proj (fc1), ff.net.2 (fc2).
  - Kept at 16-bit: proj_out, adaln_single (all of it), caption_projection, pos_embed, the modulation tables, T5, VAE.
  - This matches deepcompressor's PixArt skip list (embed / proj_in / proj_out / norm / add-projections). It was checked
    by reading the config, not by running it.
  - Cross-attention K/V (input = the projected caption) are quantized like every other linear.
- **Simulated W4A4 (fake quant, no kernels):**
  - *Weights:* INT4 symmetric, groups of 64 along the input dimension, per output channel. Scale = max|w|/7 per group,
    levels −8..7.
  - *Activations:* INT4 symmetric, per token in groups of 64 channels, dynamic. Scale = max|x|/7 per (token, group).
  - Quantization arithmetic is fp32; layer matmuls are bf16.
  - The same grid is used in every arm, including RTN and GPTQ-r0.
- **Low-rank branch:** y = X̃L₁L₂ + Q_A(X̃)·R̂ᵀ + b, with X̃ = X·D⁻¹ the smoothed, *unquantized* activation. Only the
  residual sees Q_A.
  - Branch parameters r(c_in + c_out) per layer, in 16-bit.
  - Bits per weight are reported: 4 + 16·r(c_in + c_out)/(c_in·c_out), plus the 16-bit group scales (16/64 per
    weight), which every arm pays.

## Calibration (all arms)
- **Data:** the first 32 of deepcompressor's 128 calibration prompts (`examples/diffusion/prompts/qdiff.yaml`, in file
  order). Each prompt is sampled once with the bf16 model, seed = prompt index. All 20 steps, both CFG branches, pooled
  uniformly: 1,280 rows × 4,096 image tokens.
- **H₀ = XᵀX/m:** fp32, over all tokens of all rows.
- **Between blocks:** FP-reference inputs. Block l's inputs are the bf16 model's block-l inputs, propagated by the FP
  blocks.
- **Within a block:** sequential replay. A layer's inputs are recollected after the layers before it are quantized,
  with W4A4 fake quant active. The passes are, in order:
  1. attn1 q, k, v (shared input);
  2. attn1 out;
  3. attn2 q;
  4. attn2 k, v (caption input, independent of the block state; collected in pass 1);
  5. attn2 out;
  6. fc1;
  7. fc2.
  - This matches the paper's Table 1 setting as far as the paper states it.
- **Per-token statistics:** the activation surrogate Σ̄_A, and the per-layer W4A4 output-MSE scoring of H-SVDQuant's
  outer iterations, use a fixed random subsample of 65,536 of the layer's calibration tokens (seed = layer index).
  Grams use all tokens.

## Methods
- **RTN:** W4A4 grid only.
- **GPTQ-r0:** GPTQ (lazy-batch, groups of 64, group scale taken when the group starts, act_order off, damping 1% of
  mean diag) on H₀ with A4 fake quant. No smoothing, no branch.
- **SVDQuant r32:** deepcompressor, if it runs in an isolated venv within 1.5 engineering hours. It is evaluated through
  our screen and our image eval, and its own eval if that runs.
  - *Otherwise, a reimplementation:* SmoothQuant migration d_i = max|X_i|^α / max|W_:,i|^(1−α), with α grid-searched over
    {0.3, 0.4, …, 0.8} by W4A4 output MSE on the token subsample; SVD r = 32 of the smoothed weight; GPTQ on the
    residual against H̃. The report says which was used.
- **H-SVDQuant (the base):** the paper's Algorithm 1 as listed in the handoff §3, with rank 4, λ_A = 0.05, κ_A = 7,
  smoothing exponent p = 2 (Eq. 13), τ = 1e-2 · mean diag(H̃) for the weighted init, and ε = 1% of mean diag for the
  Cholesky.
  - Two outer iterations; the state with the lower W4A4 output MSE on the token subsample is kept.
  - † = pseudo-inverse, by eigendecomposition with relative cutoff 1e-10.
  - **H-SVDQuant r0:** smoothing + surrogate + GPTQ, no branch.

## Screening metric (the dev loop; no images)
- **Cache:** the bf16 model runs once on 64 MJHQ-30K prompts: a fixed subset, `random.Random(0).sample` of the sorted
  meta_data keys, disjoint from the image-eval sets below; seed = prompt index. At 10 evenly spaced steps
  (0, 2, 4, …, 18) it caches (latent model input, timestep, text embedding + mask, FP noise prediction) per CFG
  branch. That is 64 × 10 × 2 = 1,280 triples.
- **Per arm:** one transformer forward per triple, at batch 32 with fixed batch composition.
  - Metric: relMSE = ‖ε_q − ε_FP‖² / ‖ε_FP‖² on the 4 noise channels of the output (the learned-σ channels are
    excluded), averaged over triples.
  - Paired bootstrap over triples vs the base (2,000 resamples, seed 0).
  - Each arm's run is checked for determinism: the base, run twice, gives identical numbers.
- **Per-block table, every arm:** block-l output relMSE vs the FP block on FP-reference block inputs, for 48 held-out
  rows (the first 8 screening prompts × steps 0, 8, 16 × 2 branches).

## Image evals (only these two)
- **Base validation:** MJHQ-256 at 1024² for bf16, SVDQuant-r32 and H-SVDQuant-r4.
  - PSNR and LPIPS (VGG backbone, `lpips` 0.1.4) between each quantized image and the bf16 image with the same prompt,
    seed and scheduler.
  - The protocol offset is ours minus the paper's for SVDQuant-r32 (paper: 16.27 dB / 0.385).
- **Final:** MJHQ-1K for bf16, the base, and the single best arm that passed the screen, with paired bootstrap CIs on
  the per-image PSNR and LPIPS differences.
- **Subsets:** both MJHQ subsets are fixed with a recorded seed (`random.Random(1)`; MJHQ-256 ⊂ MJHQ-1K) and are
  disjoint from the screening prompts.
- sDCI and the original §7 are cancelled.

## Experiments (each = the base with one change; screened)
- **X1, subspace:**
  - r = 4: L₁ from β = 0 (plain SVD of W̃); β = ½ (the base); and heig (top-4 eigenvectors of H̃, gauge
    L₁ = V_rΛ_r^(−1/2)).
  - r = 8: β = ½ and heig only.
  - L₂ is refit in every arm.
  - tr(H̃_⊥)/tr(H̃) is reported per layer for each choice.
- **X2, rank allocation:**
  - e_k = λ_k‖E v_k‖², with E the GPTQ-r0 residual expressed on W̃ and (λ_k, v_k) the eigenpairs of H̃.
  - r_ℓ = the smallest r holding ≥ 90% of Σe_k, floor 0, then capped. The cap is chosen so total branch parameters equal
    the fixed-r4 total within ± 2%.
  - *If degenerate:* if the cap binds in every layer, as happened on DiT-XL/2, X2 is recorded as degenerate and the
    greedy fallback runs instead. It hands out branch directions by the largest marginal e_k per parameter across
    layers, at the same total. That fallback is stated in the report.
  - Subspace = X1's winner (the base's β = ½ if X1 is a tie).
  - The r_ℓ histogram is reported by layer type.
- **X3, asymmetric target:**
  - B1 = the base (within-block order qkv, out, cross-attn, fc1, fc2), so no extra arm is needed.
  - B4 = error order (V, out, fc1, fc2, cross-attn, Q, K) with the asymmetric target.
  - The target is written in correction form, W̃_t = W̃ + W̃(C − H̃)(H̃ + damp)⁻¹, which equals W̃ when C = H̃. The literal
    W̃·C·(H̃ + damp)⁻¹ shrank the weights by 40% at an exact C = H on DiT-XL/2, recorded in `results/engine/REPORT.md`.
  - C = E[x x̃_qᵀ] is the cross-Gram between FP-prefix and quantized-prefix inputs, from a paired pass. Between blocks
    stays FP-reference, so the asymmetric target acts only within the block.
- **X4, network-level:** only if the ρ gate passes (below).
  - X4a: subspace by e_k^F = λ_k‖F_l^(1/2) E v_k‖², L₂ refit.
  - X4b: additionally, the two-sided solve with H_row = F_l on attn1 out and fc2.

## ρ diagnostic
- **Where:** PixArt-Σ blocks 2, 9, 18, 27, with the H-SVDQuant-r4 model's block-output residual on FP-reference
  inputs.
- **F_l:** from 64 random probes through the FP suffix (to the 8-channel transformer output), on 32 screening rows
  (16 prompts × 2 branches) × 3 steps (2, 10, 18).
- **Reported:** λ_max/λ_median and ρ = Σ_t e_tᵀF_le_t / (Σ_t‖e_t‖² · tr(F_l)/1152).
- **Gate:** X4 runs iff ρ ≥ 1 and spread ≥ 10 (step-mean F_l) in at least two of the four blocks.

## Reads
- **Screening read (every arm):** an arm advances iff it cuts the step-error proxy by ≥ 5% vs the base, with the CI
  excluding 0.
- The dB bars below apply only to the final MJHQ-1K confirmation of the single best advancing arm.
- **For the subspace read (2),** the screen decides the direction:
  - heig "wins" a rank iff it cuts the proxy by ≥ 5% vs β = ½ at that rank, CI excluding 0;
  - β = ½ wins a rank iff the reverse holds;
  - otherwise that rank is a tie.

1. **Base valid** iff H-SVDQuant r4 on MJHQ-256 is within 0.5 dB of 16.47 after the protocol offset, **and** it is ≥
   our SVDQuant-r32 on at least one of PSNR (higher) or LPIPS (lower).
2. **Subspace finding** iff heig wins at both r = 4 and r = 8, or β = ½ wins at both. In the second case their choice
   stands and that is said. A tie is a result and is recorded. Confirmation at MJHQ-1K by ≥ 0.3 dB (CI excluding 0)
   applies only if heig is the arm sent to the final.
3. **Allocation adopted** iff X2 passes the screen vs the fixed-r4 base, and, if it is the final arm, beats the base by ≥
   0.2 dB at MJHQ-1K (CI excluding 0).
4. **Asymmetric target earns its place** iff B4 passes the screen vs B1 (= the base), and, if it is the final arm, by ≥
   0.3 dB at MJHQ-1K (CI excluding 0).
5. **Network-level is the paper's center** iff the ρ gate passes **and** an X4 arm passes the screen, **and** it beats
   the base by ≥ 0.5 dB at MJHQ-1K (CI excluding 0) at equal parameters.
6. **A paper exists** iff read 5 holds, or reads 2 (heig wins), 3 and 4 all hold on top of the base. Otherwise this
   line closes and the open results go in a tech note.

Only one arm goes to the final MJHQ-1K. If several pass the screen, it is the one with the lowest proxy.
- **Its dB bar:** the one belonging to the read it belongs to. If it carries several changes, the strictest bar
  applies.
- **Reads without an image check:** for an arm that passes the screen but is not sent to the final, the read records
  "passed screen, not image-confirmed".

## Order and cap
- **Order:**
  1. Infrastructure, and the caches: calibration, screening, image prompt embeddings.
  2. bf16 MJHQ-256 images.
  3. GPU 0: RTN, GPTQ-r0, SVDQuant-r32. GPU 1: H-SVDQuant r4 and r0.
  4. Base validation images (read 1).
  5. ρ on GPU 1 while X1 starts on GPU 0.
  6. X2, X3.
  7. X4 if allowed.
  8. Final MJHQ-1K.
- **Cap:** 14 A30-hours of GPU time, counted from `gpu_hours.csv` rows of stage `w4a4`. Expected ≈ 8–11 h.
- **If the cap binds, cut in this order:** X3, the r = 8 arms of X1, the base-validation SVDQuant image run (the offset
  then cannot be applied, and read 1 is reported without it).

---

## Results
See `REPORT.md`.
