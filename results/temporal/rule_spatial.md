# Spatial structure of activation rounding error on PixArt-Σ W4A4 (consolidated plan §1 and §4; pre-registration)

Written 2026-10-28, before any run of this phase. Branch `temporal-oracles`, results in `results/temporal/spatial/`.
Nothing above the marker changes afterwards; deviations are appended below it, dated.

## Budget
- Consolidated allocation **3.0 A30-hours** in total (§1–§5), stage label `c3`, counted across both branches'
  `results/gpu_hours.csv`. Guard: a job does not start if its pilot-projected cost would cross 3.0.
- §1 is planned at ≈ 0.2 h and §4 at ≈ 0.6 h.

## §1 Diagnostic (Read S)
- **Model:** PixArt-Σ 1024², H-SVDQuant W4A4 base `hsvd_r4_b0.5_kv16_u2_rs`: W4 groups of 64 + 16-bit rank-4 branch,
  dynamic per-token INT4 activations in groups of 64 on X̃ = X/d, shifted-unsigned fc2 input; cross-attn K/V 16-bit (not
  quantized, not measured). DPM-Solver++ 20 steps, guidance 4.5.
- **Data:** design prompts 0–7 (the design set; no evaluation prompt is touched), sampled along the W4A4 trajectory.
  Statistics are taken at the 10 even steps 0, 2, …, 18, both CFG rows.
- **Measured quantity:** for every quantized linear whose input has 4096 image tokens (64×64 grid, row-major patch order),
  the activation error e = Q(x̃) − x̃ in X̃ units. This includes the unsigned path's clipping at its zero bound.
  - The same input is also quantized at **A8 dynamic** (same groups; signed 8-bit, levels −128…127, scale max|x|/127;
    unsigned levels 0…255). This is a side computation only: the trajectory is the A4 base.
- **Layer types:** the 8 quantized sites per block (attn1.to_q, to_k, to_v, attn1.to_out, attn2.to_q, attn2.to_out,
  ff fc1, ff fc2), pooled over the 28 blocks.
- **Per layer type and precision:**
  1. **Error share:** output-space error energy ‖e Rᵀ‖² (R = the layer's W4 residual weight), as a share of the total
     over types. This is "the activation error a layer type carries".
  2. **Adjacent-token correlation** of e, horizontal and vertical neighbours on the 64×64 grid, pooled Pearson over
     samples, channels and positions, with the per-(sample, channel) mean over tokens removed ("centred"). The
     uncentred normalized cross-product is reported alongside. The same correlation of x̃ itself is reported for context.
  3. **Same-cell fraction:** the fraction of elements whose horizontal (vertical) neighbour lies in the same grid cell
     of the element's own grid, round(x̃_nb/Δ) = round(x̃/Δ) with the element's Δ.
  4. **Spectrum:** the energy of e in the 4 equal-area radial bands of the 64×64 DFT (`objective/d2.py` band_masks,
     band 1 lowest, DC included in band 1), per (sample, channel), summed. The DC share is reported separately.
     White error gives 25% per band.
- **Read S (on A4):** the spatial lever is live iff **either**
  - (a) the centred adjacent-token correlation (mean of horizontal and vertical) is ≥ 0.3 in **every** layer type of the
    minimal set of types (taken in decreasing error share) that carries ≥ half the output-space error; **or**
  - (b) ≥ 40% of the rounding-error energy lies in band 1, computed as the error-share-weighted mean over all types of
    each type's band-1 fraction.
  If Read S fails, the GPU leg of the spatial direction is closed and §4 is skipped.

## §4 Spatial shaping within a step (runs only if Read S passes; Read P)
- **Arms** (PixArt W4A4, applied at every quantized 4096-token site's activation quantizer):
  - plain rounding (the base);
  - uniform random dither (subtractive, U(−½, ½)Δ, fresh per element and step);
  - (i) **Floyd–Steinberg error diffusion** across the token grid within a step, per channel, in units of each token's Δ
    (the error is carried as a fraction of the source token's Δ and re-scaled by the destination's Δ). This is the oracle:
    it is sequential over 4096 tokens and is not a deployable kernel;
  - (ii) a precomputed **void-and-cluster 64×64 blue-noise mask** (seed 0, σ = 1.5), mapped to offsets in
    [−½, ½) Δ, added before rounding and subtracted after (subtractive), the same for every channel and step;
  - (iii) the same mask, **toroidally shifted per step** by a fixed pseudo-random offset (seed 1).
- **Metrics:** proxy (per-step network prediction error on design prompts, as for the earlier PixArt arms),
  MJHQ-64 PSNR / LPIPS vs the bf16 batch-1 references, endpoint latent rel. MSE (paired bootstrap CI vs plain,
  2,000 resamples, seed 0), and the prediction-error band spectrum.
- **Read P:** (ii) beats plain by ≥ 0.3 dB PSNR **and** ≥ 5% latent rel. MSE with the CI excluding 0, **and** shifts the
  prediction-error spectrum toward the high bands (lower band-1 share than plain). (i) is reported as the ceiling.

---

## Deviations and results
(appended below as they happen)
- 2026-10-28, §1 result (`spatial/diag.json`): **Read S fails** on both criteria.
  - (a) The half-error set is {attn2.to_q} (72% of output-space error); its centred adjacent-token correlation is 0.014.
  - (b) The weighted band-1 share is 0.276 (white: 0.25).
  - The spatial GPU leg is closed and **§4 is skipped** (not run).
