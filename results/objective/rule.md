# What does the latent error respond to? Objective diagnostics D1–D4 and two loss arms, DiT-XL/2 256 (pre-registration)

Written 2026-10-26 12:10 UTC, before any run of this phase. GPU 0 was preempted for this phase: the deepcompressor
SVDQuant PTQ was stopped after 28 min and logged as partial, and the W4A4 GPU-0 queue resumes after this phase. The
W4A4 base build on GPU 1 is untouched. Results go to `REPORT.md`, generated from the JSONs. Nothing above the marker
changes afterwards.

## Conventions
- **Base model** for every residual and every arm: heig-8 = `deflate_heig_r8_w4`.
- **Sampler, as the stored latent-MSE eval uses it:** DDIM-50 (eta 0), cfg scale w = 1.5 (DiT `forward_with_cfg`: CFG on
  the first 3 eps channels), fp16 autocast, batch 32, seeds in order. The eval seeds are 0–511 and the class labels
  come from `labels_for_seeds`.
- **Calibration:** seeds 1000–1031, all 50 steps, B = 10 buckets, cfg branch weights, as in `quantize_dc.py`.
- **"Guided prediction" (D2 injection, D4):** the sampler's eps, the 4 eps channels of the CFG-combined output (3
  guided + the 4th from the cond half).
- **"Branch errors" (D1.3):** the 3 guided channels.
- **Steps probed:** 10 evenly spaced, 0, 5, …, 45 (0 = the first, noisiest step).
- **"Beats by ≥ x% (CI excluding 0)":** the point-estimate reduction is ≥ x and the bootstrap 95% CI's lower bound is >
  0. Bootstraps use 2,000 resamples, seed 0.
- **Cap:** 2 A30-hours (stage `objective` in `gpu_hours.csv`).

## D4 — validate the step-error proxy (first)
- **FP trajectory:** for eval seeds 0–63, cache the latent x_t at the 10 probed steps.
- **Proxy:** for each stored W4 arm, rel. MSE of its guided prediction vs FP's on the same x_t, averaged over the 640
  (seed, step) pairs.
- **Arms:** every stored W4 arm with a seeds 0–511 latent MSE:
  - uniform, static-8, static-32, wsvd-8, dc-8, dc-32, heig-8;
  - E1–E6, H2, H4, H4dsf, Hrg;
  - and the D1/D2 arms when they exist.
- **Reported:** Spearman (and Pearson) between the proxy and the stored latent MSE, and the scatter.
- **Read 3:** the proxy is valid iff Spearman ≥ 0.9. If it is < 0.8, the PixArt-Σ screen switches to latent MSE on 64
  prompts, and the cost is reported.

## D1 — guidance
- **Definitions:** v = w·x_c + (1 − w)·x_u for the same (seed, step, token), w = 1.5. Paired rows are the same image in
  the cond and uncond halves of a slab.
- **1. Grams:** one calibration pass along heig-8's own quantized prefix. Blocks 0..l−1 are heig-8 with carries; block l
  has FP linears and W8 adaLN, which is exactly the state `quantize_dc.py` collected H in.
  - Per (layer, bucket) it accumulates S_uu = E[x_u x_uᵀ], S_dd = E[d dᵀ], and S_ud = E[x_u dᵀ], with d = x_c − x_u.
  - H_cfg = S_uu + w(S_ud + S_duᵀ) + w²S_dd, bucket-uniform.
  - H_cur = the pipeline's 0.9·E[x_c x_cᵀ] + 0.1·E[x_u x_uᵀ], bucket-uniform.
- **2. Per layer:** E_eff = (W − W_q)(I − VVᵀ), with heig-8's W_q and V.
  - Reported: r_cfg = tr(E_eff H_cfg E_effᵀ)/tr(E_eff H_cur E_effᵀ).
  - Also the split of tr(E_eff H_cfg E_effᵀ) into shared tr(E S_uu Eᵀ), difference w²·tr(E S_dd Eᵀ) and cross
    w·tr(E(S_ud + S_udᵀ)Eᵀ).
  - By block and layer type; per bucket for blocks 2, 10, 27.
- **3. Network level:** eval seeds 0–31 × the 10 probed steps of the D4 trajectory.
  - Branch errors: δ_c, δ_u = heig-8 − FP eps on the cond and uncond rows of the same forward. δ_g = w·δ_c + (1 − w)·δ_u.
  - Reported: ‖δ_g‖²/mean(‖δ_c‖², ‖δ_u‖²) (total and per step), the Pearson correlation of δ_c and δ_u, and the same
    three-way split of ‖δ_g‖².
  - Also how well the layer-level aggregate predicts it: model-level r_cfg = Σ_ℓ tr(E H_cfg Eᵀ)/Σ_ℓ tr(E H_cur Eᵀ),
    against the network ratio ‖δ_g‖²/(0.9‖δ_c‖² + 0.1‖δ_u‖²), the same normalization as H_cur.
- **Arms (one-sided, heig-8 otherwise):**
  - **CFG-solve:** H_cfg replaces H_cur in the GPTQ solve and grid search; the carry subspace stays the top-8 of H_cur.
  - **CFG-full:** H_cfg for both the solve and the top-8 subspace.
  - Both: latent MSE on seeds 0–511 with a paired CI vs heig-8, plus the per-block table (`downstream.py --all_blocks`
    protocol).

## D2 — sampler frequency response
- **Bands:** the 32×32 2-D DFT (fftfreq grid) is split into 4 radial bands of equal area: frequencies sorted by radius
  |k| and cut at the quartiles. Band 1 is the lowest.
- **1. Gain:** on the FP model, 32 seeds (eval seeds 0–31), starting from the FP trajectory's x_t. At step t, add to the
  guided prediction a random-phase error confined to one band. It is white Gaussian noise per channel and sample
  (seed = 1000·t + band), FFT-masked to the band and scaled.
  - The energy per sample equals heig-8's measured mean ‖δ_g,t‖² over the 4 eps channels at that step, from D1.3's
    forwards.
  - The sampler then continues in FP to the end, with the same batch layout; the unperturbed continuation is
    bit-identical to the FP latent.
  - G(t, band) = mean over seeds of ‖x0_inj − x0_FP‖² / injected energy, with bootstrap CIs over seeds.
  - The white-error G(t) uses the same energy over all bands.
  - The Jacobian-free (TCEC-style) prediction is reported alongside: the squared DDIM coefficient product, i.e. the eps
    coefficient at step t times the x-coefficients of the later steps, from the respaced schedule.
- **2. Spectrum:** S(t, band) = the band spectrum of heig-8's guided error δ_g,t (4 eps channels), same steps and
  seeds, as a fraction of its total per step.
- **3. Tables:** G, S, and harm = G ⊙ S normalized per step. Also the ratio G(t, band 1)/G(t, band 4) per step, with a
  bootstrap CI.
- **4. H_K arm (only if read 2 passes):** blurred-activation Gram over the 16×16 token grid, a Gaussian blur with one
  σ (in tokens) fitted so that the blur's frequency response best matches √G(t, band) averaged over steps 0–20. It
  is applied to both the solve and the subspace, with the carry recomputed. Latent MSE on 0–511, paired CI, per-block
  table.

## D3 — suffix sensitivity ρ
Already run as round-2 §2 (`results/engine/sensitivity.json`): blocks 2 / 10 / 20 / 27, bucket-mean ρ = 2.30 / 0.18 /
0.015 / 0.029, spread ≥ 10 everywhere. The gate passes in 1 of 4 blocks. It is reported here as the D3 read; nothing
is re-run.

## Reads (handoff §6)
1. **Guidance is a lever** iff both of the following hold:
   - the layers whose (difference + cross) share of tr(E_eff H_cfg E_effᵀ) is ≥ 30% together hold ≥ 50% of
     Σ_ℓ tr(E_eff H_cfg E_effᵀ);
   - the network-level ‖δ_g‖²/mean(‖δ_c‖², ‖δ_u‖²) is ≥ 2.
   **The CFG loss earns a place** iff CFG-full or CFG-solve beats heig-8 by ≥ 5% latent MSE (CI excluding 0). The arms
   run regardless of the diagnostic.
2. **The sampler is directional** iff G(t, band 1)/G(t, band 4) is ≥ 3 or ≤ 1/3 at ≥ 2 of the first five probed steps
   (0, 5, 10, 15, 20), with the ratio's CI excluding 1. If it is not, the report records that the per-token Gram is
   right on the frequency axis, and H_K is skipped. **H_K earns a place** iff it beats heig-8 by ≥ 5% (CI excluding 0).
3. **The proxy is valid** iff Spearman ≥ 0.9 (D4).
4. **Whatever passes 1 or 2** is the first loss built on the H-SVDQuant base on PixArt-Σ, as an H_col replacement,
   before X1–X4. If nothing passes 1 or 2 and ρ fails as well (it does, D3), the report records that the latent error
   on this model is not steerable by any per-layer objective at W4, and the engine line closes.

---

## Results
See `REPORT.md`.
