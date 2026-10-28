# E3 confirmation and the PixArt rollout refit (protocol, 2026-10-28, committed before any new result)

Stage tag `f5`. Branch `e3-confirm` (worktree `/home/boa-dit-e3c`, from `followup-weights` at c0edc15).

**Budget.** On 2026-10-28 the user removed the GPU cap for this work ("there is no gpu budget anymore. just keep running
the experiments until they are over"). Every job is still logged in `results/gpu_hours.csv` (stage `f5`, plus the
`deflate/quant_deflate_heig_r16_w4` build). CPU-only steps are logged separately in `results/e3c/cpu_log.md`.
Failed GPU jobs are kept in the log.

## Seed registry (sweep of all five branches, 2026-10-28)
DiT sample seeds already used somewhere:
- 0–3999 (FID sets and tsens);
- 1000–1031 (quantizer calibration);
- 3000–3199;
- 4000–4427;
- 5000–5023;
- 6000–6095;
- 7000–7231;
- 8000–8527.

9000+100·t+s and 100003·site+t are noise-generator seeds in older code. The range **20000–39999 appears nowhere** and is
the source of every new DiT seed here.

| use | seeds |
| --- | --- |
| training states (reused from E3; already inspected) | 8200–8263 |
| M recalibration for the C2 carry (FP vs C2 endpoints) | 7000–7063 (the E1 calibration seeds) |
| paired evaluation (new, untouched) | **20000–20127** (128) |
| FID-2k (new, untouched) | **30000–31999** (2000) |

Labels come from `labels_for_seeds` (one class per seed, the repository convention), so every arm sees the same seeds and labels.

## Part 1: E3 confirmation (DiT-XL/2 256, heig-8 W4, DDIM-50, cfg 1.5)

**Frozen from E3 (no retuning).** Adam with lr 1e-4, B = 4, K = 5, λ = 0.5 (C2), the same loss normalization constants
(`ckpt/followup/e3/cache.pt`), the same training states, and the heig-8 initialization.

**Training seed.** The seed controls the minibatch and start-step draws. The initialization is the fixed heig-8 carry by design.
- Seed 0 of C1t and of C2 is the E3 final run (same code, same seed), reused unchanged.
- Seeds 1 and 2 are new.

**Arms.**

| arm | objective | stopping | training seeds |
| --- | --- | --- | --- |
| C0 | original heig-8 carry (rank 8, 0.148 extra bits/param) | — | — |
| H16 | **original heig recipe at rank 16** (`quantize_dc --mode heig --r 16 --bits 4`, ≈ 0.296 extra bits/param; new W4 codes from the rank-16 deflated solve) | — | — |
| C1t | L_local | 0.3 h (time-matched to C2) | 0, 1, 2 |
| C1u | L_local | 841 updates (update-matched to C2's E3 count) | 0, 1, 2 |
| C2 | L_local + 0.5·L_roll,full | 0.3 h | 0, 1, 2 |
| C2L | as C2 | 0.9 h (3×, saturation) | 0, 1, 2 |
| C1L | as C1t | 0.9 h (3×, saturation) | 0, 1, 2 |
| C2+M | C2 seed 0, plus E1's moment correction M at the final latent, **refitted for this carry** by E1's frozen procedure (`state_corr.fit_M`, bands 1–2, band-1 DC) on C2 vs FP endpoints of seeds 7000–7063 | — | 0 |

M is refitted because E1's coefficients were calibrated on heig-8 endpoints, and the C2 carry changes the endpoint statistics.
E1's frozen heig-8 coefficients are not reused on C2.

**Paired evaluation (seeds 20000–20127).**
- Every arm and seed above gets latent rel. MSE vs FP, decoded PSNR, and LPIPS-VGG.
- An arm's per-image value is the mean over its three training seeds. Paired bootstrap over the 128 images (2,000 resamples, seed 0).
- Each training seed's contrast is also reported separately, with sign agreement out of 3.

**FID-2k (seeds 30000–31999).** Arms C0, H16, C1t s0, C2 s0, C2+M, C2L s0.
- Same reference statistics as every earlier FID (`results/stage0/ref_stats.npz`, ADM VIRTUAL_imagenet256 batch); pytorch-fid pool3.
- Per-image Inception features and latents are saved.
- Uncertainty: paired bootstrap over the 2,000 generated images (the same resampled indices for every arm, 1,000 resamples), against the fixed reference statistics.
  This CI is conditional on the reference set and on n = 2000. It is not a population-quality estimate.
- C2+M reuses the C2 s0 latents; only the decode differs.

**Pre-registered reads.**
1. **Trajectory objective, replicated:** C2 vs C1t on seed-averaged latent. Passes if the CI excludes 0 and all 3 per-seed contrasts share the sign.
2. **vs matched updates:** C2 vs C1u, same rule.
3. **Better carry vs more carry:** C2 (rank 8) vs H16 (rank 16, twice the extra bits), on paired metrics (seed-averaged) and on FID (C2 s0 vs H16).
   - "C2 ≥ H16" means C2 is not significantly worse on latent and FID.
   - "C2 > H16" means C2 is significantly better on at least one of them and worse on neither.
4. **Saturation:** C2L vs C2 and C1L vs C1t. If either improves significantly, 0.3 h was not saturated. Then C2L vs C1L is the time-matched comparison at 0.9 h.
5. **Distributional:** FID of each arm vs C0, with bootstrap CIs.
6. **M on top of C2:** C2+M vs C2 on FID and paired metrics.

Latent, PSNR and LPIPS are fidelity against FP. FID-2k is one distributional estimate. Neither is called "better generation" alone.

## Part 2: PixArt-Σ 1024² W4A4: rollout refit of H-SVDQuant's low-rank branch

The details are fixed by a dated amendment below, **after profiling and before any training result**. Fixed now:
- Base `hsvd_r4_b0.5_kv16_u2_rs`. W4 codes, scales, smoothing and activation quantizers are frozen. Only the existing rank-4 L₁L₂ branch is trained, in its existing shapes.
- Arms:
  - base;
  - L₁L₂ refit with L_local (time-matched);
  - L₁L₂ refit with L_local + λ·L_roll,full (K-step rollout through the quantized model, gradients through the state updates).
- Gradients through dynamic activation quantization use the straight-through estimator.
- Training prompts are drawn from mjhq1k[600:1000] (untouched); design uses the `design` prompts.
- Evaluation:
  - MJHQ-64 (mjhq1k[0:64], the established SOTA-table set): PSNR, LPIPS vs bf16, and ImageReward, paired vs base;
  - plus mjhq1k[288:352] (untouched) as the confirmation set.

## Positioning (recorded for the paper, not tested here)
- CTEC: rollout distillation.
- MPQ-DMv2: low-rank plus temporal distillation.
- EfficientDM and QuEST: quantization-aware fine-tuning lines.
- Ours optimizes only the existing low-rank branch of a frozen PTQ model, in minutes of GPU time, with unchanged deployment (same shapes, dtype, and kernel path).

---

## Amendments
(dated, appended before the affected arms run)
