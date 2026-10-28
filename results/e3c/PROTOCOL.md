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

### 2026-10-28 16:40: H16 build note (before any H16 evaluation)
`quantize_dc --mode heig --r 16 --bits 4` completed all 28 blocks and saved `ckpt/ckpt_deflate_heig_r16_w4/model.pt` and
`results/deflate/quant/deflate_heig_r16_w4/carry.pt`. It then failed in its **post-hoc diagnostic**, which compares weight drift against
the uniform-W4 arm's checkpoint (`ckpt_uniform_w4_K1_lam1_brcfg/model.pt`, absent on this machine). That step only annotates the JSON.
The JSON was written by hand from the saved per-layer diagnostics, without the `drift_uniform` fields. Measured extra bits/param: **0.2963**
(heig-8: 0.1481). H16 itself is unchanged by this.

### 2026-10-28 17:05: Part 2 details (before any PixArt run)
- **Code.** `boa_dit/e3c/pixart_refit.py`.
  - Sampler: the deployed one (diffusers DPM-Solver++(2M), 20 steps, guidance 4.5, batch 1, seed = prompt index).
  - Rollouts restore the scheduler's multistep history (the previous x0 prediction, `lower_order_nums`, the step index) from the cached base trajectory.
    Gradients flow through every state update and through the history the rollout itself produces.
- **Checks (must pass before training).** All listed in `check`:
  - own loop == pipeline, bitwise;
  - untrained trainable layer == the deployed `QLinear`, bitwise;
  - untrained export == the base tensors, bitwise;
  - a K-step rollout from cached states ≤ 1e-3 relative error;
  - a nonzero gradient at block 0.
- **Arms.**
  - P0 = base.
  - P1 = L_local (time-matched).
  - P2 = L_local + λ·L_roll,full, with K = 5 and B = 1 prompt (two CFG rows).
  - Both start from the base branch, use Adam and training seed 0. One training run per arm (cost), so training-run variance is not measured on PixArt.
- **Training data.** Base W4A4 trajectories of mjhq1k[600:664] (64 prompts, untouched). The loss normalization constants are base values on the first 4 training prompts.
- **Design.** design[24:32] (8 prompts; E2 used design[0:24]).
  - lr ∈ {1e-4, 3e-4}, picked by P1's design-local loss after T/4.
  - λ ∈ {0.5, 2}, picked by P2's endpoint latent rel. MSE vs bf16 after T/4.
- **T_train.** 1.0 h per arm. If profiling shows fewer than 150 P2 updates in 1.0 h, T_train is raised to give 150 P2 updates, capped at 2.0 h.
  The value is written to `plan.json` before design.
- **Evaluation.**
  - MJHQ-64 (mjhq1k[0:64]). Base images (`e1_w4a4_base_screen`) and bf16 batch-1 references (`bf16_b1`) are reused: same generator, seeds, batch and model files.
  - Untouched confirmation set mjhq1k[288:352]: new bf16 references and base images.
  - Metrics: PSNR, LPIPS-VGG and latent rel. MSE vs bf16, plus ImageReward v1.0. Paired bootstrap over prompts vs P0, and P2 vs P1.
- **Reads.**
  - P1 > P0: refitting the branch helps.
  - P2 > P1 (CI excludes 0 on latent): credits the trajectory objective on this bed.
  - ImageReward is reported, but no generation-quality claim is made from it alone.

### 2026-10-28 18:05: Part 3, PixArt pilot (user specification, run **after** Parts 1–2 and the DNS work; registered before any pilot step)
User text (verbatim):
- "Pre-registered check (no training): the exported branch reproduces the corrected H-SVDQuant-r4 base bit-exactly in sampling, as E3's check did for heig-8."
- "Profile one update at batch 1, K = 3 rollout steps, gradient checkpointing per block, straight-through on the activation quantizer with stop-gradient on any residual; record seconds per update and peak memory. If an update exceeds 3 s or memory exceeds 20 GiB, drop to K = 2."
- "Two arms, 0.4 h training each, same seed, design prompts: C1 local per-step reconstruction loss; C2 local plus K-step rollout loss against the FP teacher from the same cached student state, λ = 0.5 (E3's value). Train L₁ and L₂; D and the codes stay frozen."
- "Evaluate base, C1, C2 on mjhq1k[256:288] (untouched): latent, PSNR, LPIPS with paired CIs; sec/image unchanged by construction."

**Read (verbatim, written before step 3).** "The pilot passes iff C2 beats the base by ≥ 0.3 dB PSNR with the CI excluding 0 and C2 beats C1 on latent; a C1-only gain means the H-SVDQuant branch was under-fit, which is still a result but not the rollout claim. Fewer than 200 updates per arm, or a failed step-1 check, makes the pilot uninformative rather than negative; report it as such."
- "C2 beats C1 on latent" is read as C2's latent rel. MSE being lower, with the paired CI excluding 0.

**Implementation choices (not specified by the user).**
- Code: `pixart_refit.py --pilot`.
- Training states: base W4A4 trajectories of all 32 `design` prompts. Same seed 0; lr 1e-4 (E3's frozen value); Adam; sampler DPM-Solver++(2M), 20 steps, g 4.5.
- If K = 2 is also over a limit, K = 2 is kept and the overrun is reported.
- STE: forward = the deployed quantizer output (bit-identical); backward = identity. This equals x + stop_grad(Q(x) − x).
- The step-1 check is `check`, extended to sampling with the **exported** untrained branch through the deployed `QLinearFB` path, which must be bitwise equal to base sampling.
- Evaluation: bf16 references and base images for mjhq1k[256:288] are regenerated with the same generator (batch 1, seed = prompt index).
- **Provenance note.** mjhq1k[256:288] was the confirmation set of follow-up E2 (tile-balanced rounding, branch `followup-act`).
  It was not used by any branch-refit choice, so it is untouched for this pilot's purposes. It is not unused overall.
