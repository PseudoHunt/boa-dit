# DiT activation-feedback + weight-correction pilot (protocol, 2026-10-29; implementation decisions frozen before any quality result)

**Status: REGISTERED, NOT STARTED.** It is the active work of branch `dit-joint-feedback`. Every other experiment is parked (handoff/PARKED_WORK.md).
Stage tag `j1`. Outputs under `results/dit_joint/`, checkpoints under `ckpt/dit_joint/`. Code goes in a new module `boa_dit/dit_joint/`.
The user's full specification (Part C of the 2026-10-29 instruction) is reproduced in §C below. §D lists the implementation decisions fixed now, from an audit of the existing code.

## Research question
Does calibrating the existing rank-8 weight-correction branch **with** compressed temporal activation feedback give better full-sampling fidelity than
(a) either component alone, and (b) training the branch without feedback and enabling feedback afterwards?
This is a new integrated experiment. The parked PixArt pilot does not test it, and the claim is unproven.

## §D Implementation decisions (audit of temporal-oracles @ 216e4e6 and e3-confirm @ 21f0c55, 2026-10-29)
D1. **Existing activation implementation.** `boa_dit/temporal/dit_act.py::Quantize`:
- static per-tensor asymmetric A8 tables; `single8` = one encoding per site across all 50 steps (`dit_static_b_single8.pt`, calibrated on FP trajectories of seeds 1000–1031);
- applied as a **forward pre-hook** on the 112 block-linear inputs;
- smoothing only at `mlp.fc1` and `mlp.fc2`: it quantizes x/s and feeds q·s into the **unchanged** linear (fake-quant with per-channel effective steps Δ·s_c);
- DCT-4 feedback = `token_basis("dct4")`: an orthonormal 2×2 lowest DCT-II basis on the 16×16 token grid, row-major patch order. `fb_step` details:
  - **state:** 4 fp16 coefficients per channel, per site and per batch row (CFG halves are separate rows), reset at step 0;
  - **update rule:** u = clip-guarded x̃ + s_prev, ũ = clip(u), q = Q(ũ), r⁰ = ũ − fp16(q), r = clamp(r⁰, ±Δ/2), s = Uᵀr stored fp16;
  - clip error is never fed back.
D2. **Carry location.** The carry (`deflate.carry.install` / `followup.carry_opt.Carry`) is a forward hook on the same linear, and forward hooks receive the pre-hooked input.
  So in the existing combination **the carry sees the quantized (consumed) activation**. This choice is kept in every arm.
D3. **Deployability of smoothing with W4.** Combining the fake-quant smoothing with the existing W4 weights gives W_q·diag(s) acting on integer codes, which is **not** the packed W4 representation.
  **Decision:** build **one common smoothed rank-8 W4 base**.
  - Fold s into the FP weights at fc1/fc2 (W̃ = W·diag(s), x̃ = x/s; the other 82 sites are unchanged).
  - Run the existing heig recipe (`deflate.quantize_dc --mode heig --r 8 --bits 4`, calibration seeds 1000–1031, the established weight-calibration set) on the reparameterized model.
    This gives new W4 codes/scales and a rank-8 carry, with V in the smoothed input space.
  - Store it as `ckpt/ckpt_deflate_heig_r8_w4_s/` and freeze it for all arms. Its provenance (s file sha, builder commit, seeds) goes in `results/dit_joint/base.json`.
  - **Check first:** the FP reparameterization (folded s at fc1/fc2, no quantization) must reproduce FP outputs. Tolerance: fp32 forward rel. diff ≤ 1e-5 on 4 seeds × 3 steps.
  - The original heig-8 W4 + carry is kept unchanged as the **W4A16 control** (context, not a matched feedback baseline).
D4. **Activation ranges** are recalibrated **once**:
  - common smoothed base + carry, feedback OFF, `single8` recipe (`dit_act.calib` / `ACT` alphas and clipping rule);
  - trajectories of the **common base** (not FP), seeds **60000–60031**, both CFG halves, all tokens.
  Tables are frozen in `ckpt/dit_joint/static_single8.pt` (sha in base.json). Rounding and clipping are logged separately.
D5. **Feedback in training.** `fb_step` is reimplemented as a pure function (state in, state out), so gradient-checkpoint recomputation cannot advance the state twice.
  - Forward values must equal `Quantize.fb_step` bit for bit (check 2).
  - **Backward:** identity STE through rounding; the stored residual is detached at every temporal update; full gradients flow through the state update and the denoiser along the rollout latent.
    This is a truncated surrogate, not an exact derivative, and it is identical in E_L and E_R.
D6. **Training-window state.** For every update:
  - run the **current student** from the original noise to the window start without gradients, carrying its actual feedback state (or none, in feedback-off arms);
  - detach the latent and the state at the boundary;
  - differentiate through the K-step window.
  The same prefix policy applies to the feedback-off arms. Any prefix cache must be a dated amendment with a refresh policy and consistency checks.
D7. **FP teacher:** no activation quantization, no feedback, same start state, K-step continuation. Loss = local (step-start ε) + λ·full-latent K-step. No trajectory-anchored target.
D8. **Loss normalizers:** computed once from the common untrained base, feedback OFF, on 16 of the training seeds, per start step.
  The same constants are used in every arm. A floor of 1e-6 × the mean normalizer is declared, the denominators are audited, and gradient norms are printed per step at initialization.
D9. **Schedule.**
  - Adam, seed 0, lr 1e-4, K = 5, λ = 0.5, B = 4 (or the same accumulation in every arm if memory demands it).
  - Start steps are drawn uniformly over 50, with windows shortened at the end. The same (seed, start-step) draw sequence is used in all four runs.
  - **200 optimizer updates** per run, with checkpoints at 0, 25, 50, 100 and 200.
  - **lr fallback:** if lr 1e-4 is clearly unstable on development data, one dated common screen at 1e-5 restarts **all** affected runs.
D10. **Tolerances, declared now:**
  - check 1, A8 and feedback disabled == W4+carry reference: bitwise in the deployed dtype path;
  - check 2: bitwise vs `Quantize` on matched inputs;
  - check 3, trainable carry and its export: bitwise;
  - check 4, prefix + restored state == full rollout: ≤ 1e-5 fp32 and ≤ 1e-3 fp16 relative;
  - check 5, checkpointing on/off: forward bitwise, gradients ≤ 1e-4 relative in fp32, and the state-update count asserted;
  - check 7, save/reload: bitwise; resume restores optimizer/RNG/draw cursor (asserted).
  A failed tolerance is amended only after diagnosis, transparently.

## §C Specification (user, 2026-10-29), condensed and authoritative
- **Bed:**
  - DiT-XL/2 256, DDIM-50, the repository CFG convention at w = 1.5;
  - frozen W4 block linears plus the existing rank-8 carry (the common base, D3);
  - static asymmetric per-tensor A8, one encoding per site;
  - DCT-4 activation residual state exactly as D1;
  - the same 112 sites, with the embedding/modulation/final-layer exclusions;
  - only the carry factors are trained.
- **Excluded:** DNS, PixArt, dynamic A4, any rank increase, INT4 residual compression, new DCT sizes, bit sweeps, kernels.
- **Arms:** four training runs, eight configurations, all starting from the base's original rank-8 factors.

  | ID | carry objective | feedback while training | feedback at evaluation |
  |---|---|---|---|
  | A | untrained | — | off |
  | B | untrained | — | DCT-4 |
  | C_L | local | off | off |
  | D_L | the same checkpoint as C_L | off | DCT-4 |
  | E_L | local | DCT-4 | DCT-4 |
  | C_R | local + 5-step rollout | off | off |
  | D_R | the same checkpoint as C_R | off | DCT-4 |
  | E_R | local + 5-step rollout | DCT-4 | DCT-4 |

  A D arm never uses a different checkpoint from its C arm. E vs D differs only in feedback during training.
- **Correctness checks 1–9 (§C5)** gate training. A failure blocks training, not the idea; failure records are kept.
- **Seeds:** see `handoff/SEED_REGISTRY.md`; all verified unused 2026-10-29.
  - calibration 60000–60031;
  - training 60100–60163;
  - development 60200–60231;
  - **sealed** pilot evaluation 61000–61127 (128), opened once, after the selections are frozen.

  Identical labels and noise are used across arms.
- **Machine controls** (this machine, first): FP; the common base W4+carry with A8 off (W4A16 control); the common W4A8 base with feedback off / on.
  Historical numbers are context only.
- **Selection:**
  - **primary mechanistic comparisons:** fixed 200-update snapshots;
  - **secondary (selected on development 60200–60231 only):** C and E arms choose among {0, 25, 50, 100, 200}.
    - C picks with feedback off, and D uses exactly C's choice.
    - E picks with feedback on.
    - An untrained winner is labelled untrained.
  - All selections are frozen before the sealed set is opened.
- **Evaluation** (sealed 128 seeds, all eight configurations + FP):
  - final latent rel. MSE, PSNR and LPIPS vs FP, with per-image records;
  - paired differences: relative mean-MSE change with paired bootstrap CI (2,000 resamples), win count and median;
  - a fixed diagnostic subset (seeds 61000–61031) with prediction error and temporal coherence C, I, C/I, using the E3 formula on FP-trajectory states;
  - fixed representative seeds (61000, 61042, 61085) plus labelled worst cases;
  - state memory, peak memory, inference time with diagnostics off, training and calibration compute, denoiser calls (50).
  - "Better generation" is never claimed from paired MSE, and FID is not required for this pilot.
- **Predeclared contrasts:**
  - feedback effect: B vs A;
  - carry without feedback: C_L / C_R vs A;
  - stacking: D vs its C, and D vs B;
  - **primary joint-training contrast: E_R vs D_R (fixed updates)**;
  - local integration control: E_L vs D_L;
  - rollout with feedback: E_R vs E_L;
  - selected-checkpoint comparisons are secondary.
- **Decision rule (pilot prioritization; not a publication threshold):**
  - **useful signal:** E_R ≥ 5% lower mean latent MSE than D_R with the paired CI below 0, E_R also better than A and B, and no clear LPIPS regression;
  - **crediting the rollout term:** E_R also beats E_L;
  - **otherwise inconclusive.**
- **Narrow interpretations:**
  - B fails → feedback did not transfer to this base.
  - D > B, C but E ≯ D → stacking without a joint-training benefit.
  - E_L ≈ E_R → feedback-aware calibration without rollout-specific credit.
  - E_R > D_R, E_L → promising; replicate seeds and confirm generation next.
  - All flat → this pilot did not establish integration. That does not show that no combination can work.
- **Stop after the pilot report.** No PixArt, DNS extension, FID campaign or other architecture follows automatically.

## Profile (fill in on the target machine before training; no extrapolated A30 numbers)
GPU model, seconds per update per arm (prefix included), peak memory, projected total. The record goes to `results/dit_joint/profile.json`.

---
## Amendments
(dated; appended before the affected step runs)

### 2026-10-29 15:10: run on the current VM; implementation details (before any pilot output)
- **Machine.** The user elected to run the pilot directly on the current VM ("directly move to the actual DiT experiments"). The regeneration drill is skipped, and the existing on-disk artifacts are used:
  - DiT weights;
  - the original heig-8 W4 + carry (W4A16 control);
  - `dit_smooth_s.pt` (sha in base.json).

  The common smoothed base and the A8 tables are built fresh by the pilot.
- **Smoothing reparameterization.** W̃ = W·diag(s) at mlp.fc1/fc2, plus a permanent forward pre-hook x̃ = x/s, registered before every other hook.
  So the builder's Gram hooks, the A8 quantizer and the carry all see x̃. The divided input is cast back to the incoming dtype (fp16 under autocast).
- **STE.** The quantizer's forward returns the deployed value bit-exactly (custom autograd function); backward is identity w.r.t. the site input, including through the clip interval.
  The feedback residual is computed without gradients, i.e. detached at each update.
- **Training-path sampler.** `followup.carry_opt.guided_eps` + `ddim_step` (the E3 differentiable re-implementation, ≤ 2e-4 from `ddim_sample` over 5 steps in E3).
  Evaluation uses `sample.ddim_sample`. Check 4 reports the difference.
- **Normalizers.** From the untrained common base with A8 static (feedback off), on training seeds 60100–60115, using its own prefix states, per start step.
- **Checkpoint-safety.** The feedback state is passed as explicit arguments to each checkpointed denoiser call. Check 5 asserts bitwise forward agreement, gradients ≤ 1e-4, and equal end-of-window states with checkpointing on and off.

### 2026-10-29 15:15: check 5 size (after the first checks run hit OOM, before any training)
The first `checks` run hit OOM at check 5: the no-checkpoint side of a 5-step window stores every call's activations for 8 rows.
Check 5 now uses a **2-step window on 2 seeds** (4 rows). It still asserts:
- bitwise forward equality;
- gradients ≤ 1e-4 relative;
- equal end-of-window feedback states;
- the call counts (2 without, 4 with checkpointing).

Checkpoint safety does not depend on the window length. Training rollouts always use checkpointing; local arms make a single call.
