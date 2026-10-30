# Activation feedback + measured anchoring (aa1): integration / accuracy pilot — pre-registration

Written 2026-10-30, **before any check or generation of this pilot**. Branch `activation-anchor-pilot` from weight-anchor-shared
80db0aa79dff22e85ffe577c579b622493299ae7 (contains the wa8 anchor loop and, blob-identical to dit-joint-feedback 2c80172, `boa_dit/dit_joint/run.py`,
`temporal/act.py`, `temporal/dit_act.py`, `followup/carry_opt.py`). Code `boa_dit/activation_anchor/run.py`, report `scripts/make_activation_anchor.py`,
results `results/activation_anchor/`, latents `ckpt/activation_anchor/` (not in Git). Stage tag **aa1**. The user's handoff (2026-10-30) is the
specification; amendments below the marker, dated.

## 0. Status of the superseded shared-weight phase (ws1)
Already complete before this handoff arrived: confirmation and REPORT.md at weight-anchor-shared 80db0aa (one commit after the inspected 123b695).
No ws1 process was running (GPU idle, no python jobs); nothing was cancelled because nothing was queued. Spend 0.367 A30-h. Parked by user
decision (not a failed read) in CLAUDE.md and `handoff/PARKED_WORK.md` on this branch.

## 1. Is A+C new?
All 15 remote branches (fetched 2026-10-30) and all local worktrees were searched for files combining anchor reuse with DCT-4 / static_single8 /
feedback. None exists: `results/dit_joint` combined feedback with weight-branch refitting (its "trajectory-anchored target" is a registered,
unimplemented training target); wa1 / wac1 / wa8 / ws1 used high-precision activations. **A+C is new.**

## 2. Frozen setup
- **Cheap model:** the existing unrefitted common smoothed rank-8 W4 base — `ckpt/ckpt_deflate_heig_r8_w4_s/model.pt` (sha256 f017af87…7ced) + its original
  carry `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` (sha256 b4a537b0…edd5; copied from the dit-joint worktree, not committed) + smoothing
  `ckpt/temporal/dit_smooth_s.pt` (7ee15c0c…db26) — with static per-tensor A8 tables `ckpt/dit_joint/static_single8.pt` (b73030de…dd67) applied by
  `dit_joint.run.JointQuant` in mode `static` (Q) or `fb` (DCT-4 temporal feedback, A), exactly as built by `dit_joint.run.student(dev, None, mode)`.
  No training, rebuild, range retuning, guard change, W8 or shared-weight anchor.
- **Teacher / anchor / scorer:** the FP model `load_dit` (FP weights, high-precision activations), same fp16 autocast forward and fp32 sampler as all
  validated runs. One teacher for anchors and scoring.
- **Bed:** DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5 (ε channels 0–2; channel 3 from the cond half; channels 4–7 unused by DDIM), guided output
  `weight_anchor.run.guided`, sampler loop as `weight_anchor.w8.sample_arm` (the DDIM op order validated bitwise in wac1/wa8).
  Batch 32 for confirmation; development uses its 16 seeds as one batch of 16 (all dev arms at batch 16).
- K = 4 (anchors at calls 0, 4, …, 48: 13), β = 1. No K / β / feedback-size / checkpoint selection.

## 3. Arms (each on its own trajectory)
| ID | DCT-4 feedback | anchor prediction used | error reused between anchors | cheap calls | FP calls |
|---|---|---|---|---|---|
| Q | off | no | no | 50 | 0 |
| A | on | no | no | 50 | 0 |
| C | off | FP, every 4 calls | yes | 50 | 13 |
| AC | on | FP, every 4 calls | yes | 50 | 13 |
| C_anchor_only | off | FP, every 4 calls | no | 50 | 13 |
| AC_anchor_only | on | FP, every 4 calls | no | 50 | 13 |
| FP50 | — | FP at every step | — | 0 | 50 |

- Every anchor-containing arm runs the cheap model at all 50 steps (its feedback state advances exactly once per step, including anchors) and FP at
  the 13 anchors, on exactly the same x_a, timestep, labels tensor and CFG layout. d_a = f_cheap(x_a, state_a) − f_FP(x_a) on channels 0–3 (fp32).
  The anchor step uses the FP output. At non-anchor t: C/AC use f_cheap(x_t, state_t) − d_a (channels 0–3); anchor-only controls use the raw cheap output.
- In AC, d_a is measured with feedback on, on AC's own trajectory. Feedback state (`jq.state_in/state_out`) and the cached anchor error reset at the
  start of every sampling batch; the feedback state also resets at t = 0 (JointQuant's eval rule). CFG rows/samples stay separate (per-row state).
- FP has no quantizer state and no hooks; cheap and FP are distinct module instances (asserted: no pre-hooks on FP block linears, disjoint parameters).

## 4. Correctness gates (stage `checks`; failures block results)
- G1: Q, A and FP50 on seeds 61000–61031 at batch 32 reproduce the joint pilot's cached latents bitwise (`ckpt/dit_joint/eval_latents.pt` "A" = static,
  "B" = DCT-4, both base carry; `fp_sealed.pt`). These seeds were opened in the joint pilot; here they are used only for this bitwise test, no metric.
- G2: correction disabled (d_a multiplied by 0) reproduces the anchor-only control (C and AC, dev seeds 60200–60215, batch 16).
- G3: K = 1 — corrected and anchor-only arms (feedback on) equal FP50 bitwise.
- G4: no-anchor mode (anchor arm with no anchors) reproduces Q (feedback off) and A (feedback on) bitwise.
- G5: identical deterministic low/high (cheap := FP) gives exactly zero d_a and the FP50 trajectory.
- G6: in an instrumented AC run, at each anchor the cheap and FP calls receive identical x, t and labels tensors; counters are 50 cheap / 13 FP for every
  anchor arm, 50 / 0 for Q and A, 0 / 50 for FP50; the feedback state advances exactly once per cheap call (JointQuant per-site call counter = 50 per
  trajectory) and never on FP calls.
- G7: save/resume — an AC trajectory stopped after step 30, its state (latent, CUDA/CPU RNG state, step index, feedback state, cached d_a and anchor
  index) saved to disk with torch.save and restored into a fresh sampler call, finishes bitwise equal to the uninterrupted run.
- G8: all outputs finite; channels 4–7 zeroed on the Q path leave the endpoint unchanged (variance-channel handling); quantizer arithmetic is the
  unchanged JointQuant (G1 confirms bitwise).

## 5. Development (60200–60215, already inspected; exploratory)
All seven arms, same metrics as confirmation. Nothing is selected.

## 6. Confirmation (fresh 96100–96163)
Verified 2026-10-30 absent from the seed registry and from `*.py/*.md/*.csv` on all remote branches and local worktrees; 95100–95163 not used.
All seven arms on the same 64 seeds (two batches of 32), identical labels/noise; FP50 generated fresh here (no historical reference reused).
- Metrics: per-image endpoint latent rel. MSE vs FP50, PSNR and LPIPS-VGG (sd-vae-ft-ema decode; `weight_anchor.controls.metrics`).
- Statistics: paired image bootstrap, **10,000 draws, numpy seed 0** (`weight_anchor.shared.contrast`); relative change = ratio of mean errors
  recomputed per draw; paired mean and median differences; win counts; PSNR and LPIPS differences with CIs.
- **Primary: AC vs C.** *Useful additional feedback signal:* ratio E(AC)/E(C) ≤ 0.95, CI upper bound < 1, and the ΔLPIPS (AC − C) CI not entirely above 0.
  *Smaller positive:* CI upper < 1 but ratio > 0.95 (or LPIPS clearly worse, reported). *Harmful:* CI lower > 1. Otherwise *inconclusive* (never equivalence).
- **Secondary:** A/Q, C/Q, AC/Q, AC/A, C/C_anchor_only, AC/AC_anchor_only, AC_anchor_only/C_anchor_only.
- **Interaction:** I = [E(C) − E(AC)] − [E(Q) − E(A)] with bootstrap CI (positive: feedback's absolute benefit grows with anchoring); also the relative
  form E(AC)/E(C) ÷ E(A)/E(Q) with CI. No redundancy inference from AC < C alone.
- Latency/memory: per-arm wall time (CUDA-synchronized) and peak allocated memory recorded during confirmation, diagnostics off, labelled
  simulation (fake-quant, fp32 weights); not a deployable speed. FP diagnostic calls never appear in inference counts.

## 7. Mechanism diagnostic (optional if the cap binds; 8 dev seeds 60200–60207, batch 8)
For Q, A, C and AC on their own trajectories: at every one of the 50 states, the same-state FP prediction (diagnostic-only FP calls at non-anchor steps;
never used live; no extra cheap forwards, so no feedback state is touched twice). Recorded: raw cheap error e_t = f_cheap − f_FP (channels 0–3), and for
C/AC the actually used (corrected) error. At lags L ∈ {1, 2, 4, 8}, pooled over all pairs (a, t = a + L), 0 ≤ a ≤ 49 − L, and all 8 images (sums over
pairs and images): E_t = Σ‖e_t‖², E_a = Σ‖e_a‖², X = Σ⟨e_t, e_a⟩, ρ = X / √(E_t E_a), Σ‖e_t − e_a‖²/Σ‖e_t‖² (image-bootstrap CI, 2000 draws),
identity residual |Σ‖e_t − e_a‖² − (E_t + E_a − 2X)| / Σ‖e_t − e_a‖², per-step energies. For C/AC: directly measured corrected-error energy at
non-anchor steps vs raw, and the anchor-structured identity (corrected error at t = e_t − e_{a(t)}). Feedback clip rates (JointQuant's counter of
representable-range clipping) are reported separately; residual-clamp and discarded-subspace energies are not measured by the frozen quantizer and are
reported as not measured. Within-arm traces describe each arm's own trajectory; cross-arm differences include state-distribution changes and are not
a causal decomposition.

## 8. Budget
GPU cap = min(1.5, 2.0 − 0.367) = **1.5 A30-h** in `gpu_hours.csv` rows tagged `aa1` (checks, dev, diagnostics, confirmation, failures).
Confirmation reserve 0.35 h held by every earlier guard; runtime stop before each confirmation batch. Estimates (joint pilot: static ≈ 1.6 s/img,
DCT-4 ≈ 2.6 s/img at batch 32; FP ≈ 0.6 s/img): checks ≈ 0.20 h, dev ≈ 0.10 h, diagnostics ≈ 0.06 h, confirmation ≈ 0.30 h. Cut diagnostics first.

## Stages (`python -m boa_dit.activation_anchor.run <stage>`)
`checks` → `dev` → `diag` (optional) → freeze (commit `frozen.json`) → `conf` → report. Commit/push after every stage. Stop after this pilot.

---

## Amendments and results
(appended below, dated)
