# Anchor-drift calibration of the low-rank branch (ad1) — pre-registration

Written 2026-10-30, **before any check, training step or generation of this pilot**. Branch `anchor-drift-pilot` from local-error-refresh-pilot
1b00a67 (ler1 stopped at E1). Code `boa_dit/anchor_drift/run.py`, report `scripts/make_anchor_drift.py`, results `results/anchor_drift/`, checkpoints
and latents `ckpt/anchor_drift/` (not in Git). Stage tag **ad1**. Specification: the user's proposal "calibrate the branch for the anchored predictor"
(2026-10-30) and the plan agreed in this session (cap 2.0 A30-h). Amendments below the marker, dated, before the affected result.

## 1. Question
Output anchoring with reuse (aa1 C / AC) uses Q̃_t = Q_θ(x_t) − Q_θ(x_a) + F(x_a), whose error is e_t − e_a, while every earlier branch refit minimized
Σ‖e_t‖². Does refitting the existing rank-8 carry against the **anchored error drift** improve live anchored sampling compared with the same refit against
ordinary per-step reconstruction, with feedback off and on? Does feedback help once the branch is calibrated for the anchored predictor?
Not tested: W8 anchors, kernels, speed, novelty (literature check below is limited).

## 2. Frozen setup
- Base: unrefitted common smoothed rank-8 W4 base (`ckpt/ckpt_deflate_heig_r8_w4_s/model.pt` f017af87…), original carry
  `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` (b4a537b0…), smoothing (7ee15c0c…), static single8 tables (b73030de…); JointQuant `static`
  (feedback off) or `fb` (DCT-4 feedback), unchanged. Only the 112 carries (V [d_in, 8], δ [d_out, 8], fp32 masters, exported fp16) are trained, via
  `followup.carry_opt.Carry` exactly as in the joint pilot.
- Anchoring: K = 4 (anchors at i = 0, 4, …, 48), reuse on guided channels 0–3, FP teacher `load_dit`. Sampler for every evaluation: aa1
  `activation_anchor.run.sample` (kind `corr` = anchored, `plain` = no anchors), DDIM-50, CFG 1.5, fp16 autocast, fp32 sampler.
- Existing local controls (joint pilot, not retrained): **C_L** = local objective, feedback off; **E_L** = local objective, feedback on;
  `ckpt/dit_joint/carry_{C_L,E_L}_u200.pt` (sha256 ac20161034dbae00…, 77b5d0ac43b9ed3e…). Training config: 200 updates, Adam lr 1e-4, batch 4 images
  (8 CFG rows), training seeds 60100–60163, draw plan `dit_joint.draws(200)` (generator seed 0), loss normalized by `norm.pt` (656ddff9…) `loc[i]`,
  states from each student's own plain trajectory (`dit_joint.prefix`), feedback state detached at every temporal update.

## 3. New arms (trained here; identical config except the objective and the state source)
- **C_D** (feedback off) and **E_D** (feedback on). Update u uses the joint pilot's draw u: the same 4 training seeds; drawn step i0 → anchor
  a = 4·⌊i0/4⌋, window t ∈ {a+1, …, min(a+3, 49)}.
- States: the CURRENT student's own **anchored** trajectory, no gradients: from noise to a with cheap calls every step and FP anchors with reuse (training
  path: `dit_joint.call_model` / `guided_eps` / `ddim_step` with explicit feedback state); at a, cheap call (state_a), FP call, d_a = Q(x_a) − F(x_a), step
  with F(x_a); for t in the window, cheap call (state_t), FP call F(x_t) (training-only teacher), step with Q(x_t) − d_a. Latents and feedback states are
  recorded and treated as fixed inputs.
- Loss (gradient pass, each call checkpointed, `call_model` with the recorded state):
  L = mean_t ‖[Q_θ(x_t) − Q_θ(x_a)] − [F(x_t) − F(x_a)]‖² (sum over the 4 guided channels and pixels, mean over the 4 images and over window steps)
  / loc[a]. Uniform weights within the window; the per-window normalizer loc[a] is the joint pilot's local normalizer at the anchor step, so loss scales are
  comparable with C_L / E_L. **Gradients flow through both Q_θ(x_t) and Q_θ(x_a).**
- Snapshots u0, 50, 100, 200; resume file every snapshot. **Evaluated checkpoint: u200 for all four refits** (no selection; the joint pilot's snapshot
  selection used 60200–60231, overlapping this pilot's development seeds).

## 4. Checks (stage `checks`; failures block training)
- D1 training-path anchored prefix restored from a saved state continues bitwise (save at a = 12, continue to 24, vs uninterrupted).
- D2 gradient reaches the anchor term: on one draw at u0, the carry gradient with Q_θ(x_a) detached differs from the full gradient (rel. L2 of the difference
  ≥ 1e-3) and the gradient of the anchor term alone is nonzero and finite.
- D3 checkpointed vs non-checkpointed gradient (1 window, 2 images): forward bitwise, gradient rel. L2 ≤ 1e-4.
- D4 one quantizer update per site per cheap call in the training pass (prefix + window + gradient pass counts as expected).
- D5 untrained export (u0) equals the base carry bitwise; evaluation sampler with the base carry reproduces aa1's C / AC dev latents on 60200–60215
  (batch 16) bitwise (`ckpt/activation_anchor/checks_dev_latents.pt`).
- D6 reported only: training-path anchored trajectory vs evaluation sampler endpoint rel. L2 (joint pilot's plain-path value was 0.026).
- Profile: seconds per update for C_D and E_D (10 updates each, not kept) → budget reforecast.

## 5. Evaluation
Arms (anchored, K = 4): **C0** (base, fb off), **AC0** (base, fb on), **C·L** (C_L carry, fb off), **AC·L** (E_L, fb on), **C·D** (C_D, fb off),
**AC·D** (E_D, fb on). Unanchored dev-only references: Q0, A0, Q·L, A·L, Q·D, A·D. FP50 reference.
- **Development** 60208–60223 (16, batch 16; inspected in ler1, exploratory): all arms. Diagnostics on 60208–60215 (batch 8) for the six anchored arms on
  their own trajectories with same-state FP calls (diagnostic only): held-out raw local energy Σ‖e_t‖², corrected (drift) energy at non-anchor steps
  Σ‖e_t − e_a(t)‖², their ratio, window cross term Σ⟨e_t, e_a(t)⟩ and uncentered correlation, anchor identity residual.
- **Confirmation** 97100–97163 (fresh; ler1 never opened it; re-reserved for ad1), two batches of 32, opened once: C0, AC0, C·L, AC·L, C·D, AC·D, FP50.
  Runs unless a drift arm is non-finite or its dev anchored error exceeds 2× its local control (divergence guard; reported).
- Metrics: per-image endpoint latent rel. MSE vs FP50, PSNR, LPIPS-VGG. Paired image bootstrap 10,000 draws, numpy seed 0, ratio of mean errors.
- **Co-primary (objective under anchoring):** C·D / C·L (feedback off) and AC·D / AC·L (feedback on), each read separately — useful: ratio ≤ 0.95, CI upper < 1,
  ΔLPIPS CI not entirely above 0; smaller positive: CI upper < 1 otherwise; harmful: CI lower > 1; else inconclusive. No multiplicity correction; reported as two
  separate readings, not a joint pass.
- **Secondary:** AC·D / C·D (feedback after drift calibration; same reading rule), AC·L / C·L, C·L / C0, AC·L / AC0, C·D / C0, AC·D / AC0; objective × feedback
  interaction [E(C·L) − E(C·D)] − [E(AC·L) − E(AC·D)] with CI.
- Allowed outcome, recorded not gated: a drift arm may have worse unanchored (dev) error but better anchored error.
- One training seed per arm; the image bootstrap does not measure training variance. 64 images may be inconclusive; no additions.

## 6. Budget
Cap **2.0 A30-h**, ledger tag `ad1` (checks, profile, training, dev, diag, conf, failures). Estimates: checks + profile 0.15, training C_D ≈ 0.35 and E_D ≈ 0.5
(rollout-arm cost in the joint pilot: 6.2 / 9.2 s per update), dev + diag 0.15, confirmation 0.35. Confirmation reserve 0.40 h held by every earlier guard.
Cut order: diagnostics, unanchored dev references.

## Stages (`python -m boa_dit.anchor_drift.run <stage>`)
`checks` → `train C_D` → `train E_D` → `dev` → `diag` → freeze (commit `frozen.json`) → `conf` → report. Commit/push after every stage.

---

## Amendments and results
(appended below, dated)
- 2026-10-30, checks (0.102 h) + profile: D1–D5 pass (restore bitwise; anchor-term gradient rel. L2 3.31 of the full gradient; checkpoint forward bitwise,
  gradient identical; 8 = 2 × 4 quantizer calls per site in the gradient pass; u0 export = base; base carry reproduces aa1 C / AC bitwise; carry swap
  reproduces joint C_L / E_L / A latents bitwise). D6 training-path vs eval-sampler endpoint rel. L2 0.042 (AC0, 4 images; reported, joint plain path 0.026).
  Profile 7.26 s/update (C_D), 10.67 (E_D), peak 16.0 GiB → training ≈ 0.40 + 0.59 h.
- 2026-10-30, train C_D (0.36 h): 200 updates; training-draw drift loss 0.54 → 0.35 (first/last 10), window local loss 0.88 → 5.3 (absolute error grew ~6×; drift loss is blind to window-constant error — recorded, not gated, per §5).
- 2026-10-30, train E_D: 200 updates; see train_E_D.json (drift loss first/last and window local loss in REPORT).
