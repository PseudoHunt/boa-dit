# dar1 — anchor-aware refit of the DiT low-rank branch (calibrate for the anchored + bridged sampler) — pre-registration

Written 2026-10-31, **before any dar1 measurement**. Specification: the user's handoff "SANA repeatability, anchor-aware refit, and W8 anchor
spacing" (2026-10-31), task 2. The SANA track was parked by a user decision on 2026-10-31; its 0.6 h allocation is not reassigned.

Branch `dit-anchor-refit` (worktree /home/boa-dit-dar), from `activation-bridge-overnight` 5023a0a (verified head; abn1 harness, dit_joint,
anchor_drift unchanged). Code `boa_dit/anchor_refit/run.py`. Tag **dar1**. Planning cap **2.5 A30-h** within the handoff total of 4.0, checked over both
handoff ledgers (`scripts/h1031_budget.py`: sr1 + dar1 + w8s1; the 0.5 reserve is usable only through the total). Amendments below, dated.

## 1. Fixed inference configuration (the validated abn1 CB / LCB configuration; `activation_bridge.run.sample`)
- Model: DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5 (CFG on the first 3 eps channels; channel 3 from the cond half; d on the 4 eps channels), fp16 autocast,
  fp32 sampler.
- Cheap model **W4A8 static**:
  - common smoothed rank-8 W4 base (f017af87…);
  - static single-table A8 (b73030de…), `JointQuant` mode `static`: per-tensor asymmetric, calibrated interval, `q_static` with hi_code qmax−1;
    no feedback, no guard path;
  - smoothing folded (7ee15c0c…).
  - **Branch** = the 112 rank-8 carries (V [d_in, 8], delta [d_out, 8]), installed as forward hooks (`deflate.carry`). Original carry b4a537b0…;
    archived local refit `carry_C_L_u200.pt` ac201610…
- FP anchors, K4 at 0, 4, …, 48:
  - at an anchor, cheap and FP are evaluated on the pre-correction latent and d = Q − F;
  - **bridge** x ← x − λ(d_c − d_a)·Σ_{j=1}^{k−1}(j/k)·b_{a+j}·Π_{m=j+1}^{k−1}A_{a+m}, with λ = 1;
  - then the DDIM step from the corrected latent uses the pre-correction FP prediction (the tested approximation, reproduced in training);
  - 50 cheap + 13 FP calls, 12 corrections; the final interval after 48 is uncorrected.
- Arms:
  - **B0** = original carry (abn1 CB);
  - **BL-arch** = archived local refit (abn1 LCB);
  - **BL** = new controlled local refit;
  - **BA** = anchor-aware refit;
  - all with the identical inference loop.
- Frozen: integer codes, scales, A8 tables, ranks and shapes, scheduler, anchors, λ. Only V and delta are trained (fp32 masters, exported fp16).

## 2. Data (audited 2026-10-31: no 124000–126999 use on any branch or worktree)
- Training: dit_joint TRAIN **60100–60163** (previously used for training; label: training). Draws are a fixed list (torch seed 0):
  (4 seed indices, window start a ∈ {0, 4, …, 48} uniform).
- Normalizers: TRAIN[0:16] (common base).
- Development / checkpoint selection: **60200–60231** (32; previously inspected; label: inspected development).
- Fresh confirmation: **124000–124127** (128), reserved now.
- Diagnostics: 60208–60215.

## 3. Window (both objectives share prompt/start draws)
- **Start state:** the cached common-base (B0) corrected-trajectory latent at anchor a, *before* a's anchor operations, from `sample(..., diag)`.
- **Boundary convention:** the window begins with the anchor measurement at a and applies **no bridge at a**, because the history d_{a−4} lies outside
  the window; a is treated like step 0. Bridges are applied at the internal anchor a+4 and the right-boundary anchor a+8.
- **Endpoint:** the corrected latent at a+8, after its bridge and before its DDIM step.
- **Terminal windows:** a = 44 covers 44…49 with an internal anchor 48 (one bridge), and its endpoint is the final sample x_50; a = 48 covers 48–49, no
  bridge, endpoint x_50. Late steps are kept, not dropped.
- **Calls:** a ≤ 40: 8 cheap + 3 FP, 2 bridges; a = 44: 6 + 2, 1; a = 48: 2 + 1, 0. Asserted.
- **Anchor measurements** use the *current* trainable cheap model and FP **at the student's current latent**. Teacher weights are frozen, and
  derivatives with respect to the latent are retained (checkpointed).
- **BA loss:** ‖x_end^student − x_end^FP‖² (sum over the latent, mean over images) / den[a].
  - x^FP is the FP DDIM continuation from the same start latent over the same transitions (no grad).
  - den[a] = mean over TRAIN[0:16] of the same quantity for the B0 window (positive, frozen, common base, per start bucket; floor 1e-6 × mean).
- **BL loss:** the existing per-step prediction reconstruction ‖Q(x_i) − F(x_i)‖² / loc[i] (dit_joint `norm.pt` loc, frozen), averaged over the window's
  transitions i ∈ [a, end), at the FP-continuation states x_i^FP of the same draw.
- **Shared settings:** dit_joint optimizer (Adam, lr 1e-4, betas default), batch 4 images. No feedback, regularizers or sweep.
- **STE:** the static activation quantizer passes gradients straight through (`dit_joint._Ident`). The sampler, bridge and teacher are smooth; the
  quantizer is not, and no exact gradient through it is claimed.

## 4. Checks (stage `checks`; all must pass before training)
1. Untrained differentiable inference vs validated B0 (abn1 `sample`, 4 TRAIN images, batch 4). Fixed tolerance: endpoint rel L2 ≤ 0.05 (the
   training path uses `ddim_step`/`guided_eps`, eval uses `ddim_sample`; ad1 measured 0.042 for its feedback arm). Step-1 latent rel ≤ 1e-4. Per-step
   divergence is reported.
2. Export ↔ install: for the untrained and a perturbed carry, the exported fp16 carry installed by `deflate.carry.install` reproduces the trainable
   (Carry-hook) single-call prediction bitwise, and `sample` with the exported untrained carry reproduces abn1 CB bitwise (110000–110031, batch 32).
   The archived C_L u200 reproduces abn1 LCB bitwise.
3. Zero update: export equals the original carry bitwise; tables unchanged; calls 50 + 13; output = B0.
4. Window: per-start call/bridge counts as §3. The a = 0 window prefix equals the full training-path rollout's step-8 corrected latent bitwise.
   Training resume (2 + 2 updates vs 4) restores parameters, optimizer and draws (bitwise expected; reported).
5. Gradients reach b0 and b27 carries; FP parameters get none; the latent derivative through the teacher is retained (gradient changes when the teacher
   input is detached).
6. Central finite differences, deterministic reduced case: quantizer `off`, fp32, batch 1, window a = 0 (2 bridges).
   - Directional derivative along a random carry direction and along a random start-latent direction; h ∈ {1e-1, …, 1e-5} × ‖θ‖.
   - Resolved iff some h gives rel. error ≤ 1e-3 with a neighbouring h ≤ 1e-2.
   - The STE-quantized case is reported only (no exactness claim).

## 5. Training and selection
- Profile 10 updates of each objective (draws not consumed).
- **N** = ⌊0.95 × 0.35 h / (BA s/update)⌋, with N ≥ 100 required (else the screen is budget-limited and reported as such). Checkpoints at round(N/4),
  round(N/2) and N.
- **BL** uses the identical draw list through N, with checkpoints at the same counts. If BL is faster, it continues on the same deterministic draw list
  until its training wall time reaches BA's (cap 0.35 h): the **BL-tm** checkpoint.
- Logged: updates, examples, wall time and GPU time.
- **Development** (60200–60231, full 50-step sampler, batch 32): B0, BA@{N/4, N/2, N}, BL@{N/4, N/2, N}, BL-tm, BL-arch.
  - BA and BL are each selected independently by lowest mean endpoint latent rel. MSE vs FP50; ties within 1e-6 relative go to the earlier checkpoint.
  - Strongest applicable BL comparator = the lowest dev mean among BL-sel, BL-tm and BL-arch.
- **Promotion:** BA-sel mean ≤ 0.95 × the strongest comparator, and BA-sel mean LPIPS ≤ the comparator's. This is a screen, not significance. On
  failure: report the curves and stop BA, and leave 124000–124127 unopened.
- **Confirmation** (if promoted and budget permits):
  - B0, BA-sel, BL-sel, BL-tm and BL-arch on 124000–124127 (batch 32), FP50 reference.
  - **Primary BA/BL-strongest** (as frozen on dev): ratio of mean endpoint latent rel. MSE, paired seed bootstrap 10,000, seed 0.
  - Confirmed improvement iff the CI upper bound < 1 and the ΔLPIPS CI is not wholly above 0.
  - Secondary: BA/B0, the other comparators, LPIPS, PSNR, wins, medians.
  - A wide interval is inconclusive.
- **Diagnostics** (60208–60215; B0, BL-sel, BA-sel):
  - short-window loss on held-out draws;
  - teacher-forced states (FP50 trajectory): raw prediction error, lag-1/2/4/8 discrepancy energy, interval bridge residual (H vs v);
  - the same on live trajectories.
  - No causal time-structure claims from different trajectories.
- A single training seed means this is a pilot.

## 6. Budget and stopping
Each stage estimates its cost from the profile and stops if the dar1 cap or the handoff total would be exceeded. Checkpoints are resumable.
Unopened confirmation seeds are reported by exact ID.

---

## Amendments and results
(appended below, dated)
- 2026-10-31 09:31 UTC, checks attempt 1 (FAILED, OOM): check 1 passed (endpoint rel 0.046 ≤ 0.05; step-1 rel 1.2e-7; 50 + 13 calls, 12 bridges), then CUDA OOM in the BL-loss gradient check (8 per-step graphs alive at once + check models still resident). Fix (implementation only, objective unchanged): BL accumulates each per-step term's gradient immediately (scaled 1/n; identical gradient to backpropagating the mean); check models freed. Checks rerun in full.
- 2026-10-31 09:38 UTC, checks attempt 2 (FAILED, OOM at the reported-only STE finite-difference case, after the fp32 FD checks ran): memory not released by the resume check's three short training runs. Fix (implementation only): explicit gc/empty_cache at the end of train(); FD windows use activation checkpointing (identical values; recomputation). Checks rerun in full.
- 2026-10-31 09:46 UTC, checks attempt 3 completed with pass = false (0.105 A30-h). All checks passed except two, both kept on record:
  (a) **check 4 counts:** the protocol's expected count was wrong. Full windows (a ≤ 40) make **9** cheap calls: 8 transitions plus the right-boundary
  anchor measurement that the bridge at a+8 requires; FP calls 3, bridges 2. Terminal windows match (a = 44: 6 / 2 / 1; a = 48: 2 / 1 / 0). The
  expectation is corrected to 9 / 3 / 2; the code is unchanged.
  (b) **check 6 parameter direction unresolved at fp32:** along the random carry direction the directional derivative is tiny (−4.1e-5, from
  cancellation over ~4M parameters), so central differences sit at the fp32 noise floor. Best rel. error 6.2e-3 at h = 1e-2 (> 1e-3). The latent
  direction (through student, teacher and bridge) is resolved: 7.3e-4 at h = 1e-2 and 1.8e-3 at 1e-3.
  Added check 6b, a resolvable regime: a parameter-direction central FD along the **normalized gradient direction**, same reduced case (quantizer off,
  fp32, batch 1, a = 0), same h grid and tolerance (≤ 1e-3 with a neighbour ≤ 1e-2). It runs as stage `fdgrad`, which updates checks.json and keeps the
  original entries. Training starts only if 6b is resolved.
- 2026-10-31 09:50 UTC, check 6b result: along the normalized gradient direction (fp32), autograd 0.1906 vs FD rel. errors 4.9 / 2.1 / 0.42 / 6.7e-3 /
  2.5e-3 at h = 1e-1 … 1e-5. The error converges with h but stalls at the fp32 noise floor (~0.25%), so it is **not resolved** and not counted as
  validation. Added check 6c, a resolved regime by precision: the same reduced case (no activation quantizer, batch 1, window a = 0 with 2 bridges)
  entirely in **float64**. fp64 variants are used only in this check: the guided call without autocast or fp32 cast, the carry hook without fp32 cast,
  and DiT's timestep embedder output cast to fp64. FD along the gradient and a random direction, h ∈ {1e-1 … 1e-7}, same tolerance. Training proceeds
  only if both directions are resolved.
- 2026-10-31 09:52 UTC, fd64 attempt 1 FAILED (OOM: two fp64 DiT-XL copies + an unchecked 11-call graph). Fix: activation checkpointing of the fp64 calls (identical values). Rerun.
- 2026-10-31 09:58 UTC, check 6c (fp64) result: **gradient direction resolved** (rel. err 2.85e-5 at h = 1e-5; neighbour 6.5e-3 at 1e-4). **Random
  all-site direction not resolved:** best 1.45e-3 at h = 1e-4 (> 1e-3), directional derivative −4.1e-5, ~4,600× smaller than along the gradient
  because ~4M components cancel. So 6c fails as written (both directions were required); this is kept on record. Tolerances are not loosened.
  Bounded diagnostic 6d (fp64, same reduced case, same tolerance): random directions restricted to the carries of one early site (b0.*) and one late
  site (b27.*). Training proceeds only if both resolve; otherwise this implementation stage stops and is documented.
- 2026-10-31 10:05 UTC, check 6d result: single-site random directions **not resolved** (b0: 2.9e-5 at h = 1e-3, but neighbours 1.3e-2 / 1.4e-2 >
  1e-2; b27: best 1.8e-3 at 1e-3). Kept on record. Both 6c and 6d showed an h-dependent noise floor far above fp64 precision.
  **Harness defect found** by a dtype audit (every module output was fp64, but): `dit_joint.apply_smoothing`'s pre-hook computes `x.float() / s` at all
  56 fc1/fc2 inputs, so the "fp64" cheap model truncated activations to fp32 precision there. This is a deterministic bug in the fp64 check harness
  only; the training path is fp32/fp16 by design and unaffected.
  Fix: in the fp64 copy only, the smoothing pre-hooks are replaced by fp64 equivalents (x / s with the same s). **Check 6c is rerun as originally
  amended** (gradient and random all-site directions, same tolerance). It decides; no further check variants will be added. If it fails, the BA
  implementation stage stops and is documented.
