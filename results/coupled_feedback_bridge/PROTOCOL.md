# cfb1 — zero-reference activation feedback with jointly calibrated gains and bridge strength (pilot J) — pre-registration

Written 2026-10-31, **before any check or quality result of this experiment**. Specification: the user's handoff "two coupled activation-feedback /
anchor / bridge pilots" (2026-10-31), section 4 (pilot J) with shared sections 1, 2, 5, 6; it supersedes the queue order of the earlier
coupled_feedback_bridge_protocol.md (which was never committed or started on any branch). Pilot R (rfb1, branch `reference-feedback-bridge-pilot`) is a
separate experiment and ran first; no parameters are shared.

Branch `coupled-feedback-bridge-pilot` (tag **cfb1**), from `activation-bridge-overnight` 5023a0a, plus `boa_dit/feedback_bridge/core.py` (byte-identical
to rfb1's, sha256 37a4ce1a…) and `boa_dit/feedback_bridge/cfb.py`. J did not start before this protocol: no prior spend, cohorts or decisions exist.

## 1. Fixed bed
This is identical to rfb1 §1 / abn1: DiT-XL/2 256, DDIM-50, CFG 1.5, fp16 autocast with an fp32 sampler. The cheap model is the original unrefitted
common smoothed rank-8 W4 base with its carry, smoothing and static single8 A8 tables. The FP anchor model is a separate instance.
- K4 anchors at 0, 4, …, 48. d_a = cheap − FP on the same pre-correction latent; between anchors the prediction is cheap − d_a.
- Bridge x_c ← x_c − λ(d_c − d_a)·Σ_j (j/k)·b_{a+j}·Π_m A_{a+m}, then the stored FP prediction. There is no re-evaluation, and the committed cheap
  state is kept. The final interval is uncorrected.
- Calls: 50 cheap + 13 FP, 12 corrections, and 50 quantizer updates per site.
- Latent error is `weight_anchor.run.rel` (per image, ‖x − x_FP50‖²/‖x_FP50‖² over the endpoint latent) against the paired FP50 at the same batch.

## 2. Rule J (`core.FBQuant._fb_gain`)
θ = (g_anchor, g_between, λ). At each quantizer call, fed = clip(g_t·s_prev, lo − xcl, hi_rep − xcl); everything else is unchanged from the original DCT-4
feedback: second clip, spare code, fp16 cast, r = clamp(uc − consumed, ±Δ/2), and state fp16(Uᵀr). g_t is g_anchor on anchor calls and g_between
otherwise. All sites share the gains. The rolling state updates at anchors and is not reset there. There is no reference tensor, so J and R are
different constructions.

## 3. Grid, families, selection (frozen before data)
- Grid: g_anchor, g_between ∈ {0, 0.5, 1}; λ ∈ {0.5, 1, 1.5} — 27 points.
- Families:
  - **B**: g_anchor = g_between = 0, λ tuned.
  - **U**: g_anchor = g_between, with shared gain and λ tuned.
  - **J**: all 27 points.
- Objective: mean endpoint latent rel. MSE vs paired FP50 over the complete live sampler (each point on its own trajectory, state, discrepancies and
  corrections), on fit data, latent metrics only. There is no local-error pre-fit, STE, finite differences or refinement.
- Selection: smallest fit mean. Exact ties within relative 1e-6 are broken by fewer nonzero gains, then lower gain sum, then smaller |λ − 1|, then the
  lexicographic tuple.
- B, U and J are frozen (written to fit.json) before validation. The fixed configurations B0 = (0, 0, 1), N = (1, 1, 1) and H = (0, 1, 1) are also
  validated, and identical configurations are deduplicated (generated once, scored under every role).
- The frozen-window identity E‖H_g − λv_g‖² = E‖H_g‖² − 2λE⟨H_g, v_g⟩ + λ²E‖v_g‖² and its least-squares λ are diagnostics only and never replace
  endpoint calibration.

## 4. Data and batch
- Cohorts (proposed in the handoff; audited 2026-10-31 by a word-boundary search of *.py/*.md/*.csv/*.txt/*.yaml on all local and remote branches and
  all worktrees — no 120000–122999 match; the method finds the known 96100 block; no shift; reserved in `handoff/SEED_REGISTRY.md`):
  - fit **120000–120031** (32);
  - validation **120100–120131** (32);
  - confirmation **121000–121255** (256).
  These are disjoint from rfb1 (122000–122127) and earlier work. Diagnostics use the first 8 fit images.
- Checks use evaluated or inspected seeds only (overnight 110000–110031 at batch 32, 60200–60207).
- Batch: stage `profile` (development noise 60200–60231, 8 steps) — batch 32 if the J peak reserved memory is ≤ 0.85 × device memory, else 16. One
  batch size and precision for all configurations and references, recorded as an amendment before fit data are opened.
- Before opening fit data, the complete grid plus reserved validation, confirmation and diagnostics must fit the 3.0 h cap (profiled estimates ×1.10).
  Otherwise J stops and reports the required cost; it does not shrink or extend adaptively.

## 5. Correctness gate (stage `checks`)
- With J configurations, reproduce overnight FP50, G (J(0,0) without bridge), GB (J(0,0,1)) and ACB (J(1,1,1)) bitwise at batch 32. Check that the
  stored states of J(1,1,1) equal those of the original-rule ACB.
- J(1,1) without bridge equals AC including state.
- λ = 0 equals no bridge, including state.
- K1 equals FP50 with zero corrections.
- Counts, 50 updates per site, and 13 identical cheap/FP anchor inputs.
- Perturbing one sample leaves the others bitwise unchanged; there is a reset between batches.
- Resume before (8) and after (9) an anchor is bitwise.
- Instrumentation on/off is bitwise.
- Gain placement: an anchor call with g_anchor = 0 equals the G call; a held call with g_between = 1 equals the AC call, including state.
- The synthetic constant/linear examples pass.
- A deterministic bug may be fixed and documented before evaluation; tolerances are not loosened.

## 6. Validation gate and confirmation (frozen)
- Gate: J must use nonzero feedback, mean error ≤ 0.95 × B, and mean LPIPS ≤ LPIPS(B) + 0.005.
  - If it fails, the scoped failure is reported; there is no runner-up and confirmation stays unopened.
  - If J selects a B configuration, it is reported as "the J family chose feedback off".
- If the gate passes, frozen B0, N, H, B, U and J run on all 256 confirmation images, with no new choices.
- Primary **J / B**: "useful" iff ratio ≤ 0.95, 95% paired upper < 1, and no clear LPIPS regression (ΔLPIPS CI not wholly above 0).
- J / U is tested as the phase-specific contribution only after primary success. If J = U, phase specificity is not established.
- If only λ differs from B0, the effect is labelled bridge recalibration. Improvement over N alone does not establish useful activation feedback.
- Other contrasts (U/B, B/B0, J/N, N/B0, H/B0, J/H, J/B0) are exploratory and unadjusted.

## 7. Statistics
Paired image bootstrap, 10,000 draws, numpy seed 0 (`weight_anchor.shared.contrast`). Reported: mean, median, ratio of means, PSNR, LPIPS-VGG and wins.
There is no step or token bootstrap, nonsignificance is not called equivalence, and there are no head-to-head significance claims against R.

## 8. Diagnostics (stage `diag`, 120000–120007)
- Live: the nine gain pairs at λ = 1, plus the selected B, U and J. Recorded: H_g, v_g, E‖H‖², E⟨H, v⟩, E‖v‖², cosine, frozen-window objective and
  λ_LS, anchor vs between raw errors, jump norms and endpoint error.
- Common forced trajectory: cheap and FP are called on the FP50 latents, with each gain pair keeping its own state history; the same interval
  quantities are recorded.
- The alternating-error toy is not used to explain live endpoints.

## 9. Budget and queue
- Cap **3.0 A30-h** (ledger tag cfb1), including profile, checks, failures, decoding and metrics. The combined R + J ceiling is 6.0.
- Each stage is skipped whole if spent + estimate + reserve > cap; there is a runtime stop before every trajectory, with per-(batch, configuration)
  checkpoints.
- Order: profile → checks → fit → val → conf (if the gate passes) → diag → report. Queue `bash scripts/cfb1_queue.sh` (shared GPU lock with rfb1, one
  retry, commit/push per stage).
- Latents stay local, with sha256 in artifacts.json.

---

## Amendments and results
(appended below, dated)
