# abn1 — activation feedback, anchor reuse and retrospective bridge (overnight) — pre-registration

Written 2026-10-30, **before any check or quality result of this experiment**. Branch `activation-bridge-overnight` (worktree /home/boa-dit-abn) from
anchor-bridge-pilot 382e405 (verified remote heads 2026-10-30: anchor-bridge-pilot 382e405 [13cb833 + CLAUDE.md], activation-anchor-pilot d1eab46,
local-error-refresh-pilot 1b00a67, weight-anchor-w8 dc76c1e, anchor-drift-pilot b42f065). The source branch contains, unchanged: aa1 sampler conventions,
ler1 `LocalQuant` (matched-range stateless G), br1 bridge, wa8 `h8_model`, dit_joint. Code `boa_dit/activation_bridge/run.py`; queue
`scripts/abn1_queue.sh`; report `scripts/make_activation_bridge.py`. Tag **abn1**. Specification: the user's overnight handoff (2026-10-30), which
authorizes the queue below without further approval. Amendments below the marker, dated, before any affected result.

Pre-existing work: no other job was running at launch (checked 2026-10-30 18:17 UTC: GPU 0 MiB, only jupyter-lab). No delay.

## 1. Frozen setup
- Cheap: unrefitted common smoothed rank-8 W4 base (f017af87…) + original carry (b4a537b0…) + smoothing (7ee15c0c…) + static single8 tables (b73030de…).
  `LocalQuant` (parent `JointQuant` arithmetic unchanged): **C** mode `static`; **AC** mode `fb` (DCT-4 feedback); **G** mode `fb` with the previous state
  dropped before every call (the parent creates zeros, exactly as at t = 0) — same clipping endpoints, guard, spare code, cast and rounding as AC, zero carried
  state. Stage 4 swaps in the joint-pilot local refits `carry_C_L_u200.pt` (ac201610…) with C and `carry_E_L_u200.pt` (77b5d0ac…) with AC.
- High: FP (`load_dit`) or **H8** = validated wa8 `h8_model` (heig-8 W4 state with the 112 weights replaced by dequantized fixed-calibration W8 codes,
  own carry δ8; h8.pt c574916e…, heig-8 W4 model facee154…), its own preprocessing, A16 activations. Separate module instances; no shared hooks/state.
- Bed: DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5 (`weight_anchor.run.guided`; guided channels 0–3 consumed), fp16 autocast, fp32 sampler, batch **32**.
- Sampler (br1 convention generalized): anchors at i ∈ {0, K, 2K, …} ∩ [0, 49]; at an anchor the cheap model is called once on the pre-correction latent
  (its feedback state commits here) and the high model on the same latent/timestep/labels/CFG layout; d = Q(x) − H(x) on channels 0–3. Between anchors the
  cheap prediction minus the held d. **Bridge** at anchor c after anchor a (k = c − a, explicit): x_c ← x_c − (d_c − d_a)·Σ_{j=1}^{k−1}(j/k)·b_{a+j}·Π_{m=j+1}^{k−1}A_{a+m},
  then the DDIM step with the already-computed high prediction; d_c held. Multiplier **1** (fixed by br1). Final tail after the last anchor (48) uncorrected.
  No re-evaluation, no state reset or recomputation after the jump. K4: 13 anchors / 12 corrections; K8: 7 / 6; K1: zero coefficient.

## 2. Arms
| stage | arms (all on their own trajectories) | images |
|---|---|---|
| 1 | FP50, C, AC, CB, ACB, G, GB (FP anchors, K4) | 256 |
| 2 | H8_50, C.w8, AC.w8, CB.w8, ACB.w8, GB.w8 (W8 anchors, K4; GB.w8 dropped only if C and G are bitwise identical) | first 128 |
| 3 | C.k8, AC.k8, CB.k8, ACB.k8 and the same four with W8 anchors (.w8.k8) | first 128 |
| 4 | LCB (C_L + bridge, feedback off), LACB (E_L + bridge, feedback on), FP K4 | first 128 |
| 5 | C, AC, CB, ACB, G, GB diagnostics | 8 dev (60208–60215) |
| 6 | FP50, CB, ACB on the FID cohort | 2,000 |
| 7 | timing: FP50, C, CB, AC, ACB at batch 1 and 32 (32 only if it fits) | — |
Stages 2–4 reuse Stage-1 latents for the first 128 images (identical seeds, labels, batches, models, precision, sampler).

## 3. Cohorts (audited 2026-10-30 on all 20 remote branches and all 17 worktrees, word-boundary search of *.py/*.md/*.csv/*.txt/*.yaml; method
validated by finding the known 96100 block)
- Confirmation: **110000–110255** (unused; reserved). Labels `labels_for_seeds`. Batches 110000–110031, …; first 128 = the first four batches.
- FID cohort: noise seeds **112000–113999**, labels i // 2 (two per class, explicit), paired across arms (unused; reserved).
- Development (inspected, exploratory; checks and catastrophe diagnosis only): 60208–60223; diagnostics 60208–60215. Checks also use 60200–60215 (aa1
  cache), 94000–94031 (wa8 H8 cache) and br1's 60208–60223 cache for bitwise reproduction only.

## 4. Correctness gate (stage `checks`; all must pass before any quality sampling)
C1 reproduce aa1 C, AC, FP50 (60200–60215, batch 16) and br1 CB1 (60208–60223, batch 16) bitwise; C2 bridge multiplier 0 = C and AC bitwise incl. final
feedback state; C3 no anchors = plain static / feedback (aa1 Q / A) bitwise; C4 K1 = H50 for FP and W8, feedback off and on, zero corrections (batch 4,
references regenerated at batch 4); C5 identical stateless cheap/high (FP/FP) → zero discrepancy, zero bridge, FP50 trajectory; C6 counts 50 + 13 (K4) and
50 + 7 (K8), 50 quantizer updates per site, corrections 12 / 6; C7 anchor calls receive the same tensors (13 recorded) and perturbing one sample leaves the
others unchanged (bitwise expected; reported); C8 save/resume immediately before (i = 8) and after (i = 9) an anchor bitwise in latent and state; C9 synthetic
fp64: constant discrepancy → zero bridge, linear drift cancels for K4 and K8 with the real coefficients (≤ 1e-12); C10 H8 hash-verified and H8-50
reproduces the wa8 cache bitwise; C11 G ignores any carried state (junk state = zero-state AC, bitwise) and uses AC's guarded path; C12 diagnostics
(including passive quantizer-observer accounting) on/off leave endpoint and state bitwise unchanged. C vs G on dev trajectories: aliased only if bitwise.

## 5. Statistics (frozen)
Endpoint latent rel. MSE vs FP50 (existing `weight_anchor.controls.metrics` definition), PSNR, LPIPS-VGG; per-image values saved. Paired image bootstrap
10,000 draws, numpy seed 0 (`weight_anchor.shared.contrast`), ratio of mean errors, median paired differences, wins.
- **Single primary: ACB / CB at FP K4 on all 256.** Useful: ratio ≤ 0.95, CI upper < 1, ΔLPIPS CI not wholly above 0. Harm: CI lower > 1. Smaller positive:
  CI upper < 1 otherwise. Else inconclusive (never equivalence).
- Key mechanistic secondary: ACB / GB. Other secondaries (exploratory, unadjusted): AC / G, ACB / AC, CB / C, AC / C, GB / G, G / C, GB / CB; matched interaction
  I = [E(GB) − E(ACB)] − [E(G) − E(AC)]; package interaction with C / CB in place of G / GB.
- Stage 2: CB.w8 / C.w8, ACB.w8 / CB.w8, ACB.w8 / GB.w8, CB.w8 / CB, ACB.w8 / ACB, each W8 bridge arm vs H8_50, AC.w8 / C.w8; H8_50's own error vs FP50.
- Stage 3: bridge and feedback gains within each precision; CB.k8 and ACB.k8 vs C (K4) and vs CB (K4); W8 analogues vs C.w8 / CB.w8. Exploratory
  noninferiority: upper CI of E(bridge_K8)/E(ordinary C_K4) < 1.05 and ΔLPIPS upper bound < 0.005; otherwise "unresolved" or "inferior beyond margin".
- Stage 4: LCB / CB, LACB / ACB (LACB / LCB reported, not attributed to feedback).
- Stage 6: FID-2k (pytorch-fid pool3 vs the ADM VIRTUAL_imagenet256 reference stats a11af8b0…, images b3273271…), paired generated-feature bootstrap
  (1,000 draws) for ACB − CB, CB − FP50, ACB − FP50, conditional on reference and n.
- Catastrophe screen: an arm whose development or cohort mean latent error is > 5× its matched baseline, or nonfinite, is reported as a failed screen.

## 6. Budget and queue
Hard cap **6.0 A30-h** (ledger tag abn1 in `results/gpu_hours.csv`, all checks/diagnostics/failures/decoding/metrics) and **8 h wall clock** from launch
(`ckpt/activation_bridge_overnight/launch_utc.txt`). Planning: checks 0.5, s1 0.9, s2 0.6, s3 0.7, s4 0.4, s5 0.3, s6 1.2, s7 0.2 (reserve 1.2).
Each stage estimates its full workload from measured per-batch timings (Stage 1 after it runs; fallback 60 s static / 92 s feedback / 21 s high-only per
32-image trajectory) and is skipped whole if spent + estimate + 0.15 h shutdown reserve exceeds the cap or the wall time left; Stage 6 also keeps Stage 7's
0.2 h. Runtime stop before every batch. Single sequential queue (`flock`), `timeout` watchdog at the wall deadline minus 9 min, one retry after a crash
(generation resumes from per-batch checkpoints), commit/push after every stage. Order: checks → s1 → s2 → s3 → s4 → s5 → s6 → s7 → report.
A failed gate blocks all quality stages. Large artifacts stay local (standing instruction: nothing trained is uploaded; regenerate from code), with hashes.

---

## Amendments and results
(appended below, dated)
