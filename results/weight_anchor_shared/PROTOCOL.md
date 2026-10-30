# Shared W4 / higher-precision anchors (ws1): storage-and-accuracy pilot — pre-registration

Written 2026-10-30, **before any build, check or generation of this pilot**. Branch `weight-anchor-shared` from weight-anchor-w8
dc76c1e4a4b338e644dcab38680aad804475f9d0 (verified equal to the completed wa8 head). Code `boa_dit/weight_anchor/shared.py`,
report `scripts/make_weight_anchor_shared.py`, results `results/weight_anchor_shared/`, artifacts `ckpt/weight_anchor_shared/` (not in Git).
Stage tag **ws1**. The user's handoff text (2026-10-30) is the specification; this file fixes its open choices. Amendments below the marker, dated.

## Hypotheses (none concerns speed or peak device RAM)
- H1: frozen heig-8 W4 base + 4 refinement bits per quantized dense weight gives a useful anchor.
- H2: reuse of its discrepancy preserves most of the independent-W8 hybrid's (wa8 R8) accuracy benefit.
- H3: the packed shared representation reduces stored bytes vs an equally packed independent Q4 + H8 pair.
The independent H8-50 (plain W8) has less endpoint error than R8; it stays in every table.

## Budget
**New hard ceiling 2.0 A30-hours** in `results/gpu_hours.csv` rows tagged `ws1` (builds, checks, diagnostics, evaluation, failures, retries).
Planning ceilings: builds + checks 0.65 h; development + diagnostics 0.55 h; confirmation 0.55 h; reserve 0.25 h. Every stage has a pre-launch
guard (spent + estimate + confirmation reserve ≤ 2.0) and a runtime stop before each batch (as in wa8). Cut order: optional traces, then A-RTN
live sampling. A failed construction is reported, never silently dropped.

## Fixed bed (from wa8, unchanged)
DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5 (ε channels 0–2, channel 3 from the cond half, channels 4–7 unused by DDIM), fp16 autocast forward, fp32
sampler, batch 32, labels/noise via `model.labels_for_seeds` / `noise_for_seeds`, guided output `weight_anchor.run.guided`, metrics as wa8
(latent rel. MSE vs FP50, PSNR and LPIPS-VGG after sd-vae-ft-ema decode, `weight_anchor.controls.metrics`). K = 4 (anchors 0, 4, …, 48), β = 1.

## 1. Representations (low path frozen)
For each of the 112 block linears: W = original FP weight; Q4 = s4·(c4 − z4), the exact stored heig-8 weight; c4 recovered as round(Q4/s4 + z4) and
required to dequantize to Q4 bitwise; s4, z4, V (fp32, from `ckpt/temporal/heig8_layers.pt`), δ4 (fp16, stored heig-8 carry). The W4 model is
reconstructed from the packed artifact and must equal the heig-8 state dict bitwise, and its trajectory must equal the cached wa8 Q4 latents.
adaLN (W8 RTN) and all other tensors are unchanged (adaLN codes recovered from the FP weights with the RTN rule on CPU, verified bitwise).

- **A (residual refinement, own grid):** D = W − Q4; D̂ = sr·(cr − zr), cr ∈ 0..15, per-output-row affine, integer zr; QA = Q4 + D̂ (fp32).
  - **A-RTN:** `qutils.rtn_weight(D, 4)` semantics (per-row min/max range including 0, scale = range/15, zero = round(−min/scale), nearest
    rounding, no search). Local reconstruction control; live hybrid only if budget permits; never promotable without a pre-result amendment.
  - **A-GPTQ:** `qutils.boa_gptq_rows` unchanged on target D, bits 4, with the saved projected metric P H P, replace 1/256, BoA preprocessing
    (dead columns, 1% damping, retry damping if Cholesky fails), natural column order, and BoA's own per-row grid rule (`qparam_comput="Hessian"`:
    the H-weighted scan of shrunken min/max ranges built into the audited solver, exactly as heig-8 and wa8's H8 used it). No grid/clip/order
    search is added beyond what that solver call already does. The solver matrix actually used (after damping and any retry damping, via
    `weights_joint.joint.prep`) is logged per layer (diag mean, damping added).
  - This is a sum of two affine 4-bit reconstructions (8 payload bits), not uniform INT8; executing it may need reconstruction or two low-bit products.
- **B (fixed-prefix nested uniform grid):** c8 = 16·c4 + r, r ∈ 0..15; s8 = s4/16 (exact power-of-two scaling); z8 = 16·z4 + 8 (int32 arithmetic);
  QB = s8·(c8 − z8) evaluated as one fp32 product of s8 and the exact integer (c8 − z8). Then c8 >> 4 = c4 exactly.
  - **B-GPTQ:** BoA's GPTQ column loop (same preprocessing, damping, natural order and Cholesky factor U as `weights_joint.joint.prep`/`gptq_static`,
    which reproduced heig-8 bitwise in the weights-joint audit) on target W, with the scalar rounding replaced by the nearest admissible level of the
    element's fixed prefix: c8 = clamp(round(w/s8) + z8, 16c4, 16c4 + 15) (BoA's `fake_quantize` form on the fine grid, torch.round
    = round-half-to-even, then clipped to the cell); error feedback uses the full post-clip error (w − q)/U_jj; the target is never clamped.
  - **Validation before GPU use** (`shared selftest`, CPU): (i) exhaustive comparison of the constrained rounding against brute-force enumeration
    of the 16 admissible levels on random, boundary (cell edges, outside below/above) and exact-tie inputs (ties resolved as half-to-even on the
    fine grid; the brute force reports which tie rule it matches); (ii) the vectorized-over-rows column loop equals an unblocked per-element
    reference (explicit row and column loops) on a small random matrix; (iii) with the cell constraint replaced by the full 4-bit grid (s4, z4)
    the same loop reproduces Q4 bitwise on one real layer (on GPU, in the build stage).
  - **Ceiling diagnostics:** fraction of elements whose desired fine code round(w/s8) + z8 lies outside the cell, and the clip error energy
    Σ(w − q)² of those elements vs all, per layer and total. A failure of B is not evidence against jointly optimized nested quantizers.
- **Carry:** for every candidate QH: δH = (W − QH) V by `deflate.carry.make_carry` (fp32 product, stored fp16; V fp16), with the fp32 V. For A the
  algebraic form δ4_fp32 − D̂ V is compared to the direct form before casting (max-abs and relative difference recorded). The old δ4 is never
  added to Q4 + D̂.

## 2. Packing and validation
Artifacts are safetensors files (`ckpt/weight_anchor_shared/*.safetensors`), metadata JSON in the header (shapes, layout, sha256 per tensor):
- nibble planes: uint8 [rows, d_in/2], byte = code[:, 2k] | code[:, 2k+1] << 4; base plane = c4, refinement plane = cr (A) or r (B);
  H8 8-bit codes as uint8 [rows, d_in].
- scales fp32 [rows] (evaluation uses fp32 scales, so fp32 is stored and counted); zero points uint8 [rows] (integers in range, verified).
- V fp16 once; δ4, δH fp16 (the evaluated carries are fp16).
- adaLN: uint8 codes + fp32 scale + uint8 zero per row; other tensors fp32 as evaluated (an fp16 conversion is a separate ablation, not run).
Files: (a) independent pair Q4 + H8 (wa8 h8.pt codes), (b) shared pairs A-GPTQ, B-GPTQ, A-RTN, (c) H8 alone, (d) Q4 alone; (e) FP16 reference
bytes computed (2 bytes × state-dict elements). Identical metadata precision and auxiliary sharing (V, adaLN, other once) across (a)–(d).
Every model used live is built **from its artifact** (unpack → fp32 weights + carry), and every unpacked tensor/plane is round-trip checked
against the in-memory build (bitwise). Bytes reported: on-disk file size; logical unique tensor bytes by component; resident simulated weights
(fp32 dequantized); measured peak allocated/reserved during live stages; hypothetical packed-kernel memory = logical bytes (unmeasured).

## Checks (stage `checks`, stop on unexplained failure)
- K1: Q4 unpacked from the shared artifact equals the heig-8 state dict + carry bitwise; Q4-50 and FP50 trajectories on 94000–94031 equal the
  cached wa8 dev latents (`ckpt/weight_anchor_w8/dev_latents.pt`) bitwise; H8 unpacked from the pair artifact gives the cached wa8 H8 latents
  bitwise (licensing reuse of the cached wa8 H8 and R8 dev latents; h8.pt sha256 checked against wa8 provenance).
- K2: A with zero refinement (cr = zr) + δ4 recovers the W4 model (weights bitwise, forward bitwise).
- K3: R_shared with the correction disabled equals A_shared bitwise (each GPTQ candidate).
- K4: R_shared with K = 1 equals H_shared-50 bitwise (each GPTQ candidate).
- K5: low = high = Q4 gives exactly zero discrepancy and the Q4 trajectory.
- K6: at every anchor, the Q4 and candidate calls receive identical latent, timestep and labels tensor (the conditional/null CFG arrangement
  is built from that labels tensor inside `guided`); the hybrid model dict contains no FP model (any FP use raises).
- K7: call counts — R_shared 50 Q4 + 13 high; A_shared 37 + 13; H_shared 50 high; all finite; seeds/labels match the cache.
Layer diagnostics (build): raw Frobenius ‖W − QH‖/‖W‖; projected quadratic tr(E·PHP·Eᵀ) (labelled projected: it annihilates span(V) and cannot
measure the carry); Frobenius error after carry ‖(W − QH) − δH Vᵀ‖/‖W‖; the full unprojected calibration metric is **unavailable** (not cached:
`heig8_layers.pt` holds only P H P; `dit_calib_hist.pt` holds activation histograms) and is not regenerated.

## 3. Development (94000–94031, already inspected, exploratory)
For A-GPTQ and B-GPTQ: H_shared (50 high), A_shared (anchors, plain Q4 between), R_shared (anchors + R_t = Q_t − (Q_a − H_a)), each on its own
trajectory; compared with Q4, H8, R8, FP50 (cached wa8 latents after K1). A-RTN live arms run afterwards only if spent ≤ 1.0 h.
**Selection (frozen now):** among A-GPTQ and B-GPTQ with all checks passing, mean H_shared latent MSE < mean Q4, and packed pair smaller than the
packed independent pair: the lowest mean R_shared dev latent MSE; if the two are within 2% (|E_A − E_B| / min(E_A, E_B) ≤ 0.02), choose B.
If none qualifies: report and stop (no confirmation).
**Optional traces** (selected candidate, first 8 dev seeds 94000–94007, along the Q4 trajectory): `w8.drift` with d = Q4 − H_shared,
e = Q4 − FP; lags 1/2/4/8; cosine, norm change, Σ‖d_t − d_a‖²/Σ‖d_t‖², corrected Σ‖e_t − d_a‖²/Σ‖e_t‖², image-shuffled control (cyclic shift).
Not used for selection.

## 4. Confirmation (fresh 95100–95163)
Verified 2026-10-30 absent from the seed registry and from `*.py/*.md/*.csv` on all remote branches and local worktrees. Seven arms on the same
64 images: FP50, Q4, H8, R8 (independent, wa8 construction, recomputed live), H_shared, A_shared, R_shared (selected candidate).
Statistics: paired image bootstrap, **10,000 replicates, numpy seed 0**; ratio of cohort mean rel. MSE (recomputed per resample), paired absolute
differences, median difference, win counts, PSNR and LPIPS differences with CIs. No per-image-ratio averages as the main statistic.
**Frozen decisions:**
1. *Useful reuse:* E(R_shared)/E(A_shared) − 1 ≤ −10%, its CI upper bound < 0, and the ΔLPIPS (R_shared − A_shared) CI not entirely above 0.
2. *Preserve independent-anchor performance:* upper CI bound of E(R_shared)/E(R8) ≤ 1.10 **and** upper CI bound of mean LPIPS(R_shared) −
   LPIPS(R8) ≤ 0.01 → "within margins". If the lower CI bound of the ratio > 1.10 or of the LPIPS difference > 0.01 → "degradation beyond margin
   established". Otherwise → "inconclusive". (Whether R_shared is worse than R8 at all is reported separately.) A CI crossing 0 is never equivalence.
3. *Storage:* on-disk shared artifact < on-disk independent pair artifact (identical metadata precision and sharing).
Each reported separately. Retained gain B = (E_Q4 − E_Rshared)/(E_Q4 − E_R8) with bootstrap CI and both absolute gains; not headlined alone.
H_shared-50 and H8-50 appear in the main table.

## 5. Runtime
No custom kernels. Simulated timings are not recorded: all models are fp32-dequantized, so simulation time carries no information about packed
cost. Cost condition stated only: K = 4 needs 50·C_Q4 + 13·C_high + overhead < 50·C_high; A's high layer may need two low-bit products.

## Stages (`python -m boa_dit.weight_anchor.shared <stage>`)
`selftest` (CPU) → `build` → `pack` → `checks` → `dev` (+ `dev_rtn`, `traces` if budget) → freeze (commit `frozen.json`) → `conf` → `report`.
Commit and push after every stage. If push fails, keep local commits and report that the work is not backed up.

---

## Amendments and results
(appended below, dated)
- 2026-10-30, selftest (CPU, before any GPU use; committed with the pre-registration):
  - The first selftest run judged the constrained rounding against level-domain distances |w − L_r|. It flagged near-tie elements whose chosen
    level was farther by ≤ 2.8e-5 fine steps (fp32 rounding of w/s8; 1 of 200k random, 9.9k of 200k constructed half-integer ties).
  - The criterion was refined to the definition of the rule: exact float64 enumeration of the 16 admissible codes in the index domain
    u = fl(w/s8) + z8 (the domain BoA's fake_quantize rounds in). Result: 0 non-tie mismatches in all three input families; all exact ties
    resolved to the even code (half-to-even); 0 cell violations; level-domain excess still reported (≤ 2.8e-5 s8, pass threshold < 1e-4 s8).
  - Column loop vs unblocked per-element reference: bitwise (cell-constrained and full-grid). The real-layer full-grid reproduction of Q4 runs in `build`.
- 2026-10-30, implementation notes (before any GPU result):
  - dev reuses the K3/K4/K6 check runs on 94000–94031 as the H/A/R development latents of each GPTQ candidate (same seeds, models, code
    and batch; saved by `checks`), instead of regenerating identical trajectories.
  - Ledger tag `ws1`; confirmation reserve 0.55 h held by every guard before `conf`.
- 2026-10-30, build + pack: completed; round-trips bitwise for every artifact and model half; B loop on the full 4-bit grid reproduces Q4 (layer 0).
  Local diagnostics and bytes in build.json / storage.json (summarized in REPORT.md).
- 2026-10-30, checks (0.128 h): K1–K7 pass for A-GPTQ and B-GPTQ; low model bitwise identical to heig-8 (state, forward, trajectory).
