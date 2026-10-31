# bar1 — audit and replication of anchor-aware branch calibration (BA vs local BL) — pre-registration

Written 2026-10-31, **before any bar1 / wkf1 measurement**. Specification: the user's handoff "Claude Code handoff: audit and replicate anchor-aware
calibration" (2026-10-31). This is a **new protocol**. It does not alter the dar1 read: dar1's primary stays inconclusive, 0.835 [0.489, 1.423].

- Branch `ba-replication` (worktree /home/boa-dit-bar), from `dit-anchor-refit` cd9a052 (dar1 + w8s1 complete, verified head).
- Kernel-feasibility track: branch `w8-kernel-feasibility` (worktree /home/boa-dit-wkf), same base. Its own `results/w8_kernel/PROTOCOL.md`.
- Tags **bar1** (this branch) and **wkf1** (kernel track).
- Common ledger `scripts/h1031b_budget.py`: hard total **4.0 A30-h**, inclusive of failures, profiling, training and evaluation.
- Unused budget from earlier handoffs is not carried over. No new GPU rental.
- SANA stays parked: no Nunchaku rebuild, no CUDA 13, no kernel changes, and the SANA confirmation is not reopened.

## 0. Deployed configuration (recovered from the completed dar1 code and reports; not from branch names)
- **Source:** `boa_dit/anchor_refit/run.py` at cd9a052, with inference by `boa_dit/activation_bridge/run.py::sample` (abn1 harness).
- **Model and sampler:** DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5.
  - CFG is applied to the first 3 eps channels; channel 3 comes from the cond half.
  - The scheduler consumes the 4-channel guided prediction; d is computed on those 4 channels.
  - fp16 autocast with an fp32 sampler.
- **Cheap model, W4A8 static:**
  - common smoothed rank-8 W4 base `ckpt_deflate_heig_r8_w4_s/model.pt` (f017af87…);
  - smoothing `temporal/dit_smooth_s.pt` (7ee15c0c…), folded through fp32 pre-hooks;
  - static A8 tables `dit_joint/static_single8.pt` (b73030de…), applied by `JointQuant` mode `static`:
    - per-tensor asymmetric 8-bit, clipped to the calibrated interval, `q_static` with hi_code qmax−1;
    - indexed by the sampler step index `t`, so the step index must be set for any out-of-sampler call;
    - stateless: no feedback state and no guard path.
- **Branch:** 112 rank-8 carries (V [d_in, 8], delta [d_out, 8]), installed as forward hooks.
  - B0 = original carry `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` (b4a537b0…).
  - BL-arch = archived strong local refit `ckpt/dit_joint/carry_C_L_u200.pt` (ac201610…).
  - Pilot checkpoints `ckpt/anchor_refit/carry_{BA,BL}_u{82,165,330}.pt` and `carry_BL_tm.pt`; hashes in `results/anchor_refit/artifacts.json`.
- **Anchors and bridge:** FP anchors K4 at 0, 4, …, 48.
  - At an anchor, cheap and FP are evaluated on the pre-correction latent, and d = Q − F (4 channels).
  - Bridge: x ← x − λ(d_c − d_a)·Σ_{j=1}^{k−1}(j/k)·b_{a+j}·Π A, with λ = 1.
  - The DDIM step from the corrected latent uses the pre-correction FP prediction.
  - 50 cheap + 13 FP calls and 12 corrections per trajectory; the interval after 48 is uncorrected.
- **Pilot training procedure (the completed dar1, with its dated amendments):**
  - Adam lr 1e-4, batch 4, TRAIN 60100–60163.
  - Fixed draw list (`draws`, torch Generator seed 0): 4 seed indices plus a window start a ∈ {0, 4, …, 48}.
  - N = 330 updates, with checkpoints at 82, 165 and 330.
  - **BA:**
    - two-interval corrected window starting from the cached B0 corrected-trajectory latent at a, taken before a's anchor operations;
    - no bridge at a; bridges at a+4 and a+8;
    - anchor discrepancies recomputed with the current student and the FP teacher at the student's current latent;
    - teacher weights frozen, teacher input-state gradient **retained** (checkpointed);
    - static quantizer gradient = **straight-through estimator** (surrogate; not exact);
    - loss ‖x_end − x_end^FP‖² / den[a].
  - **BL:**
    - per-step ‖Q(x_i^FP) − F(x_i^FP)‖² / loc[i] over the same window transitions, at FP-continuation states;
    - same draws, per-term backward;
    - BL-tm continues the same draw list until its training time reaches min(BA training time, 0.35 h) = 1260 s in the pilot (565 updates).
  - The complete check list (FD validation in fp64 after the harness fix, resume bitwise, export↔install bitwise) is in
    `results/anchor_refit/checks.json` and PROTOCOL amendments.
- **Pilot selection and primary:**
  - Dev 60200–60231: each of BA and BL is selected independently by lowest mean endpoint latent rel. MSE vs FP50 among {82, 165, 330}; ties within
    1e-6 relative go to the earlier checkpoint.
  - The pilot then took as its comparator the strongest dev mean among {BL-sel, BL-tm, BL-arch}. Realized: BL-sel = BL@330 was strongest, so the pilot
    primary was BA@330 / BL@330 (the update-matched, dev-selected local refit).
  - **This protocol's primary is BA-sel / BL-sel**, as the handoff specifies: the update-matched, selected local comparator. BL-tm and BL-arch are named
    secondary comparators, fixed now.

## 1. Data and cohorts
| set | seeds | status before this protocol | use here |
|---|---|---|---|
| training | 60100–60163 | training | replication training (unchanged) |
| dev / selection | 60200–60231 | inspected | per-seed checkpoint selection (the pilot rule) |
| **mechanism** | **60200–60215** (first 16 of dev, frozen now, before any example is seen) | inspected | common-state traces, replay, standalone |
| dar1 confirmation | 124000–124127 | evaluated | audit (1A) and power planning only — not fresh |
| w8s1 | 126000–126063 | evaluated | not used |
| **fresh evaluation** | **128000–128511 reserved** (registry audit 2026-10-31: no use on any branch or worktree) | fresh | first N, with N frozen by §3; the remainder stays unopened |

## 2. Step 1A — existing-output audit (CPU; dar1 confirmation outputs)
- **Inputs:** `ckpt/anchor_refit/conf_latents.pt` (FP50, B0, BA@330, BL@330, BL-tm, BL-arch; 128 × 4 × 32 × 32) and `results/anchor_refit/conf.json`
  (per-image rel_mse, psnr, lpips).
- **Verify:**
  - seed list and order: the conf seeds are 124000–124127 in batch order;
  - shapes, finiteness and arm keys;
  - recompute rel_mse = ‖x − x_FP50‖² / ‖x_FP50‖² per image from the latents and compare with the stored values;
  - checkpoint identities against artifacts.json;
  - recorded call counts, precision and CFG settings from conf.json / checks.json.
  - PSNR and LPIPS are the original per-image values (VAE decode + VGG-LPIPS of `weight_anchor.controls.metrics`). A GPU spot-check
    re-decodes 8 images per arm (mechanism stage) to verify export.
- **Report** for B0, BA@330, BL@330, BL-tm and BL-arch:
  - absolute mean / median endpoint rel. MSE;
  - ratio of means (the original primary estimand) with the original paired bootstrap (10,000 draws, numpy seed 0);
  - mean paired difference with its CI;
  - median paired difference, wins, ΔLPIPS and ΔPSNR;
  - per-image paired scatter and distributions;
  - share of each arm's total error, and of each contrast's total paired difference, contributed by its largest 1, 5 and 10 images;
  - leave-one-image-out ratio range (**diagnostic only**).
- **Keep every valid image.** No trimming, winsorizing, metric switching or deletion.
  - A data/implementation bug is defined independently of outcome: seed mismatch, non-finite output, a stored metric not reproducible from the stored
    latents, or a call-count mismatch.
  - Only such a bug triggers a consistent rerun of the affected arms.
- **"5.5×" definition:** recovered from `diag.json` / `interval_stats`.
- Fixed examples are repeated only if needed to separate nonrepeatability from image variation; the pilot path is deterministic at fixed batch (dar1
  bitwise reproduction checks), so no repeat is planned.

## 3. Step 1B — evaluation size, frozen from old data before any new output
- **Planning alternative:** a true 10% reduction in mean endpoint error (ratio 0.90), *not* the pilot's observed 0.835.
- **Variance model:** paired image distribution of (BA@330, BL@330) on 124000–124127 under a multiplicative shift: BA values × (0.90 / 0.835), so that
  the population ratio of means is 0.90 with the observed heavy-tailed image spread.
  - For N ∈ {128, 256, 512}: 2,000 simulated cohorts resampled with replacement from the 128 pairs.
  - Each cohort gets the percentile-bootstrap CI (1,000 draws).
  - Reported: projected median CI half-width, and the probability that the upper bound is < 1 (power).
  - Delta-method SE of the ratio is reported as a cross-check.
- **Aggregate over two new pairs:** simulated by pairing the same resampled images with the BL@330 and BL-tm outputs as two "comparator realizations"
  (their correlation is the available proxy for between-pair correlation). Labelled approximate.
- **Limits:** 128 images bound tail estimates. The largest-image share (§2) shows how much a single extreme image can move the ratio; resampling
  cannot produce tails heavier than those observed.
- **Cost model:** measured dar1 ledger rates.
  - Anchored arm ≈ 0.064 A30-h per 128 images, including decode and LPIPS; FP50 ≈ 0.022 h per 128.
  - Training per pair ≈ 0.39 (BA) + 0.36 (BL incl. BL-tm) + dev ≈ 0.12 h.
- **Freeze rule:**
  - Fresh N = the largest of {512, 256, 128} for which this sum stays ≤ 4.0 A30-h:
    - spent so far + mechanism (projected) + two pairs (training + dev) + kernel track (0.3 planned) + fresh evaluation of the required arms at N
      + 0.4 reserve.
  - Required arms: FP50, B0, BL-arch, and BA-sel / BL-sel / BL-tm for each of the two new seeds (8 anchored arms + FP50).
  - If even N = 128 does not fit, stop before training and write a reduced-budget proposal.
  - N is recorded as a dated amendment before training starts and is not changed after any new output is seen.
  - If projected power at the frozen N is < 0.8 for the 10% alternative, the study is labelled a **bounded replication, not a decisive test**.
- **Pilot pair on the fresh cohort:** BA@330, BL@330 and BL-tm (seed 0) are added only if, after both new pairs' training and dev are complete, the
  remaining budget covers the required arms plus the pilot arms with 0.25 h left over.
  - This is decided by measured spend only, before generation, and recorded.
  - The pilot pair never replaces a new pair.

## 4. Step 2 — mechanism audit on common states (60200–60215)
- **Arms:** B0, BL@330 (pilot BL-sel), BL-tm (pilot), BL-arch, BA@330 (pilot BA-sel). Read-only use of frozen checkpoints.
- **Common states:** one B0 corrected trajectory per image (`sample(CB, diag)`): x_t for t = 0…49, stored *before* anchor operations at anchors.
  - At each x_t, with the quantizer step index set to t:
    - each arm's guided cheap prediction Q_θ(x_t, t) (the 4-channel scheduler-consumed prediction after CFG);
    - the FP prediction H(x_t, t), evaluated once and shared by all arms.
  - e_θ,t = Q_θ − H.
  - The static quantizer has no state, so nothing needs initializing (§0). Traces are saved as fp32.
- **Measures (all on the same states):**
  1. Raw error energy Σ_t ‖e_t‖², normalized by the common Σ_t ‖H_t‖² (identical for all arms); per-step profile.
  2. For each anchor interval I = [a, a+4), a = 0…44:
     - the interval mean ē_I and the deviation energy Σ‖e_t − ē_I‖²;
     - descriptive only; not proof that the mean is removable at the endpoint.
  3. Held residual r_t = e_t − e_a at non-anchor t:
     - absolute energy, and its ratio to B0's and to BL@330's;
     - the within-interval weighted sum H_I = Σ_j w_j (e_{a+j} − e_a) with the actual DDIM coefficients;
     - bridge estimate v_I = c_I (e_{a+4} − e_a) and residual ‖H_I − v_I‖².
     - These are linearized coefficient-only latent approximations, labelled so.
  4. Lagged cross-products Σ⟨e_t, e_{t+L}⟩, correlations and increment energies Σ‖e_{t+L} − e_t‖² for L = 1, 2, 4, 8, plus the mean-offset
     contribution. Correlation alone is not evidence: a constant offset raises it.
  5. **Live controlled replay** (bridge accounting), for every interval a ∈ {0, 4, …, 44}. Grouped early 0–12, middle 16–28, late 32–44; all 12 are
     reported.
     - **Start:** the common B0 corrected-trajectory latent x_a before a's anchor operations.
     - The arm measures d_a = Q_θ(x_a) − H(x_a) and takes the DDIM step with H(x_a).
     - Steps a+1…a+3 use Q_θ − d_a.
     - At a+4: x⁻ = latent before the kick; d_{a+4} is measured at x⁻; kick k = −λ c (d_{a+4} − d_a); x⁺ = x⁻ + k.
     - **Reference:** the FP DDIM continuation from x_a to x^F_{a+4}.
     - **Report:**
       - ‖x⁻ − x^F‖² and ‖x⁺ − x^F‖² (absolute, and relative to ‖x^F‖²);
       - cos(k, x^F − x⁻) and ‖k‖ / ‖x^F − x⁻‖;
       - image-bootstrap CIs over the 16 images (descriptive).
     - Same for the two-interval window (a → a+8, BA's training window; the a+8 endpoint after its kick).
     - An exact scalar-coefficient replay (kick = coefficient × Δd recomputed offline) is run as an arithmetic check only, separate from these live
       measurements.
     - Bridge kicks are never subtracted from prediction-space residuals.
  6. **Standalone vs anchored** (scope, not a gate): B0, BL@330, BL-arch and BA@330, each as a plain 50-cheap-call sampler (no anchors, no bridge)
     and as the anchored sampler, plus FP50, on 60200–60215 at batch 16. Endpoint rel. MSE, PSNR and LPIPS.
- **Readings:**
  - "Error shaping supported" requires lower **absolute post-correction** error for BA than B0/BL on the replay (x⁺ error) and common-state
    measurements. A higher raw-error correlation or a lower residual/raw ratio is not enough.
  - Higher raw energy with unchanged corrected error is not shaping.
  - None of this substitutes for the fresh full-trajectory evaluation, and no mechanism result gates the replication. Only a correctness failure or
    the budget stops the study.

## 5. Step 3 — two new training-seed replications
- **Training seeds s = 1, 2:** the draw-list generator seed (torch Generator seed s; the pilot is s = 0).
  - This changes the data and window order only.
  - Everything else is frozen:
    - base, quantizer, tables, TRAIN set, cached B0 anchor states, den[a], loc;
    - optimizer and LR, batch, branch parameterization and initialization (the original carry), λ;
    - N = 330 and checkpoints 82 / 165 / 330 (the pilot profile, not re-profiled);
    - BL-tm rule: time cap min(BA_s training time, 1260 s);
    - BA and BL of the same seed share the same draws.
  - No tuning.
- **Retry rule (predeclared):**
  - A non-finite loss or gradient stops that run.
  - The last exported checkpoint serves as the final one, and the event is recorded as a training-stability result.
  - No retry with changed settings.
  - A process crash (OOM or infrastructure) resumes from the bitwise-verified resume file.
- **Selection (the pilot rule):** per seed and objective, the lowest dev mean among {82, 165, 330}, ties to the earlier.
  - Selected checkpoints are frozen before the fresh evaluation.
  - No promotion gate: every seed is evaluated, whatever its dev result.
  - Training seeds are not selected.
- **Checks (`repchecks`, before training):**
  - draw list s = 0 reproduces the pilot draw sha;
  - BA and BL with s = 0 for 3 updates reproduce the pilot logs (loss and grad norm) exactly;
  - base W4 / table hashes unchanged;
  - trained-seed export: installed exported carry ≡ trainable hook output (bitwise, one call);
  - resume parity (2 + 2 vs 4 updates, s = 1);
  - teacher receives no parameter gradients.
  - Eval call counts are asserted per trajectory.
  - The pilot's FD and gradient-path validations apply unchanged, because the code path is identical. Only the draw seed is new.
- **Cost:** about 0.74 h training + 0.12 h dev per pair (pilot ledger). If a pair is projected to exceed 1.0 h, report before launching; runs are
  never shortened.

## 6. Step 4 — fresh evaluation (128000–128000+N−1, frozen N)
- **Arms:** FP50, B0, BL-arch, and for s = 1, 2: BA-sel_s, BL-sel_s, BL-tm_s. Optionally the pilot (s = 0) arms (§3).
  - Same FP50 reference and noise; batch 32; abn1 `sample`; counts asserted.
- **Primary:** BA-sel_s / BL-sel_s per new seed, as a ratio of mean endpoint latent rel. MSE.
  - Aggregate R̄ = ½(R_1 + R_2).
  - **Joint image bootstrap:** 10,000 draws, numpy seed 0. The same resampled image IDs apply to all arms and both pairs. Outputs of one image are
    not independent images.
- **Also reported:**
  - per pair: CI, mean paired difference + CI, median difference, wins, ΔLPIPS + CI, ΔPSNR + CI;
  - secondary contrasts BA-sel_s / BL-tm_s, BA-sel_s / BL-arch and BA-sel_s / B0;
  - a training-seed variability table (s = 1, 2, and 0 where available);
  - an exploratory nested bootstrap over seeds and images, labelled fragile at 2–3 seeds.
- **Reading (fixed now):**
  - **Confirmed conditional mean-fidelity improvement:** the aggregate R̄ CI upper bound is < 1 **and** both new pairs' point estimates are < 1.
    Each pair's CI is reported even if it crosses 1.
  - Aggregate improved, but one pair ≥ 1: **mixed training stability**, not robust superiority.
  - Aggregate CI crosses 1: the latent conclusion stays **inconclusive**. LPIPS is reported separately as secondary.
  - A significant LPIPS regression (ΔLPIPS CI wholly > 0) is reported prominently.
  - No broad-superiority claim if BL-tm or BL-arch remains better than BA.
  - Fidelity only: no FID, perceptual-equivalence, speed or readiness claim.
- No further N increases or objective changes after this evaluation. The evaluated cohort is recorded in the registry.

## 7. Budget and stopping
- **Planning split (A30-h):** audit/mechanism 0.4; training 1.6; evaluation 1.3; hardware 0.3; reserve 0.4.
  - Reallocation within 4.0 is recorded as an amendment.
- Each GPU stage checks `h1031b_budget.ok` with an estimate from measured rates and stops if the total would exceed 4.0.
- GPU jobs are serialized. Training runs never share the GPU.
- Stages are resumable; exact resume commands go in RUN.md.
- Stopping at the cap reports which stages remain and which seeds stay unopened.

---

## Amendments and results
(appended below, dated)
- 2026-10-31 14:07 UTC, **step 1A audit complete (CPU; `audit_1a.json`, `audit_1a.png`, `audit_1a_per_image_rel_mse.json`).** No implementation/data bug by the predeclared criteria: all arms finite, shapes 128×4×32×32, per-image rel. MSE reproduced from the stored latents (max rel. diff 2.4e-7), call counts 50+13/12 (FP50 50), arms mutually distinct, checkpoint hashes match the manifest, stored ratios/CIs reproduced exactly. **"5.5×" = squared-L2 energy** of the per-step 4-channel guided prediction error summed over 50 steps × 8 images (60208–60215), on **FP50 teacher-forced states (common across arms)**: BA 16,468 vs B0 2,987 = 5.51× (relative-L2 equivalent 2.35×); the live-trajectory ratio is similar (16,425 / 2,980). Primary BA@330/BL@330 0.835 [0.489, 1.423]; mean paired diff −7.2e-4 [−2.6e-3, +1.7e-3]; median diff −1.6e-4; wins 79/128; ΔLPIPS −0.0063 [−0.0118, −0.0005]; ΔPSNR +1.31. Heavy tail: one image (124092: BA 0.128 vs BL 0.014, B0 0.070) is BA's largest error (27.6% of BA's total) and the largest |paired difference| (against BA; −124% of the total signed difference); leave-one-out ratio range 0.620–0.891 (diagnostic only; the image stays in every analysis).
- 2026-10-31 14:07 UTC, **step 1B frozen: N = 256** (`power_1b.json`; 128000–128255; 128256–128511 stay unopened). Cost projection (dar1 ledger rates: anchored arm 5.27e-4 h/image incl. decode+LPIPS, FP50 1.72e-4): fixed 0.15 mechanism + 1.73 two pairs (train + dev) + 0.3 kernel + 0.4 reserve; evaluation 1.12 h at 256 → 3.70 ≤ 4.0; 512 → 4.83 (does not fit). Projected precision under the 10%-reduction alternative (shift model): single-pair median CI half-width 0.47 / 0.34 / 0.25 at N = 128 / 256 / 512 (delta-method SE 0.26 / 0.19 / 0.13); approximate two-pair aggregate half-width 0.38 / 0.28 / 0.20; probability of meeting the confirmation rule ≈ 0.16–0.21 at every N (the percentile bootstrap undercovers with these tails at small N, which is why the 128 figure is not lower). **This study is therefore a bounded replication, not a decisive test.** Limits: tails cannot exceed those of the 128 observed pairs; the aggregate proxy shares BA outputs across the two comparator realizations.
- 2026-10-31 14:15 UTC, correction of record: the two amendments above were first written with wrong clock times (14:40 / 14:45); the actual time is 14:07 UTC (commit 3274296), before any new GPU output. Mechanism stage launched 14:11 UTC (a first launch at 14:10 failed at import — missing external/DiT link in the new worktree — before any GPU work or ledger entry).
- 2026-10-31 14:31 UTC, **mechanism stage complete** (`mech.json`, `mech_analysis.json`, `mech.png`; 0.256 A30-h; parity checks: B0 traces equal the sampler's own diagnostic errors exactly (max |diff| 0), static quantizer repeat bitwise, anchored B0 endpoint = trajectory endpoint bitwise, conf export spot-check PSNR/LPIPS within 1.4e-4 / 2.5e-6 of stored). Results are reported in REPORT.md; none gates the replication.
- 2026-10-31 14:31 UTC, **repchecks attempt 1: pass = false due to a check-implementation bug** (kept on record as `pass_original_attempt`). The provenance check compared full sha256 strings with equality against `expected` values, some of which are 16-hex prefixes; every hash matches by prefix (base f017af87, carry b4a537b0, tables b73030de, smoothing 7ee15c0c, C_L ac201610, …). All substantive checks passed: draw list s = 0 reproduces the pilot sha; BA and BL with s = 0 reproduce the pilot's first 3 (loss, grad-norm) pairs exactly; s1/s2 draws distinct; resume bitwise (s = 1); trained-carry export ↔ install bitwise; teacher gets no parameter gradient; window counts 9/3/2; W4 codes and tables unchanged. Fix: the criterion is re-scored from the recorded provenance with prefix matching (no GPU rerun; criterion unchanged in substance) → pass = true; code fixed for future runs.
- 2026-10-31 16:19 UTC, training and dev complete for both new seeds, and selections frozen before any fresh output.
  - **s = 1:** BA 330 updates in 1389 s; BL 330 + BL-tm 561 updates in 1260 s; all finite. Dev selected BA_s1@82 (0.00222) and BL_s1@82 (0.00260).
  - **s = 2:** BA 330 updates in 1387 s; BL-tm 557 updates in 1261 s; all finite. Dev selected BA_s2@330 (0.00204) and BL_s2@82 (0.00348).
  - Dev FP50 equals the pilot dev FP50 bitwise.
  - **Pilot-arm decision** (by measured spend only, `eval_plan.json`): 2.082 + 1.124 + 0.405 + 0.25 = 3.86 ≤ 4.0, so the pilot arms BA@330, BL@330 and BL-tm (s = 0) are included.
  - Fresh evaluation of 128000–128255 started 16:17:52 UTC.
