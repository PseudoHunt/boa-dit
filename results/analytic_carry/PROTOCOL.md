# Analytic carry correction and trajectory-response calibration (protocol, 2026-10-30 IST; frozen before any candidate result)

Stage tag `ac1`. Branch `analytic-carry-pilot` (from `dit-joint-feedback` @ 2c801728f75f9d855b47100d81bd05384ce170da; DNS-side inputs from
`dns-response-pilot` @ c57529558e38ba192a8dacbb7a53ef5918015483). Code: `boa_dit/analytic_carry/run.py`; report: `scripts/make_analytic_carry.py`;
launcher: `scripts/run_ac1.sh`. Outputs `results/analytic_carry/`, checkpoints/caches `ckpt/analytic_carry/` (not uploaded).
The user's instruction of 2026-10-30 is the authoritative specification; this file records how it is implemented.

## 0. Reconciliation (2026-10-30, before any GPU work)
- No AGENTS.md exists in this repository; `CLAUDE.md` and the joint/DNS reports were read. No GPU process was running; no unpushed work on any branch.
  No earlier carry-response / analytic-carry output or protocol exists anywhere (searched all worktrees and branches). E1 therefore has **not** run before; it runs under the rules below.
- Completed experiments are not rerun or overwritten. Old reports stay; corrections are added as dated notes (§1).
- Not resumed: PixArt, full C-DNS, optimizer sweeps, the old joint training queue.
- **Budget:** new ceiling 4.0 GPU-h for this campaign (everything: teacher calls, failed runs, profiling, responses, evaluation), separate from completed runs.
  Planning allocations: audit/checks 0.15 h, E1 1.10 h, E2 2.25 h, reserve 0.50 h. Ledger = rows with stage `ac1` in `results/gpu_hours.csv`
  (unique attempts, failed ones included) + `results/analytic_carry/budget.json`. Each stage refuses to start if spent + its profiled estimate exceeds the ceiling;
  E1 stages must additionally leave the profiled E2 core untouched. Long stages check the ceiling between models and stop safely with partial outputs (exit 5).
  No FID in this allocation. A concrete schedule from on-machine profiling (`profile.json`) is saved before the long stages.

## 1. Reporting fixes (no new sampling)
- **Coherence.** Joint code: C = T·||mean_t e_t||², I = mean_t ||e_t||², reported C/I = ΣC/ΣI over images (≈ 9.3).
  New correctly named field **c = ||mean_t e_t||² / mean_t ||e_t||² = (C/I)/T** (T = 50; 9.3 ↔ 0.186). Both the ratio of sums across images and the mean of imagewise ratios are reported.
  The old field is kept with its definition. Raw e_t tensors were not stored, so the check is against the stored per-image C and I lists only. → `audit.json`, plus a dated note in `results/dit_joint/REPORT.md`.
- **DNS budget.** `budget.json` of dns-response-pilot logs spend *before* each stage, so the last stage (fid_plain ≈ 0.43 h) is missing from its 2.51 h.
  The actual total is computed by summing the unique r1 rows (failed attempt included) of that branch's `gpu_hours.csv`, not hardcoded. → `audit.json` + a dated correction note in `results/dns_response/REPORT.md` (on that branch).
- **Feedback/refit overlap:** skipped. Aligned prediction-error tensors for A, B, C_L, D_L were never cached (only final latents); no new GPU collection for this optional diagnostic.

## 2. Bed, data, correctness
- Bed: the joint pilot's COMMON smoothed DiT-XL/2 W4 rank-8 base (`ckpt_deflate_heig_r8_w4_s`, carry `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt`),
  static single8 A8 tables (`ckpt/dit_joint/static_single8.pt`), feedback OFF, DDIM-50, cfg 1.5, the deployed fp16-autocast sampler (`sample.ddim_sample`). No DNS.
  Codes, scales, smoothing, activation encodings, biases, exclusions and rank are frozen (hashes checked against `results/dit_joint/base.json` / `calib_act.json`).
- All changes are exported into the existing carry tensors (V [d_in, 8], delta = D [d_out, 8]; carry(x) = (xV)Dᵀ, fp16) and installed by the deployed `deflate.carry.install`.
  Both experiments start from the ORIGINAL carry C0 of the common model.
- Inference quantizer: `QuantNL` = JointQuant 'static' without the per-call clip-count GPU synchronization (pure speed); checked bitwise against JointQuant.
- **Seeds** (verified 2026-10-30 against every branch's code and protocols, and `handoff/SEED_REGISTRY.md`: no prior use of 82000–83327; no replacement needed):

  | | fit | development | confirmation (128, opened once) |
  |---|---|---|---|
  | E1 | 82000–82031 | 82100–82131 | 82200–82327 |
  | E2 | 83000–83031 | 83100–83131 | 83200–83327 |

  Neither experiment's confirmation set is used as development for anything. E1 and E2 are not compared head-to-head (different seeds).
- Batch structure: 32-image sets in one batch of 32; confirmation in 4 batches of 32; h checks on the first 8 fit images in a batch of 8 (all compared runs in the same batch structure).
- **Checks before quality** (`checks.json`; failure blocks the quality stages):
  zero coefficients reproduce the baseline bit-exactly through save → load → install (E1 and E2); E1 all-ones reproduce `carry_C_L_u200.pt` (tensors and samples, bitwise);
  base/carry/tables/smoothing hashes unchanged; carry shapes rank 8, 112 sites; exactly 50 denoiser calls per site per sample; exactly one forward (carry) hook per linear
  (no doubled rank / extra branch); repeated baseline on four fit images (repeat noise recorded; batch-4 vs batch-32 difference recorded as context);
  QuantNL ≡ JointQuant static (bitwise); CFG ordering and sampler indexing: guided ε at recorded state i vs the sampler's own CFG-combined ε at step i, ≤ 1e-4 relative;
  E2 site pairing (112 sites, 64 tokens × both CFG rows per image, finite), FP teacher has no quantization pre-hooks, biases student = FP recorded;
  ridge unit checks (stationarity < 1e-10, vs direct least squares < 1e-8, λ = 0 exact fit < 1e-8, V unchanged, rank-8 shapes).
- FP endpoints: the FP model (no activation quantization) from identical noise/classes, same batch structure. Metrics as in the joint report:
  final-latent relative MSE ||x − x_FP||²/||x_FP||² per image, PSNR and LPIPS-VGG of decoded images vs FP.

## 3. E1 — calibrating an existing Adam update
Source: joint-pilot `C_L` at exactly 200 updates, `ckpt/dit_joint/carry_C_L_u200.pt` (sha256 recorded in checks.json), chosen regardless of quality.
Depth groups g: blocks 0–6, 7–13, 14–20, 21–27 (all four linears per block). V(a) = V0 + a_g(V1 − V0), D(a) = D0 + a_g(D1 − D0) (fp32, export fp16).
Rank preserved; the effective weight path is quadratic in a (not linear). Bounds a_g ∈ [−0.25, 1].
Configurations: B (original carry, a = 0), A (C_L200, a = 1), S (a_g = α, α ∈ {0, 0.1, 0.3, 0.5, 1}), L, R (§5).
Claim scope: calibration of an existing trained update; the original C_L training cost (0.237 GPU-h, `results/dit_joint/train_C_L.json`) is accounted separately. Not training-free.

## 4. E2 — analytic directions without Adam
**4a statistics.** Original common W4A8 model (C0) samples E2 fit seeds; its own states at execution steps {0, 5, …, 45, 49} are cached.
At each state the student (C0) and the FP teacher run at the SAME latent, timestep and conditioning (both CFG rows, the deployed autocast path).
Per block linear (112): X = the student's ACTUAL consumed input (after smoothing and A8) as seen by the carry hook; Z = V0ᵀX;
Yq = the student linear's output without the carry (W4·X + b, frozen student bias); Y = the FP teacher linear's output on the teacher's own internal activation
(so upstream activation differences are included). Outputs are in the same units (smoothing does not change a linear's output); the bias is included on both sides
(student bias = FP bias is checked). Token subsampling, fixed now: token 4k + ((image index within the split + step) mod 4), k = 0..63, the same for every site
and both CFG rows of an image. Equal weight per sampled step, image and CFG row. Streamed float64 means: ⟨ZZᵀ⟩, ⟨(Y − Yq)Zᵀ⟩, ⟨||Y − Yq||²⟩. The same statistics on the development seeds (diagnostic only).
**4b solve.** E0 = Y − Yq − D0Z; ΔD = ⟨E0Zᵀ⟩(⟨ZZᵀ⟩ + λI)⁻¹ by `torch.linalg.solve` in float64 (means, not sums), λ = 1e-4·trace⟨ZZᵀ⟩/8;
zero feature energy → ΔD = 0. Conditioning recorded; no damping changes. V0 fixed; D(a) = D0 + a_gΔD (exactly linear in a before fp16 export); rank 8, no new branch.
Reported: per site/group the local surrogate before/after on fit and dev, after fp16 export, update output energy; check that the regularized surrogate does not increase on the fit features.
Exactness holds only for this fixed-feature regularized local surrogate.
**4c.** Same groups/bounds/§5 procedure. B, A_analytic (a = 1), S_analytic, L_analytic, R_analytic — all with the same V and rank; L/R/S share the same ΔD.
Tiny directions are reported as limited headroom in this fixed-V subspace, not as failure of analytic or rank-8 corrections in general.

## 5. Response calibration and selection (both experiments)
- **Endpoint.** e_i = (F_i(0) − F_i^FP)/||F_i^FP||, R_ig = [F_i(h·u_g) − F_i(−h·u_g)]/(2h·||F_i^FP||), F = the deployed full 50-step sampler. Mean-over-images convention:
  objective(a) = (1/N)Σ_i ||e_i + R_i a||² ≈ mean relative latent MSE.
- **h checks** (first 8 fit images, batch 8): columns at h = 0.1 vs h = 0.05; usable column ⇔ relative difference ≤ 25 % and ||F(+h) − F(−h)|| > 10 × the repeat-run noise norm.
  Mixed perturbation (h, −h, h, −h) vs linear superposition h·Σ m_g R_g: relative error ≤ 25 %. Pass ⇔ ≥ 1 usable column and the mixed test passes.
  Otherwise retry once at h = 0.2 (columns 0.2 vs 0.1, mixed at 0.2). If still unresolved: R fitting is stopped for that experiment (failed approximation check);
  L/S still run. Unusable columns get a zero column (coefficient driven only by regularization → 0).
  The full endpoint response is then measured on all 32 fit images (batch 32) at the accepted h.
- **Local.** At the frozen baseline (C0) states of the fit images at steps {0, 5, …, 45, 49}, guided-ε residual vs FP at the same input, and its central-difference
  responses to the SAME four perturbations (accepted h, or 0.1 if none). Each step normalized by its baseline fit-error energy (mean over images; floor 1e-6 × mean), steps weighted equally.
- **Fits.** min_a ||e + Ra||² + λ||a||², a ∈ [−0.25, 1]⁴, λ = γ·trace(RᵀR)/4, γ ∈ {0.001, 0.01, 0.1, 1} (same mean convention), solved exactly as bounded least squares;
  zero trace → a = 0. Each fitted a × shrink {1, 0.5, 0.25}; zero (a = 0) is included in each family. Identical exported models are deduplicated by carry hash.
  No second response refresh, new basis or larger range.
- **Capacity diagnostic.** Pseudoinverse projection of e on span(R) (SVD tolerance max(M, 4)·eps₆₄·s_max): fraction of FIT squared error explained, singular values.
  Not an oracle for nonlinear sampling or held-out images.
- **Development selection.** S, L, R each choose, among their predeclared candidates, the lowest actual development mean endpoint relative MSE (full sampler);
  ties → smaller ||a||; zero may win (then reported as identity, not as a learned improvement). Candidate counts recorded; L and R get the same development procedure.
  Recorded too: fit-Gram predicted vs actual development change for every candidate; actual vs predicted fit change for the selected S/L/R.

## 6. Confirmation and interpretation
Export hashes of B, A, S, L, R are frozen in `<exp>_select.json` (committed and pushed) before the experiment's 128 confirmation images are opened; identical models are evaluated once.
Report: per-image latent relative MSE, PSNR, LPIPS; paired bootstrap (2,000 resamples, rng 0) CIs of the relative mean change, medians, win counts;
primary contrasts R vs S and R vs L; also R vs B and R vs A (and L vs B, S vs B); correction magnitudes; calibration memory/time; inference cost (unchanged rank 8, no extra call);
fixed examples (confirmation positions 0, 42, 85, fixed now) plus the three worst images of R (or L when R is absent) in `<exp>_grid.png`.
**Prioritization signal (new pilot threshold, not a novelty or publication criterion):** R ≥ 3 % lower mean latent MSE than BOTH S and L with paired CIs below zero,
R better than B, and no clear LPIPS regression. Smaller gains are reported accurately.
Readings: R beats only A → shrinkage may explain it; R beats L but not S → no evidence the richer response solve is needed; all worse than B → keep B.
One negative four-direction experiment does not close weight-side research. E2 success would show a no-Adam trajectory-calibrated carry correction can help on this bed;
it would not show superiority over well-tuned Adam (that needs matched data and total calibration cost).

## 7. Stage order and cuts
audit → checks → profile → e1_resp → e1_select → e1_confirm → e2_stats → e2_solve → e2_resp → e2_select → e2_confirm → report; commit + push after every stage.
If profiling shows the campaign does not fit: optional diagnostics are cut first; E2's core (stats/solve/responses/selection/confirmation) has priority over E1
(E1 stages refuse to start unless the projected E2 core still fits). Confirmation size is never cut after results. A clearly labeled partial result is saved if a comparison cannot finish.
Stop after the report and push: no FID, no second model, no longer training, no new sweep.

---
## Amendments
(dated; appended before the affected step runs)
