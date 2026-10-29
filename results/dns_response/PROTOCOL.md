# DNS response pilot: trajectory-calibrated band-wise latent correction on top of faithful DNS (protocol, 2026-10-29; committed before any candidate result)

Stage tag `r1`. Branch `dns-response-pilot` (worktree `/home/boa-dit-dnsr`), from `park/2026-10-29/dns-cov` = b06404f5bd721b1a853c206475dccc9ac0aaacb1
(no newer DNS work existed when it was checked; there is no AGENTS.md in the repository). Outputs: `results/dns_response/`, `ckpt/dns_response/`.
This is a **new exploratory protocol**, not a previously registered experiment. User specification: the "Overnight DNS extension pilot" instructions of 2026-10-29.
The sections below implement them, with concrete choices fixed now.

## 0. Budget and queueing
- **Hard ceilings:** 6 additional GPU-hours (every `Run` record of stage `r1` counts: profiling, failures, calibration, teacher calls, evaluation),
  **and** 6 elapsed hours from this queue's first GPU job. No additional GPU.
  Each stage checks both ceilings before starting (`budget.json`). Stages are cut in the order of §8.
- **GPU work starts only after the DiT joint pilot has completely finished.** Its launcher process must have exited, its queue log must contain `DIT_JOINT_DONE`, and `results/dit_joint/REPORT.md` must exist.
  A GPU lock file (`ckpt/dns_response/gpu.lock`) is held while this queue uses the GPU. CPU preparation (protocol, code, CPU-only checks) may run earlier.
- The joint pilot and PixArt are not touched.

## 1. Bed (frozen)
- DiT-XL/2 256, heig-8 W4 + the original rank-8 carry, A16, DDIM-50, cfg 1.5; the VAE and CFG convention of the DNS reproduction.
- Faithful DNS = `boa_dit/dns/port.py` with its frozen calibration `results/dns/stats.json` (sha recorded), W_u = 0.2, per-image DCNC generators (seed·7919 + 7).
- No refitted branch, no activation feedback. Network parameters are fixed.

## 2. Seeds (sweep of every branch incl. dit-joint-feedback, 2026-10-29: 72000–75999 unused)
| set | seeds | use |
|---|---|---|
| fit | 72000–72063 (64) | moments, response matrix, fitting, acceptance |
| development | 72100–72131 (32) | iterate selection (common objective + fidelity guard) |
| sealed fidelity confirmation | 72200–72327 (128) | opened once, after both candidates are frozen |
| sealed generation evaluation | 74000–75999 (2,000) | FID-2k (exploratory points), KID with real reference features |

Labels come from `labels_for_seeds`, identical across arms; noise from `noise_for_seeds`. Sampling uses aligned batches of 32.

## 3. Correctness checks (`correctness.json`; failures block dependent stages)
1. **Zero correction == faithful DNS** through the identical path: bitwise (same code path; θ = 0 skips the add).
2. **Exactly 50 denoiser calls per image** at inference (counter); no diagnostic teacher call inside the sampler.
3. **P_low** (= `objective.d2.band_masks(32)[0]`, the lowest radial band incl. DC, the band used in the DNS/C-DNS analysis) applied as Re(IFFT(FFT(x)·m)):
   - real output, imaginary part ≤ 1e-6 relative;
   - idempotent ≤ 1e-6;
   - orthogonal (⟨P x, y − P y⟩ ≈ 0) ≤ 1e-6;
   - mask conjugate-symmetric;
   - P_low x + P_comp x == x ≤ 1e-6.
4. **The correction is applied to the latent state after the DNS update of step j** (before the next denoiser call), never to the prediction or the decoded image.
   Asserted by a hook-free trace: the state after j changes, and the prediction at j is unchanged.
5. **Final-step, DCNC and boundary handling unchanged:** the schedule dict is identical to faithful DNS's (asserted).
6. **Identical per-image DCNC streams** across paired runs and finite differences: generators are recreated per call (asserted by repeating a run bitwise).
7. **Moment identity closes:** Var(x_DNS) − Var(x_FP) == Var(d) + 2·Cov(x_FP, d), same centering, ≤ 1e-6 relative.
8. **Repeat baseline:** the same small DNS run twice gives the numerical variation (expected 0; if not, its magnitude is the resolution floor).

## 4. Component removals (fit seeds 72000–72063 and dev seeds; context only; they do not choose the backbone)
Removal ablations of faithful DNS's frozen calibration. They are not independently optimized algorithms.
1. plain heig-8 DDIM;
2. DNS affine correction only (ε ← (ε − b)/(1 + k)), ordinary DDIM coefficients (ā′ = ā_prev, C3 = 1), DCNC off;
3. DNS schedule/rescale only (ā′, C3 from the frozen schedule), affine identity, DCNC off;
4. affine + schedule/rescale, DCNC off;
5. faithful full DNS.

Saved per arm: endpoint latent rel. MSE vs FP, decoded PSNR/LPIPS (dev, if the budget allows), ensemble means/variances at the checkpoints, and per-step state RMS.

## 5. The actuator (named "band-wise latent impulse", **not** a per-band timestep)
- Updates are numbered 0–49. After the DNS update of steps j ∈ {9, 19, 29, 39, 48}: x ← x + θ_j·P_low(x).
- P_low is shared over the 4 channels. There are five scalars; no per-channel fit, mean shift, branch or inference teacher.
- **Trust region:** |θ_j| ≤ 0.01 and Σ|θ_j| ≤ 0.02 (pilot choices, not theory). Binding is recorded.
- **Checkpoints:** the states right after each insertion (after updates 9, 19, 29, 39, 48) and the final output (after update 49): six in all.
  FP states at the same sampler indices come from FP DDIM trajectories (same noise and labels).
- **Statistics per checkpoint:**
  - per-channel ensemble DC mean (sample mean of the spatial mean);
  - per band b ∈ {1..4}: centered ensemble variance, i.e. each spatial element of the real band component Re(IFFT(FFT(x)·m_b)) is centered across samples,
    its across-sample variance is summed over elements and divided by the band's real degrees of freedom (n_b = number of mask frequencies), then averaged over channels.
  - Parseval makes this equal to the band's per-coefficient variance; this is verified in check 7's code.
- **Residual vector m(θ):**
  - mean group: (DC mean_DNS − DC mean_FP)/max(sd_FP,DC, floor), 4 channels;
  - variance group: log(V_b,DNS / V_b,FP), 4 bands; floors on both, frozen from fit data;
  - six checkpoints × (4 + 4) = 48 entries;
  - weights: every entry 1/√4 within its group, so the groups have equal total weight and checkpoints weigh equally;
  - **objective J = ‖w ⊙ m‖²**. Floors and their activity are reported.
- **Diagnosis** (reported, never fed to a solver): Var(x_DNS) − Var(x_FP) = Var(d) + 2·Cov(x_FP, d) per band and checkpoint, with its sign kept.

## 6. Local (L) vs trajectory-response (T) calibration, matched parameters and information
- **Response matrix G (48 × 5)** from **full live rollouts** from initial noise on the fit seeds:
  - central differences θ_j = ±0.002, checked at ±0.004 (same images, same DCNC draws);
  - column norms and off-time blocks saved.
- **Resolution test:** a column is usable if ‖G_j(0.002) − G_j(0.004)‖ / ‖G_j(0.004)‖ ≤ 0.25 **and** its effect exceeds the repeat-run variation (check 8).
  - Otherwise: one retry in fp32 inference at ±0.004 / ±0.008. If the fp32 retry is the one adopted, it applies to **both** candidates (same numerical setup and cache).
  - Unresolved columns are fixed at θ_j = 0 for both candidates. A resolution failure is reported as a measurement limitation.
- **Candidate L (local):**
  - for impulse j, only the block of m at checkpoint j (the 8 entries right after impulse j) and only G's own column at that checkpoint, g_jj;
  - joint constrained quadratic Σ_j ‖w ⊙ (m_j(0) + g_jj θ_j)‖² under the trust region;
  - cross-time responses are excluded.
- **Candidate T (trajectory):** ‖w ⊙ (m(0) + G θ)‖² over all 48 entries, under the trust region.
- **Solver:** both convex QPs are solved by SLSQP on the split variables θ = p − n (p, n ≥ 0), with a 1e-9·‖θ‖² minimum-norm tie-break and no hyperparameters.
  Saved: objectives, G, singular values / condition number, θ, predicted J.
- **Acceptance on fit (each arm):** a live rollout with θ.
  - Accept if the actual J (that arm's objective: local blocks for L, full for T) improves over the previous iterate **and** latent rel. MSE ≤ 1.25 × faithful DNS.
  - Otherwise backtrack ×½, then ×¼, else keep the previous iterate (possibly zero). Every attempt is logged.
- **One refresh per arm:** G is re-measured at the accepted θ₁ (same perturbation protocol) and refitted from θ₁ under that arm's objective.
  The final θ stays inside the original trust region (constraints on θ₁ + Δ). The same acceptance rule applies.
- **Development selection (each arm)** among its zero / first / second accepted iterates:
  - criterion: the **common** full-trajectory objective J on dev seeds;
  - requires an improvement over zero and the 1.25× fidelity guard;
  - ties go to the smaller ‖θ‖; no per-seed selection, no FID.
  - Identity (zero) may be selected, and that is reported.
- **Claim scope:** if T beats L, credit downstream-response calibration, not covariance modelling per se.

## 7. Frozen evaluation
- **Fidelity confirmation** (seeds 72200–72327, n = 128): plain heig-8, full DNS, L, T (identical zero candidates deduplicated).
  - Metrics: per-image latent rel. MSE, PSNR, LPIPS vs FP; paired bootstrap over images (2,000 resamples), medians, win counts.
  - Also: moment errors, state RMS, correction magnitudes, cross terms, predicted vs actual responses.
  - Costs: time per image (diagnostics off), coefficient bytes (5 × fp32), FFT cost, denoiser calls.
- **Generation evaluation** (budget permitting; seeds 74000–75999):
  - FID-2k for full DNS, L and T (plain heig-8 optional), with the same reference statistics (`results/stage0/ref_stats.npz`); features and latents retained.
  - **KID** against real reference features: InceptionV3 pool3 of the 10,000 ADM reference images (computed once, retained).
  - FID-2k values are **exploratory finite-sample points**, without percentile-bootstrap CIs (the earlier raw percentile intervals do not reproduce the point estimates and are not used).
  - No choice between L and T is made from these numbers.
- **Interpretation rules:** as in the user specification §7. A gain of about 0.5 FID-2k vs both DNS and L would motivate replication; it is not a significance threshold.

## 8. Budget plan and cut order
| item | planned GPU-h |
|---|---|
| checks, component removals, FP/DNS caches | 0.75 |
| responses, fits, one refresh | 2.0 |
| paired confirmation | 0.5 |
| generation (FID/KID) | 2.25 |
| reserve | 0.5 |

- **Cut order:** baseline FID → second refresh (both arms) → component decoded metrics.
- **No unequal-sample FIDs.** Held-out samples are not reduced after inspection.
- **Stop rule:** non-finite states or state RMS > 2 × the same-seed DNS RMS stop a candidate (step and diagnostics recorded).

---
## Amendments
(dated; before the affected stage)
