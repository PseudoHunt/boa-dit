# Weight-anchor controls (wac1): pre-registration

Written 2026-10-30, **before any result of this follow-up**. Branch `weight-anchor-controls` (from weight-anchor-pilot 52e727c),
results in `results/weight_anchor_controls/`, code `boa_dit/weight_anchor/controls.py`, stage tag `wac1`. Nothing above the marker changes;
deviations are appended below it, dated.

## Questions
1. Do the W4 prediction changes contribute anything beyond holding the last FP prediction?
2. Does any tested configuration offer a useful trade-off against ordinary FP sampling?

**Out of scope:** no method expansion, alternating weights, new kernels, W4A4, DNS changes or retraining. No K = 8 and no FID. Release
assets are preserved (an inventory only; nothing is deleted).

## Budget
**0.5 A30-hour hard cap**, counted in `gpu_hours.csv` rows tagged `wac1`, including checks, profiling, failed and aborted runs.

## Bed (unchanged from wa1)
- **Model and sampler:** heig-8 W4A16 (simulated), DiT-XL/2 256, DDIM-50 (eta 0), cfg 1.5 on ε channels 0–2 (channel 3 from the cond half).
- **Numerics:** fp16 autocast, fp32 sampler, batch 32.
- **Anchors and coefficients:** K = 4 anchors at calls 0, 4, …, 48 (13 FP calls). The β^4 table is frozen from `results/weight_anchor/coefficients.json`
  (copied unchanged into this directory), with s = 1. Hashes as in `results/weight_anchor/provenance.json`.
- **Timing audit:** what each earlier wa1 timing number measures is documented in `timing_audit.md`, written before this protocol.

## Arms (each on its own trajectory, every FP anchor at that arm's current latent, no cross-arm predictions)
| arm | anchor calls (i mod 4 = 0) | non-anchor calls | W4 calls | FP calls |
|---|---|---|---|---|
| FP50 | FP | FP | 0 | 50 |
| Q | W4 | W4 | 50 | 0 |
| A | FP | W4 | 37 | 13 |
| R | FP, plus W4 at the anchor to measure e_a | ε_W4,t − β_{b(a),t−a} e_a | 50 | 13 |
| S | FP, guided output stored | the stored FP guided output, **no denoiser call** | 0 | 13 |
| R1 | as R | ε_W4,t − e_a = ε_FP,a + (ε_W4,t − ε_W4,a) (β ≡ 1, fixed diagnostic) | 50 | 13 |
| FP13 | standard 13-step DDIM (`make_diffusion(13)`, the repository's `ddim13` respacing) | — | 0 | 13 |

- **S** runs the ordinary DDIM update at all 50 scheduled timesteps, using the stored guided output.
- **Auxiliary channels:** channels 4–7 (learned variance) are not consumed by DDIM with eta 0. R, R1 and A modify only channels 0–3.
  S necessarily returns the stored channels 4–7 along with 0–3. Check C4 verifies that channels 4–7 do not affect the DDIM endpoint, so this cannot change any result.
- **FP13** is a diagnostic comparator to S's 13 FP calls. It is a different algorithm (13 updates, not 50).

## Correctness checks (before development; counted in the budget)
- **C1:** on development seeds 92000–92031, Q, A and R reproduce the wa1 cached latents (`ckpt/weight_anchor/live_dev_latents.pt`: w4, anchor_only, reuse_s1) **bitwise**, and FP50 reproduces the cached fp.
- **C2:** S with K = 1 equals FP50 bitwise (no holding occurs).
- **C3:** call counts equal the table above.
- **C4:** FP50 with channels 4–7 replaced by zeros gives a bitwise-identical endpoint.
- **On failure:** if C1, C2 or C4 fails, stop and document the cause before any further stage.

## Development (exploratory)
- **Images:** 92000–92031, already used in wa1, so the results are exploratory.
- **Arms:** FP50, Q, A, R, S, R1 and FP13, with the same metrics as confirmation.
- **Use:** no setting is selected from these results; everything in this protocol is already frozen.

## Timing (before confirmation)
- **Conditions:** complete denoising trajectories on one device (A30), at batch 32 and at batch 1, under the same precision.
  - The models are warm (one discarded warm-up trajectory per arm and batch size).
  - Diagnostics are off: no recording; the call counters are plain integer increments.
  - Timing uses `torch.cuda.synchronize()` around each trajectory, with host wall clock and CUDA events both recorded.
- **Repetitions:** 5 per arm at batch 32 (seeds 92000–92031) and 5 at batch 1 (seed 92000). The median and range are reported.
- **Batch-1** latency is **measured**, never inferred from batch-32 latency.
- **Arms timed:** FP50, R and S (the required ones), plus Q, A, R1 and FP13.
- **Excluded from denoiser timing:** model loading and VAE decode.
- **Memory:** resident model bytes are reported separately: FP alone, W4 alone, and both resident.
  Per-arm peak allocated memory is measured during the trajectory only (peak stats reset before each arm). R and A need both models resident; FP50 and FP13 need only FP; S, as built here, needs only FP.
- **First decision**, pre-registered:
  - if the FP50 median trajectory latency ≤ R's median at batch 32, record **"FP-50 is faster than R"**. FP-50 is also the fidelity reference (zero discrepancy),
    so R has no advantage in the measured latency–memory–fidelity comparison, and no equal-latency FP-k comparison is made;
  - otherwise, time FP-k trajectories for k ∈ {10, 13, 16, 20, 25, 30, 35, 40, 45} at batch 32 (same conditions, proper `ddimk` schedules),
    and freeze the largest k whose median latency ≤ R's median **before** any metric of FP-k is computed. FP-k is then added to confirmation.
  - Batch-1 timings are reported but do not drive the decision. If the two batch sizes disagree, that is reported.

## Confirmation (fresh)
- **Seeds:** 93000–93063 (64), verified unused against `handoff/SEED_REGISTRY.md` and a grep of all worktrees on 2026-10-30.
  The same initial noise and class labels are used for every arm. FP50 is generated once, as the reference.
- **Arms:** FP50, Q, A, R, S, R1, FP13, and FP-k if the timing decision calls for it.
- **Metrics** against FP50: endpoint latent rel. MSE ‖x − x_FP50‖²/‖x_FP50‖², PSNR, and LPIPS (VGG; sd-vae-ft-ema decode). Each comes with
  paired image-bootstrap 95% CIs (2000 resamples, numpy seed 0), per-image win counts and **median per-image differences**.
  The call counts and timing table come from the timing stage.
- **Primary contrasts:**
  1. **R vs S**: does the method beat stale-FP reuse?
  2. **R1 vs S**: do W4 increments help?
  3. **R vs A**: replication of wa1.

  Secondary: S vs A, R vs R1, S vs FP13, and each arm vs Q.
- **Reading of R vs S and R1 vs S** (latent rel. MSE change of R or R1 relative to S, with its CI):
  - **W4-increment signal:** change ≤ −5%, CI upper bound < 0, and ΔLPIPS CI not entirely above 0. This is a prioritization threshold, not a quality claim.
  - **Increments help, below threshold:** CI upper bound < 0 but the point estimate is above −5%. Reported as measured.
  - **S better:** CI lower bound > 0.
  - **No useful increment contribution established, with adequate precision:** CI inside (−5%, +5%) and the first two readings do not apply.
  - **Inconclusive:** anything else. A CI containing 0 is never read as equality.
- **Decision table** (from the user request): applied verbatim in the decision memo.

## Deliverables
In `results/weight_anchor_controls/`:
- PROTOCOL.md and timing_audit.md;
- checks.json;
- dev.json;
- timing.json;
- conf.json (per-image values and paired contrasts);
- release_inventory.md;
- REPORT.md (generated by `scripts/make_weight_anchor_controls.py`, including the call/latency/memory table and the decision memo).

The GPU ledger is `results/gpu_hours.csv`. Seed registry updated. Stop after the controls; no new method with any remaining budget.

---

## Deviations and results
(appended below, dated)
- 2026-10-30, implementation note:
  - The checks and the development metrics ran in one stage (`controls dev`).
  - C1–C4 were evaluated and written to `checks.json` before any dev metric was computed. The stage would have stopped on a failure.
  - No setting was selected from dev.
- 2026-10-30, results (no other deviations):
  - **Checks:** C1–C4 pass bitwise.
  - **Timing** (batch 32, medians): FP-50 19.28 s, R 27.64 s, S 5.07 s, Q 22.63 s, A 21.77 s, R1 27.64 s, FP13 5.02 s.
    **FP-50 is faster than R** at batch 32 and at batch 1 (1.44 vs 2.34 s), so no FP-k was built.
  - **Confirmation** (93000–93063, opened once):
    - R vs S −75.0% [−83.8, −65.2];
    - R1 vs S −75.5% [−83.6, −66.6];
    - R vs A −74.0% [−81.3, −66.6].
    Both readings: **W4-increment signal**.
  - **GPU:** see the ledger. No further runs.
