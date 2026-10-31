# sr1 — SANA INT4 repeatability and persistence diagnostic — pre-registration

Written 2026-10-31, **before any sr1 measurement**. Specification: the user's handoff "SANA repeatability, anchor-aware refit, and W8 anchor spacing"
(2026-10-31), task 1. Branch `sana-repeatability` (worktree /home/boa-dit-sr), from `sana-a30-feasibility` e0ba29f.
- Earlier snk1/snk1b results are kept unchanged in `results/sana_a30/`.
- Scripts: `scripts/sana_repeat.py` (reuses `scripts/sana_a30.py` unchanged).
- Tag **sr1**. Planning allocation 0.6 A30-h within the handoff's 4.0 A30-h total, which is shared with dar1/w8s1 on branch `dit-anchor-refit`
  through a common budget check over both ledgers.
- CPU-only build/debug ≤ 60 min, then report the blocker and continue with DiT. Amendments go below, dated.

## Fixed starting point
- Environment and models: unchanged `/home/nk_env` (Nunchaku 1.2.1 official wheel sha256 82026d55…, torch 2.11.0+cu130, diffusers 0.36.0). The
  checkpoints are pinned as in `results/sana_a30/env.json` (INT4 437e41b9…/320be49f…, BF16 base e2b3c0cb…).
- Workload: 1024², CFG 4.5, 20-step DPMSolverMultistep, flow prediction.
- Already known (snk1b, recorded as prior observations, not re-claimed):
  - every Nunchaku layer is individually non-bitwise, with divergence compounding through depth;
  - the reproducible-discrepancy K4 held residual is 1.31;
  - v1.2.1 has two float-atomic reductions: `lora.cuh:91` (targeted by PR #820) and the LiteLA epilogue `epilogues.cuh:665` (not targeted).

## 1A. Minimal reproducer (stage `repro`)
- Inputs: 4 dev prompts (352, 356, 361, 365) × 5 fixed states (steps 0, 5, 10, 15, 19 of their BF16 trajectories). Saved to
  `ckpt/sana_repeat/states.pt`: latent, timestep, embeddings, masks, CFG order (uncond rows first) and RNG states. These are teacher-forced fixed inputs.
- Per state:
  - 16 INT4 and 16 BF16 calls in inference mode, synchronized, outputs cloned immediately;
  - input checksums (sha256 of the bytes) before and after; all outputs finite.
- Orders: consecutive repeats, and repeats interleaved with a different fixed input (A, B, A, B, …).
- Paths:
  - official direct invocation: `pipe.transformer = Q`, with a single-step hook capturing the transformer output inside the native pipeline at the
    same state;
  - the routed invocation;
  - the bare module.
- Reported: branch-wise pre-CFG outputs (uncond and cond rows) and the guided output.
- Fresh process: 2 states rerun in a separate process with fixed seeds and `CUDA_LAUNCH_BLOCKING=1` (forces synchronous launches), compared with the
  first process.
- Localization:
  - per-layer isolation via the supported `forward_layer_at` (repeat each layer on a fixed input) for 2 states;
  - **within-layer:** the BF16 parts outside the fused stack (patch embed, time embedding, caption projection, final norm/proj) are checked for
    repeatability on the captured inputs.
  - Mutation, buffer reuse (clone-on-return vs not), hidden state (interleaving) and asynchrony (blocking launches) are separated by the tests above.
- No tolerance is relaxed. The deterministic stand-in is not used here.

## 1B. Reproducible error vs call variation (stage `traces`)
- 8 dev prompts (352–359), all 20 states of each BF16 trajectory, **R = 8** INT4 repeats per state. Saved traces: every e_i^(r) = guided Q − guided H,
  fp16, in `ckpt/sana_repeat/traces.pt`.
- Estimates:
  - q̄, p̄ = q̄ − H, V_i = Σ_r‖Q^(r) − q̄‖²/(R−1);
  - repeat-difference ‖Q^(1) − Q^(2)‖²/2 (no further factor of 2);
  - raw, mean-discrepancy and finite-sample-corrected energy ‖p̄‖² − V/R;
  - lag-1/2/4/8 cross-products;
  - K4 held-subtraction energy on the mean, corrected and split-half (4 + 4) cross estimators;
  - prompt-level bootstrap (2,000, seed 0).
- Independence check before interpreting V as noise variance:
  - repeat-order drift (correlation of the deviation with repeat index);
  - lag-1 autocorrelation of successive repeat deviations;
  - Shapiro-free summary: the empirical distribution of per-element deviations, as quantiles.
  - If independence fails, empirical distributions are reported instead of a noise subtraction.
- R is increased to 16 only if the prompt-bootstrap CI of the corrected K4 held residual straddles 1 and is wider than ±0.2.
- This is an offline observability diagnostic, not a deployment ensemble.

## Optional controlled build (stage `build`, CPU, ≤ 60 min total build/debug)
- A separate venv `/home/nk_pr820_env` (never `/home/nk_env`). The pinned v1.2.1 source (af49f249…) is patched with PR #820 head 3713fa96's 4-file diff.
- `TORCH_CUDA_ARCH_LIST=8.0`; CUDA 13.0 nvcc from pinned pip wheels; torch 2.11.0+cu130.
- If it builds, the patched path is compared with the original on identical saved states:
  - repeat spread;
  - INT4-vs-BF16 error, plus original-vs-patched mean shift (deterministic-but-biased is not an improvement);
  - per-call latency.
- The LiteLA epilogue atomic is not modified; that would be a new kernel change and is out of scope.

## 1C. Conditional rerun
Only if an implementation change reduces repeat variance materially (repeat spread ≤ 0.5× original at the five fixed states) **and** keeps INT4
accuracy (mean Q−H energy within ±10% of the original). Then:
1. fixed-state persistence;
2. packed-kernel and loop timing;
3. the 16-prompt H20/Q20/A/R screen at frozen K4, β = 1, with its original rules;
4. fresh confirmation mjhq1k 464–527 only if the original quality and speed gates pass, including the timing-selected Hk.

Otherwise the minimal reproducer is preserved and the track stops. mjhq1k 464–527 stays reserved and unopened.

## Reading rules
- Low reusability of p̄ (corrected K4 held residual ≥ 1): fixing repeat noise alone will not establish useful anchoring.
- Reusable p̄ but not single calls: a stable implementation or better anchor measurement is a plausible next step, not a guaranteed win.
- A deterministic implementation may still have fast-varying dynamic-A4 error.
- No universal method-failure claim.

---

## Amendments and results
(appended below, dated)
- 2026-10-31 ~09:40 UTC, **PARKED before any sr1 measurement** (user decision): SANA unit-gain K4 error reuse is parked on the current model / scheduler
  / A30. No GPU run, no CUDA-13 install, no build and no kernel change took place under sr1 (0.000 A30-h). `scripts/sana_repeat.py` (stages repro /
  fresh / traces) is preserved unrun as the extended reproducer. The amended conclusion is in `results/sana_a30/REPORT.md` (branch sana-a30-feasibility).
  mjhq1k 464–527 remains reserved and unopened.
