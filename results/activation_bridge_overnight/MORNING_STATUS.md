# MORNING_STATUS (abn1)

Head commit: 8a5a870 on branch activation-bridge-overnight (worktree /home/boa-dit-abn).
GPU: 3.407 of 6.0 A30-h. Running at shutdown: none (the queue exits after the last stage and releases its lock).

| stage | status |
|---|---|
| Correctness gate | pass |
| Stage 1 — FP anchors K4, 256 confirmation images | complete |
| Stage 2 — W8 anchors K4, first 128 | complete |
| Stage 3 — K8 (FP and W8), first 128 | complete |
| Stage 4 — existing local refit + bridge, first 128 | complete |
| Stage 5 — diagnostics (8 dev images) | complete |
| Stage 6 — FID-2k pilot | skipped (full three-arm FID-2k does not fit the remaining budget/wall) |
| Stage 7 — timing | complete |

Artifacts: `results/activation_bridge_overnight/` (JSON, REPORT.md, per-image metrics inside each stage JSON); latents and per-batch checkpoints in
`ckpt/activation_bridge_overnight/` (local only, not uploaded — standing instruction: models/artifacts are regenerated from code; hashes in `artifacts.json`).
Answers to the six questions: end of REPORT.md / ANSWERS.md. In short: (1) feedback harms the bridge (ACB/CB 1.690 [1.335, 2.250]); (2) the harm
survives the matched-range control (ACB/GB 1.677); (3) bridge works with W8 anchors (CB.w8/C.w8 0.279); (4) K8 vs ordinary K4 unresolved; (5) local
refit + bridge inconclusive (LCB/CB 0.715 [0.529, 1.047]); (6) FID not measured (did not fit); efficiency only simulated, no anchored arm beats FP50.
Skipped: Stage 6 FID-2k (pre-registered budget rule; cohort 112000–113999 never generated). Blocked: none. Failed: none.
Queue lock released; no process running; GPU idle. Reporting bug noted in PROTOCOL (s3.json call-count field), no data effect.
