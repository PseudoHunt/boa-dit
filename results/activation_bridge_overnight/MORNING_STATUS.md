# MORNING_STATUS (abn1)

Head commit: 660d4fe on branch activation-bridge-overnight (worktree /home/boa-dit-abn).
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
Answers to the six questions: see the end of REPORT.md (written after the queue finished).
