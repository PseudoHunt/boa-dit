# Park status, 2026-10-29 (user instruction: park all queues; the DiT joint-feedback pilot is the next active work)

All project queues and launchers were stopped at ≈ 13:30 UTC (the diag log's last write is 13:30:11; corrected 2026-10-29 13:56, an earlier draft said 13:58), in this order: launchers first (`run_px_v2.sh`, `run_px_v3.sh`), then the active child,
then the monitors and autosave loops. Nothing can restart them: no cron, no live chain scripts.

| job | status | what exists | to resume |
|---|---|---|---|
| E3 confirmation, DiT Part 1 | **COMPLETE** | REPORT.md, eval.json, fid/*, fid_boot.json, carries (archive) | — |
| PixArt Part 2 design (lr grid + guard) | **COMPLETE**; guard failed for all arms; nothing promoted | pixart/design.json, design_v1.json | — |
| PixArt diagnostics v1 (norm, branch, interv, alpha) | **COMPLETE** | pixart/diag.json keys norm, branch, interv, alpha | — |
| PixArt diagnostics v2: `steps`, `branch2` | **COMPLETE** | diag.json keys steps, branch2; per-prompt arrays; final latents `ckpt/e3c/pixart/diag_endpoints_*.pt` (archive) | — |
| PixArt diagnostics v2: `interv2` (q / q+o transplants) | **INTERRUPTED / PARTIAL** (stopped mid-sweep; results are kept in memory, no live state to serialize) | P0 and P1_lr1e-5 rows only in `logs/f5_px_diag_v2.log.txt`; P1_lr3e-5 partway (its grids exist); image grids `grid_P0`, `grid_P1_lr1e-5*`, `grid_P1_lr3e-5*` | `python -m boa_dit.e3c.px_diag interv2` (full rerun, ≈ 1.7 h on A30) |
| PixArt lr 1e-4 matched screen | **NOT STARTED** | — | `python -m boa_dit.e3c.px_screen` |
| PixArt paired α + reproducibility | **NOT STARTED** | — | `python -m boa_dit.e3c.px_diag alpha2` |
| PixArt lr 1e-5 matched screen | **NOT STARTED** | — | `python -m boa_dit.e3c.px_screen --lr 1e-5 --tag lr1e-5 --cks 10,25,50,100,200 --no-drift` |
| PixArt fresh-dev validation (mjhq1k[400:432]) | **NOT STARTED**; set unopened | — | `python -m boa_dit.e3c.px_fresh` (after the lr 1e-5 screen) |
| Trajectory-anchored target | **REGISTERED, CONDITIONAL, UNIMPLEMENTED** | PROTOCOL.md 2026-10-29 15:10 / 15:40 | — |
| PixArt Part 3 pilot (user spec) | **HELD**, never run (its one check run was aborted and discarded) | PROTOCOL.md Part 3 | `scripts/run_pilot.sh`, only after the Part 2 repair decision |

Interrupted and failed GPU jobs are all in `results/gpu_hours.csv` (stage f5). The stopped interv2 run logs no `Run` record (killed before exit),
so its GPU time is **unlogged** in the CSV. Estimate: the v2 process ran ≈ 11:06–13:30 UTC ≈ 2.4 h (steps, branch2, partial interv2). The stopped α-run v1 tail is logged.
