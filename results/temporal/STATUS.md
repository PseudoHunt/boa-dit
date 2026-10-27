# Temporal phase: run status (auto-written by scripts/temporal_checkpoint.sh, 2026-10-27 22:17 UTC)

Last checkpoint: extension: ext_e1_unif (rc 0). Logged stage-`temporal` compute: 14.70 A30-h. Hard stop lifted (rule_feedback.md).

Run from the repo root on a machine set up as in README (weights in ckpt/, PixArt in /home/jl_fs/pixart_sigma_1024). The calibration tables, smoothing scales and caches live under ckpt/temporal/ (not in git). On a fresh machine, regenerate them first: `python -m boa_dit.temporal.dit_act calib` (DiT) and `python -m boa_dit.temporal.pixart setup` plus `python -m boa_dit.temporal.pixart_b calib` then `retable_lomin` (PixArt).

| job | status | resume command |
| --- | --- | --- |
| stage3 DiT arm 4a (cyc4a) | **done** | `python -m boa_dit.temporal.dit_act arm --kind single8 --dither cyc4a` |
| FID-2k DiT arm 1 (single table) | **done** | `python -m boa_dit.temporal.fid_act --kind single8 --dither none` |
| FID-2k DiT arm 5 (feedback) | **done** | `python -m boa_dit.temporal.fid_act --kind single8 --feedback` |
| PixArt smoke (2 images, tag smoke) | **done** | `python -m boa_dit.temporal.pixart_b s3 --kind single8 --feedback --n 2 --tag smoke` |
| PixArt arm: single table (+ bf16 batch-1 refs) | **done** | `python -m boa_dit.temporal.pixart_b s3 --kind single8` |
| PixArt arm: feedback | **done** | `python -m boa_dit.temporal.pixart_b s3 --kind single8 --feedback` |
| PixArt arm: per-step reference | **done** | `python -m boa_dit.temporal.pixart_b s3 --kind step8` |
| PixArt arm: best stateless (uniform) | **done** | `python -m boa_dit.temporal.pixart_b s3 --kind single8 --dither unif` |
| stage3 DiT arm 4b (cyc4b) | **done** | `python -m boa_dit.temporal.dit_act arm --kind single8 --dither cyc4b` |
| EXT E1 build (W4A4 base) | **done** | `python -m boa_dit.temporal.ext_w4a4 build` |
| EXT E1 screen: base | **done** | `python -m boa_dit.temporal.ext_w4a4 arm --mode base --set screen` |
| EXT E1 screen: feedback | **done** | `python -m boa_dit.temporal.ext_w4a4 arm --mode feedback --set screen` |
| EXT E1 screen: uniform | **done** | `python -m boa_dit.temporal.ext_w4a4 arm --mode unif --set screen` |
| EXT E2 screen (5 arms) | pending | `python -m boa_dit.temporal.ext e2 --seeds screen` |
| EXT E3 runtime | pending | `python -m boa_dit.temporal.ext e3 --arms none,unif,full,<best compressed>` |

Partial sampling state (`ckpt/temporal/*.partial`, `ckpt/temporal/pixart/img/<arm>/`) is resumable on the same machine; after a machine loss, re-run the pending job from scratch.

Reports: `results/temporal/stage3/REPORT.md` (stage-3 arms, read 3), `results/temporal/stage3/implementation.md` (quantizer and feedback implementation), `results/temporal/rule_feedback.md` (pre-registration and dated deviations).
