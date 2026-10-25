# Feasibility: query-aware key selection in VAR (Part V) and in a video DiT (Part C) — verdict rules (pre-registration)

Written 2026-10-25, before any generation arm of either part ran. The handoff's §3 is reproduced below; the readings
of its wording that need fixing are stated and fixed here. Results are written by `scripts/make_feas_var.py` and
`scripts/make_feas_video.py` into each part's `SUMMARY.md`; this file does not change after the arms start.

## Part V (VAR-d30, ImageNet 256, cuda:0)
Arms: full / oracle-25 / positional-25 / random-25, at the last three scales (10×10, 13×13, 16×16), all blocks and
heads. 1,000 (label, seed) pairs from `numpy.random.default_rng(0)` (600 if timing forces it; recorded in
`model_facts.md`). Metrics:
- FID-1k against the ADM ImageNet 256 reference statistics (secondary);
- ResNet-50 top-1 label accuracy, with a 95% CI (Wilson interval);
- the fraction of images whose predicted class differs from the full arm's.

**Rules:**
- **Headroom:** acc(oracle-25) ≥ acc(full) − 1.5 points **and** FID-1k(oracle-25) ≤ FID-1k(full) + 1.0.
- **Room for a content-aware selector:** headroom holds, **and** acc(positional-25) ≤ acc(oracle-25) − 3 points with
  the two 95% CIs not overlapping, **and** random-25 is worse still: acc(random-25) < acc(positional-25).
- **No method to build:** |acc(positional-25) − acc(oracle-25)| ≤ 1.5 points, i.e. a fixed rule captures the headroom.

The summary notes regardless that at 256 px VAR's attention and KV cache are small. Any positive result would need
reproducing on Infinity-2B at 1024 px before it says anything about speed.

## Part C (CogVideoX-2B, 480×720, 49 frames, 50 steps, cuda:1)
Arms: full / oracle-10 / window-10 / random-10, on prompt 1, seed 0, all blocks and steps. Text keys are always kept,
and text queries always attend fully. Metrics:
- latent PSNR against the full arm (final latents; peak = the full arm's latent peak-to-peak range, as in the DiT
  harness);
- a frame grid (frames 0, 12, 24, 36, 48);
- CLIP ViT-L/14 image–text score averaged over 8 evenly spaced frames.

**Rules:**
- **Headroom:** PSNR(oracle-10 vs full) ≥ 30 dB **and** no visible degradation in the frame grid. The visual check is
  stated in words in the summary, next to the grid.
- **Room for a content-aware selector:** headroom holds, **and** PSNR(window-10) ≤ PSNR(oracle-10) − 3 dB, **and**
  random-10 is far below both. "Far below" is fixed here as at least 3 dB below window-10.
- **No method to build:** PSNR(window-10) ≥ PSNR(oracle-10) − 1 dB: locality already captures the headroom (Sliding
  Tile Attention's territory).

The step-stability Jaccard (top-10% sets, steps 10 vs 11) from C1 is reported alongside; high stability is the
caching headroom that TeaCache and PAB already exploit.

## Common
Each part has its own GPU end to end, and nothing moves between cuda:0 and cuda:1. The caps are 1.5 A30-hours
(Part V) and 2.5 (Part C), logged in `results/gpu_hours.csv` under stages `feas_var` and `feas_video`. Each part
writes its own `SUMMARY.md`, and REPORT.md gets a dated entry as each part finishes.
