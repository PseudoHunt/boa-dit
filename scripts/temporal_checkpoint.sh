#!/bin/bash
# Crash-safety checkpoint: regenerate reports from the JSONs, write results/temporal/STATUS.md, commit and push.
# Usage: scripts/temporal_checkpoint.sh "<what just finished>"
cd "$(dirname "$0")/.."
python scripts/make_stage3.py > /dev/null 2>&1
python scripts/make_implementation.py > /dev/null 2>&1
python - "$1" <<'PY'
import json, os, sys, subprocess, datetime as dt
R = "results/temporal"
jobs = [  # (label, done-marker file, resume command)
    ("stage3 DiT arm 4a (cyc4a)", f"{R}/eval/dit_bp_single8_cyc4a.json", "python -m boa_dit.temporal.dit_act arm --kind single8 --dither cyc4a"),
    ("FID-2k DiT arm 1 (single table)", f"{R}/stage3/fid2000_dit_bp_single8_none.json", "python -m boa_dit.temporal.fid_act --kind single8 --dither none"),
    ("FID-2k DiT arm 5 (feedback)", f"{R}/stage3/fid2000_dit_bp_single8_feedback.json", "python -m boa_dit.temporal.fid_act --kind single8 --feedback"),
    ("PixArt smoke (2 images, tag smoke)", f"{R}/stage3/pixart_s3_single8_feedback_smoke.json", "python -m boa_dit.temporal.pixart_b s3 --kind single8 --feedback --n 2 --tag smoke"),
    ("PixArt arm: single table (+ bf16 batch-1 refs)", f"{R}/stage3/pixart_s3_single8_none.json", "python -m boa_dit.temporal.pixart_b s3 --kind single8"),
    ("PixArt arm: feedback", f"{R}/stage3/pixart_s3_single8_feedback.json", "python -m boa_dit.temporal.pixart_b s3 --kind single8 --feedback"),
    ("PixArt arm: per-step reference", f"{R}/stage3/pixart_s3_step8_none.json", "python -m boa_dit.temporal.pixart_b s3 --kind step8"),
    ("PixArt arm: best stateless (uniform)", f"{R}/stage3/pixart_s3_single8_unif.json", "python -m boa_dit.temporal.pixart_b s3 --kind single8 --dither unif"),
    ("stage3 DiT arm 4b (cyc4b)", f"{R}/eval/dit_bp_single8_cyc4b.json", "python -m boa_dit.temporal.dit_act arm --kind single8 --dither cyc4b"),
    ("EXT E1 build (W4A4 base)", "ckpt/w4a4/q_hsvd_r4_b0.5_kv16_u2_rs.pt", "python -m boa_dit.temporal.ext_w4a4 build"),
    ("EXT E1 screen: base", f"{R}/ext/e1_w4a4_base_screen.json", "python -m boa_dit.temporal.ext_w4a4 arm --mode base --set screen"),
    ("EXT E1 screen: feedback", f"{R}/ext/e1_w4a4_feedback_screen.json", "python -m boa_dit.temporal.ext_w4a4 arm --mode feedback --set screen"),
    ("EXT E1 screen: uniform", f"{R}/ext/e1_w4a4_unif_screen.json", "python -m boa_dit.temporal.ext_w4a4 arm --mode unif --set screen"),
    ("EXT E2 screen (5 arms)", f"{R}/ext/e2_screen.json", "python -m boa_dit.temporal.ext e2 --seeds screen"),
    ("EXT E3 runtime", f"{R}/ext/e3_runtime.json", "python -m boa_dit.temporal.ext e3 --arms none,unif,full,<best compressed>"),
]
from boa_dit.temporal.decide import spent
now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
L = [f"# Temporal phase: run status (auto-written by scripts/temporal_checkpoint.sh, {now})", "",
     f"Last checkpoint: {sys.argv[1]}. Logged stage-`temporal` compute: {spent():.2f} A30-h. Hard stop lifted (rule_feedback.md).", "",
     "Run from the repo root on a machine set up as in README (weights in ckpt/, PixArt in /home/jl_fs/pixart_sigma_1024). "
     "The calibration tables, smoothing scales and caches live under ckpt/temporal/ (not in git). On a fresh machine, "
     "regenerate them first: `python -m boa_dit.temporal.dit_act calib` (DiT) and `python -m boa_dit.temporal.pixart setup` "
     "plus `python -m boa_dit.temporal.pixart_b calib` then `retable_lomin` (PixArt).", "",
     "| job | status | resume command |", "| --- | --- | --- |"]
for lab, f, cmd in jobs:
    L.append(f"| {lab} | {'**done**' if os.path.exists(f) else 'pending'} | `{cmd}` |")
L += ["", "Partial sampling state (`ckpt/temporal/*.partial`, `ckpt/temporal/pixart/img/<arm>/`) is resumable on the same "
      "machine; after a machine loss, re-run the pending job from scratch.", "",
      "Reports: `results/temporal/stage3/REPORT.md` (stage-3 arms, read 3), `results/temporal/stage3/implementation.md` "
      "(quantizer and feedback implementation), `results/temporal/rule_feedback.md` (pre-registration and dated deviations)."]
open(f"{R}/STATUS.md", "w").write("\n".join(L) + "\n")
PY
git add boa_dit/temporal results/temporal results/gpu_hours.csv scripts/make_*.py scripts/run_temporal_*.sh scripts/run_ext_*.sh scripts/temporal_checkpoint.sh 2>/dev/null
git commit -q -m "temporal checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "  committed: $(git log --oneline -1)" || echo "  nothing new to commit"
GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin temporal-oracles 2>&1 | tail -1 && echo "  pushed to origin/temporal-oracles"
