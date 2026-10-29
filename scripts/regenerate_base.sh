#!/bin/bash
# Regenerate the project-generated inputs the DiT pilot depends on (handoff/REGENERATION.md). Nothing trained is downloaded; everything is rebuilt here.
#   1. FID reference statistics       python -m boa_dit.eval_fid --ref          -> results/stage0/ref_stats.npz      (minutes)
#   2. heig-8 W4 base + rank-8 carry  python -m boa_dit.temporal.rebuild       -> ckpt/ckpt_deflate_heig_r8_w4/model.pt, ckpt/temporal/rebuild_quant/.../carry.pt
#                                     (compares the rebuilt carry with the committed results/deflate/quant/deflate_heig_r8_w4/carry.pt, bit for bit)   (≈ 0.1 A30-h)
#   3. static-A8 smoothing + tables   python -m boa_dit.temporal.dit_act calib -> ckpt/temporal/dit_smooth_s.pt, dit_static_b_*.pt   (≈ 0.2 A30-h)
# Each step is skipped if its output exists (rerun = resume). Logs: logs/regen_*.log
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
mkdir -p logs
run() { local n=$1 out=$2; shift 2
  if [ -e "$out" ]; then echo "  $n: exists ($out), skipped"; return 0; fi
  echo "  $n: running $*"; "$@" > logs/regen_$n.log 2>&1; local rc=$?; tail -3 logs/regen_$n.log; [ $rc -eq 0 ] || { echo "REGEN STEP $n FAILED (rc $rc)"; exit $rc; }; }
run ref_stats results/stage0/ref_stats.npz python -m boa_dit.eval_fid --ref
run heig8 ckpt/ckpt_deflate_heig_r8_w4/model.pt python -m boa_dit.temporal.rebuild --device cuda:0
run act_calib ckpt/temporal/dit_static_b_single8.pt python -m boa_dit.temporal.dit_act calib
echo REGEN_OK
