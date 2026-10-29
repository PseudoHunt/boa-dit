#!/bin/bash
# One-command, resumable launcher for the DiT joint-feedback pilot (results/dit_joint/PROTOCOL.md). Runs ONLY the pilot's stages, in order;
# every stage skips itself when its output JSON exists (resume = rerun this script). It never starts any parked queue (handoff/PARKED_WORK.md).
# Checkpoints stay local (ckpt/dit_joint/); nothing trained is uploaded (user instruction 2026-10-29).
# Usage: scripts/run_dit_joint.sh [stage ...]   (default: all stages)
set -uo pipefail
cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && . .venv/bin/activate
if ! python -c "import boa_dit.dit_joint.run" 2>/dev/null; then
  echo "boa_dit/dit_joint/run.py is not implemented yet: implement it per results/dit_joint/PROTOCOL.md (section D and C5 checks), then rerun."; exit 3
fi
STAGES=${*:-"base reparam_check calib_act checks profile normalizers train_C_L train_E_L train_C_R train_E_R select eval report"}
mkdir -p logs results/dit_joint
ck() { git add results/dit_joint results/gpu_hours.csv handoff >/dev/null 2>&1; git commit -qm "dit_joint: $1

Co-Authored-By: Claude <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
for s in $STAGES; do
  echo "== $s $(date -u +%H:%MZ)"
  python -m boa_dit.dit_joint.run "$s" > "logs/j1_$s.log" 2>&1; rc=$?
  tail -5 "logs/j1_$s.log"; ck "$s (rc $rc)"
  [ $rc -eq 0 ] || { echo "STAGE $s FAILED (rc $rc): fix, record, rerun this script (completed stages are skipped)"; exit $rc; }
done
echo DIT_JOINT_DONE
