#!/usr/bin/env bash
# cfb1 (pilot J) queue: single sequential GPU queue with a process lock, one retry after a crash, commit/push after every stage.
# Usage: bash scripts/cfb1_queue.sh            (resumable: completed stages are skipped, interrupted generation resumes per batch)
set -u
cd "$(dirname "$0")/.."
CK=ckpt/coupled_feedback_bridge
OUT=results/coupled_feedback_bridge
mkdir -p "$CK" logs
exec 9>"ckpt/feedback_bridge_gpu.lock"
if ! flock -n 9; then echo "[cfb1 queue] another queue holds the GPU lock; exiting"; exit 3; fi

commit() {
  git add "$OUT" results/gpu_hours.csv handoff/SEED_REGISTRY.md 2>/dev/null
  git commit -qm "cfb1: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin HEAD || echo "[cfb1 queue] commit/push skipped or failed for $1"
}

done_stage() {
  python - "$1" <<'PY'
import json, os, sys
s = sys.argv[1]; p = os.path.join("results/coupled_feedback_bridge", f"{s}.json")
if not os.path.exists(p): sys.exit(1)
r = json.load(open(p))
ok = r.get("pass") if s == "checks" else str(r.get("status", "")).startswith(("complete", "skipped", "blocked"))
sys.exit(0 if ok else 1)
PY
}

for STAGE in checks fit val conf diag; do
  if done_stage "$STAGE"; then echo "[cfb1 queue] $STAGE already done"; continue; fi
  for TRY in 1 2; do
    echo "[cfb1 queue] $(date -u +%FT%TZ) start $STAGE (try $TRY)"
    python -m boa_dit.feedback_bridge.cfb "$STAGE" > "logs/cfb1_${STAGE}.log" 2>&1
    RC=$?
    echo "[cfb1 queue] $(date -u +%FT%TZ) end $STAGE rc=$RC"
    commit "$STAGE (rc=$RC, try $TRY)"
    case $RC in
      0|4|5) break ;;
      6) echo "[cfb1 queue] checks failed: quality stages blocked"; break 2 ;;
      *) echo "[cfb1 queue] $STAGE crashed (rc=$RC); one retry"; tail -20 "logs/cfb1_${STAGE}.log" ;;
    esac
  done
done
python scripts/make_coupled_feedback.py > /dev/null 2>&1
commit "report and status (queue finished)"
echo "[cfb1 queue] $(date -u +%FT%TZ) queue finished; lock released on exit"
