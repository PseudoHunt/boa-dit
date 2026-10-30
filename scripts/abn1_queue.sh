#!/usr/bin/env bash
# abn1 overnight queue: single sequential GPU queue with a process lock, an 8 h wall-clock watchdog and commit/push after every stage.
# Usage: bash scripts/abn1_queue.sh            (resumable: completed stages are skipped, interrupted generation resumes per batch)
set -u
cd "$(dirname "$0")/.."
CK=ckpt/activation_bridge_overnight
OUT=results/activation_bridge_overnight
mkdir -p "$CK" logs
exec 9>"$CK/queue.lock"
if ! flock -n 9; then echo "[abn1 queue] another queue holds the lock; exiting"; exit 3; fi
[ -f "$CK/launch_utc.txt" ] || python -c "import time; open('$CK/launch_utc.txt','w').write(str(time.time()))"
LAUNCH=$(cat "$CK/launch_utc.txt")
WALL_S=$((8 * 3600)); RESERVE_S=$((9 * 60))

commit() {
  git add "$OUT" results/gpu_hours.csv handoff/SEED_REGISTRY.md 2>/dev/null
  git commit -qm "abn1: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q || echo "[abn1 queue] commit/push skipped or failed for $1"
}

done_stage() {  # a stage is done if its JSON exists with a terminal status (complete/skipped/blocked) or checks.json passed
  python - "$1" <<'PY'
import json, os, sys
s = sys.argv[1]; f = {"checks": "checks.json", "s7": "timing.json"}.get(s, f"{s}.json")
p = os.path.join("results/activation_bridge_overnight", f)
if not os.path.exists(p): sys.exit(1)
r = json.load(open(p))
ok = r.get("pass") if s == "checks" else str(r.get("status", "")).startswith(("complete", "skipped", "blocked"))
sys.exit(0 if ok else 1)
PY
}

for STAGE in checks s1 s2 s3 s4 s5 s6 s7; do
  if done_stage "$STAGE"; then echo "[abn1 queue] $STAGE already done"; continue; fi
  for TRY in 1 2; do
    LEFT=$(python -c "import time; print(int($WALL_S - (time.time() - $LAUNCH) - $RESERVE_S))")
    if [ "$LEFT" -le 60 ]; then echo "[abn1 queue] wall-clock deadline reached before $STAGE"; break 2; fi
    echo "[abn1 queue] $(date -u +%FT%TZ) start $STAGE (try $TRY, wall left ${LEFT}s)"
    timeout --signal=TERM --kill-after=60 "$LEFT" python -m boa_dit.activation_bridge.run "$STAGE" > "logs/abn1_${STAGE}.log" 2>&1
    RC=$?
    echo "[abn1 queue] $(date -u +%FT%TZ) end $STAGE rc=$RC"
    commit "$STAGE (rc=$RC, try $TRY)"
    case $RC in
      0|4|5|6) break ;;                      # done / skipped-blocked / partial (budget) / checks failed: no retry
      124|137) echo "[abn1 queue] watchdog killed $STAGE"; break 2 ;;
      *) echo "[abn1 queue] $STAGE crashed (rc=$RC); one retry (generation resumes from its batch checkpoint)"; tail -20 "logs/abn1_${STAGE}.log" ;;
    esac
  done
done
python scripts/make_activation_bridge.py > /dev/null 2>&1
commit "report and morning status (queue finished)"
echo "[abn1 queue] $(date -u +%FT%TZ) queue finished; lock released on exit"
