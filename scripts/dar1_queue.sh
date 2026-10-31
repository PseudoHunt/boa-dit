#!/usr/bin/env bash
# dar1 queue: waits for a given PID (serialized GPU), then runs the listed stages in order, stopping at the first non-zero exit; commit/push each stage.
cd "$(dirname "$0")/.."
WAIT=$1; shift
while kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
for S in "$@"; do
  echo "[dar1 queue] $(date -u +%FT%TZ) start $S"
  python -m boa_dit.anchor_refit.run "$S" > "logs/dar1_${S}.log" 2>&1
  RC=$?
  echo "[dar1 queue] $(date -u +%FT%TZ) end $S rc=$RC"
  git add results/anchor_refit results/w8_spacing results/gpu_hours.csv 2>/dev/null
  git commit -qm "dar1: $S (rc=$RC)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q
  [ $RC -eq 0 ] || exit $RC
done
