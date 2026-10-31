#!/usr/bin/env bash
# bar1 queue: waits for a given PID (serialized GPU), then runs the listed steps in order ("module:stage[:arg]", e.g. rep:train:1), stopping at the
# first non-zero exit; commit/push after each step.
cd "$(dirname "$0")/.."
WAIT=$1; shift
while kill -0 "$WAIT" 2>/dev/null; do sleep 10; done
for S in "$@"; do
  IFS=: read -r MOD ST ARG <<< "$S"
  LOG="logs/bar1_${MOD}_${ST}${ARG:+_$ARG}.log"
  echo "[bar1 queue] $(date -u +%FT%TZ) start $S"
  python -m boa_dit.ba_replication.$MOD $ST $ARG > "$LOG" 2>&1
  RC=$?
  echo "[bar1 queue] $(date -u +%FT%TZ) end $S rc=$RC"
  git add boa_dit/ba_replication results/ba_replication results/gpu_hours.csv scripts/bar1_queue.sh 2>/dev/null
  git commit -qm "bar1: $S (rc=$RC)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q
  [ $RC -eq 0 ] || exit $RC
done
