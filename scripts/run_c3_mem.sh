#!/bin/bash
# consolidated §5 (rule_distribution.md), after the §3 queue: rank -> eta -> eval; commit + push after each.
cd "$(dirname "$0")/.."
ck() { git add -A results/temporal/mem results/gpu_hours.csv boa_dit scripts >/dev/null 2>&1
       git commit -qm "c3 §5 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin temporal-oracles && echo "PUSHED: $1"; }
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 20; done
for c in rank eta eval; do
  python -m boa_dit.temporal.mem $c > logs/c3_mem_$c.log 2>&1; rc=$?
  grep -v -i -E "warn|loading" logs/c3_mem_$c.log | tail -5
  ck "mem $c (rc $rc)"
  [ $rc = 0 ] || { echo "mem $c FAILED"; break; }
done
echo C3_MEM_DONE
