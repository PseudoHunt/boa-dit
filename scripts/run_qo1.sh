#!/bin/bash
# qo1 queue (results/quantizer_observer/PROTOCOL.md §6); rerun = resume. exit 5 = ceiling.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ck() { git add results/quantizer_observer results/gpu_hours.csv scripts/run_qo1.sh >/dev/null 2>&1; git commit -qm "qo1: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin quantizer-observer-pilot && echo "PUSHED: $1"; }
for s in ${STAGES:-profile trace_fit trace_dev dev_orig headroom accounting errstats observer_select trace_conf observer_confirm cost}; do
  echo "== $s $(date -u +%H:%MZ)"
  python -m boa_dit.quantizer_observer.run $s > logs/qo1_$s.log 2>&1; rc=$?
  grep -v -i warn logs/qo1_$s.log | tail -6; ck "$s (rc $rc)"
  if [ $rc -ne 0 ]; then echo "STAGE $s FAILED/STOPPED (rc $rc)"; break; fi
done
echo QO1_QUEUE_DONE
