#!/bin/bash
# DNS response pilot launcher (results/dns_response/PROTOCOL.md). Waits until the DiT joint pilot has COMPLETELY finished (launcher exited,
# DIT_JOINT_DONE in its queue log, REPORT.md present), then holds a GPU lock and runs the stages in priority order; every stage checks the
# 6 GPU-h / 6 h ceilings itself (exit 5 = ceiling) and is skipped when done (rerun = resume). Commit + push after every stage.
set -uo pipefail
cd "$(dirname "$0")/.."
JQ=/home/boa-dit-joint/logs/j1_queue.log
until grep -q DIT_JOINT_DONE "$JQ" 2>/dev/null && [ -f /home/boa-dit-joint/results/dit_joint/REPORT.md ] && ! pgrep -f "[r]un_dit_joint.sh" >/dev/null; do sleep 60; done
echo "joint pilot complete: $(date -u)"
mkdir -p logs ckpt/dns_response
LOCK=ckpt/dns_response/gpu.lock; touch $LOCK; trap "rm -f $LOCK" EXIT
ck() { git add results/dns_response results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "r1: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
for s in checks components fit select confirm fid_dns fid_L fid_T fid_plain report; do
  echo "== $s $(date -u +%H:%MZ)"
  python -m boa_dit.dns_response.run $s > logs/r1_$s.log 2>&1; rc=$?
  grep -v -i warn logs/r1_$s.log | tail -6; ck "$s (rc $rc)"
  if [ $rc -eq 5 ]; then echo "CEILING reached at $s: remaining stages skipped"; python -m boa_dit.dns_response.run report > logs/r1_report.log 2>&1; ck "report after ceiling"; break; fi
  if [ $rc -ne 0 ] && [ "$s" != "fid_plain" ]; then echo "STAGE $s FAILED (rc $rc): dependent work stopped"; python -m boa_dit.dns_response.run report > logs/r1_report.log 2>&1; ck "report after failure"; break; fi
done
echo R1_DONE
