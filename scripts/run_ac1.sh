#!/bin/bash
# ac1 launcher (results/analytic_carry/PROTOCOL.md): stages in protocol order, commit + push after each; exit 5 = budget ceiling, 6 = failed checks.
# Rerun = resume (each stage skips itself when complete).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ck() { git add results/analytic_carry results/gpu_hours.csv results/dit_joint/REPORT.md >/dev/null 2>&1; git commit -qm "ac1: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin analytic-carry-pilot && echo "PUSHED: $1"; }
for s in audit checks profile e1_resp e1_select e1_confirm e2_stats e2_solve e2_resp e2_select e2_confirm report; do
  echo "== $s $(date -u +%H:%MZ)"
  python -m boa_dit.analytic_carry.run $s > logs/ac1_$s.log 2>&1; rc=$?
  grep -v -i warn logs/ac1_$s.log | grep -v "^  warnings" | tail -8; ck "$s (rc $rc)"
  if [ $rc -eq 5 ]; then echo "CEILING at $s: remaining stages not run"; python -m boa_dit.analytic_carry.run report > logs/ac1_report.log 2>&1; ck "report after ceiling"; break; fi
  if [ $rc -ne 0 ]; then
    case $s in e1_*) echo "E1 STAGE $s FAILED (rc $rc): rest of E1 skipped, continuing with E2"; SKIP_E1=1;;
      *) echo "STAGE $s FAILED (rc $rc): dependent work stopped"; python -m boa_dit.analytic_carry.run report > logs/ac1_report.log 2>&1; ck "report after failure"; break;; esac
  fi
done
echo AC1_DONE
