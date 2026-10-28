#!/bin/bash
# after the pilot queue: audit re-run (rounding / H_eff fix) + recon from stored codes; regenerate the report; commit + push
cd "$(dirname "$0")/.."
while kill -0 56367 2>/dev/null; do sleep 30; done
L=results/temporal/weights_joint/logs
echo "START audit_rerun"
python -m boa_dit.weights_joint.build --audit_only > $L/audit_rerun.log 2>&1 && echo "DONE audit_rerun" || echo "STOPPED audit_rerun"
grep -E "budget|UNRUN|audit|reproduce" $L/audit_rerun.log | cut -c1-260 | sed 's/^/  /'
python scripts/make_wj.py > /dev/null 2>&1
git add boa_dit/weights_joint results/temporal/weights_joint results/gpu_hours.csv scripts/*wj*.sh scripts/make_wj.py 2>/dev/null
git commit -q -m "weights-joint checkpoint: audit re-run (rounding + H_eff fix), recon from stored codes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "  committed $(git log --oneline -1 | cut -c1-60)"
GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin weights-joint 2>&1 | grep -v "^remote:" | tail -1; echo "  pushed"
echo WJ_FOLLOWUP_DONE
