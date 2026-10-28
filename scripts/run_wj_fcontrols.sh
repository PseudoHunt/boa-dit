#!/bin/bash
# F_all controls (PROTOCOL.md addition, recorded before results), after the follow-up; commit + push
cd "$(dirname "$0")/.."
while kill -0 58368 2>/dev/null; do sleep 30; done
L=results/temporal/weights_joint/logs
run() { log=$1; shift; echo "START $log"; "$@" > $L/$log.log 2>&1; rc=$?; grep -E "budget|UNRUN|latent|Error" $L/$log.log | cut -c1-240 | sed 's/^/  /'
        [ $rc = 0 ] && echo "DONE $log" || echo "STOPPED($rc) $log"; }
run f_controls python -m boa_dit.weights_joint.arms --arms F_all:A,F_all:B,F_all:mean
run f_build_hp python -m boa_dit.weights_joint.build --hp F_all && run f_hp python -m boa_dit.weights_joint.arms --arms F_all:hp
python scripts/make_wj.py > /dev/null 2>&1
git add boa_dit/weights_joint results/temporal/weights_joint results/gpu_hours.csv scripts/*wj*.sh scripts/make_wj.py 2>/dev/null
git commit -q -m "weights-joint checkpoint: F_all controls (static A, B, mean, half-step static)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "  committed $(git log --oneline -1 | cut -c1-60)"
GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin weights-joint 2>&1 | grep -v "^remote:" | tail -1; echo "  pushed"
echo WJ_FCONTROLS_DONE
