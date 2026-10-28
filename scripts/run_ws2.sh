#!/bin/bash
# fixed-code scale modulation (results/temporal/weights_scale/PROTOCOL.md): validate -> refit -> fit (freeze) -> eval;
# commit + push after each step; any failure stops the queue
cd "$(dirname "$0")/.."
L=results/temporal/weights_scale/logs; mkdir -p $L
ck() { git add boa_dit/weights_scale results/temporal/weights_scale results/gpu_hours.csv scripts/run_ws.sh scripts/run_ws2.sh 2>/dev/null
       git commit -q -m "weights-scale checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "  committed $(git log --oneline -1 | cut -c1-50)"
       GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin weights-joint 2>&1 | grep -v "^remote:" | tail -1; echo "  pushed"; }
for step in controls cs eval2; do
  echo "START $step"
  python -m boa_dit.weights_scale.run $step > $L/$step.log 2>&1; rc=$?
  grep -vE "Warn|warn" $L/$step.log | grep -E "budget|UNRUN|bit-exact|fp32 path|pilot|evals|latent|Error|Trace" | cut -c1-300 | sed 's/^/  /'
  ck "$step (rc $rc)"
  [ $rc = 0 ] && echo "DONE $step" || { echo "STOPPED at $step (rc $rc)"; break; }
done
echo WS2_DONE
