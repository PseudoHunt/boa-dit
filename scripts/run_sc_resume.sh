#!/bin/bash
# sampler correction phase (results/temporal/sampler_corr/PROTOCOL.md), after the scale follow-up; commit + push after each step
cd "$(dirname "$0")/.."

L=results/temporal/sampler_corr/logs; mkdir -p $L
ck() { git add boa_dit/sampler_corr results/temporal/sampler_corr results/gpu_hours.csv scripts/run_sc.sh scripts/run_sc_resume.sh 2>/dev/null
       git commit -q -m "sampler-corr checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "  committed $(git log --oneline -1 | cut -c1-50)"
       GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin weights-joint 2>&1 | grep -v "^remote:" | tail -1; echo "  pushed"; }
for step in "calib" "latent" "fid --arm base" "fid --arm qdrift"; do
  tag=$(echo $step | tr ' ' '_' | tr -d '-')
  echo "START $step"
  python -m boa_dit.sampler_corr.run $step > $L/$tag.log 2>&1; rc=$?
  grep -vE "Warn|warn" $L/$tag.log | grep -E "budget|UNRUN|plain|c_i|raw|residual|Q-Drift|FID|Error|Trace" | cut -c1-300 | sed 's/^/  /'
  ck "$step (rc $rc)"
  [ $rc = 0 ] && echo "DONE $step" || { echo "STOPPED at $step (rc $rc)"; break; }
done
echo SC_DONE
