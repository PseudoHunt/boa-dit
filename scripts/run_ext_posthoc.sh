#!/bin/bash
# post-hoc (rule_extension.md): E1 feedback with the unsigned-path guard, same 64 prompts; then commit + push
cd "$(dirname "$0")/.."
while kill -0 33305 2>/dev/null; do sleep 30; done
export TEMPORAL_CAP=$(cat results/temporal/ext_ceiling.txt)
python -m boa_dit.temporal.decide budget 0.8 2>/dev/null || { echo "CEILING: skipped ext_e1_feedback_guard"; exit 0; }
echo "START ext_e1_feedback_guard"
python -m boa_dit.temporal.ext_w4a4 arm --mode feedback_guard --set screen > results/temporal/logs/ext_e1_feedback_guard.log 2>&1 && echo "DONE ext_e1_feedback_guard" || echo "FAILED ext_e1_feedback_guard"
grep -E "pilot|psnr" results/temporal/logs/ext_e1_feedback_guard.log | cut -c1-300
scripts/temporal_checkpoint.sh "post-hoc: E1 feedback with unsigned-path guard"
echo POSTHOC_DONE
