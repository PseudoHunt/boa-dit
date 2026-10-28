#!/bin/bash
# block 3: rand4 on the confirmation seeds (rule_extension.md); commit + push
cd "$(dirname "$0")/.."
while kill -0 47582 2>/dev/null; do sleep 30; done
export TEMPORAL_CAP=$(cat results/temporal/ext_ceiling.txt)
python -m boa_dit.temporal.decide budget 0.36 2>/dev/null || { echo "CEILING: skipped ext_e2_confirm_rand4"; scripts/temporal_checkpoint.sh "block 3 skipped (ceiling)"; exit 0; }
echo "START ext_e2_confirm_rand4"
python -m boa_dit.temporal.ext e2 --seeds confirm --arms rand4 > results/temporal/logs/ext_e2_confirm_rand4.log 2>&1 && echo "DONE ext_e2_confirm_rand4" || echo "FAILED ext_e2_confirm_rand4"
grep -E "rand4|R =" results/temporal/logs/ext_e2_confirm_rand4.log | sed 's/^/  /'
scripts/temporal_checkpoint.sh "block 3: rand4 confirmation"
echo BLOCK3_DONE
