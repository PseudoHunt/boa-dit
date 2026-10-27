#!/bin/bash
# temporal phase, queue 6 (exploratory, last): period-2 dither at W16A8-static, gated on the 4.0 h cap; then stop
cd "$(dirname "$0")/.."
python -m boa_dit.temporal.decide budget 0.29 || { echo "CAP: skipped b2x_period2"; exit 0; }
python -m boa_dit.temporal.pixart arm --weights w16 --dither period2 > results/temporal/logs/x_period2.log 2>&1 && echo "done x_period2" || echo "FAILED x_period2"
echo Q6_DONE
