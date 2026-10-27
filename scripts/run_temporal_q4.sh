#!/bin/bash
# temporal phase, queue 4 (single GPU): B2 = calibrated-dither solve (W16) + 4 arms at W16A8-static
set -e
cd "$(dirname "$0")/.."
L=results/temporal/logs
python -m boa_dit.temporal.pixart dsolve --weights w16 > $L/dsolve_w16.log 2>&1
for d in none golden random calibrated; do python -m boa_dit.temporal.pixart arm --weights w16 --dither $d > $L/b2_$d.log 2>&1; done
echo Q4_DONE
