#!/bin/bash
# temporal phase, queue 3 (single GPU): A1 pair, A2 arms (a diag only; b c e on 512; d f on 128)
set -e
cd "$(dirname "$0")/.."
L=results/temporal/logs
python -m boa_dit.temporal.pair > $L/pair.log 2>&1
python -m boa_dit.temporal.dit diag_only --arm a > $L/diag_a.log 2>&1
for a in b c e; do python -m boa_dit.temporal.dit arm --arm $a --n 512 > $L/arm_$a.log 2>&1; done
for a in d f; do python -m boa_dit.temporal.dit arm --arm $a --n 128 > $L/arm_$a.log 2>&1; done
echo Q3_DONE
