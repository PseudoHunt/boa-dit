#!/bin/bash
# temporal phase, queue 2 (single GPU): PixArt setup (embeds, design20, bf16 refs, fp16 control), static calib W16, B1
set -e
cd "$(dirname "$0")/.."
L=results/temporal/logs
python -m boa_dit.temporal.pixart setup > $L/pixart_setup.log 2>&1
python -m boa_dit.temporal.pixart calib --weights w16 > $L/calib_w16.log 2>&1
python -m boa_dit.temporal.pixart b1 > $L/b1.log 2>&1
echo Q2_DONE
