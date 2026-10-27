#!/bin/bash
# temporal phase, queue 1 (single GPU, one job at a time): DiT setup, A0, heig-8 rebuild, 32-seed check
set -e
cd "$(dirname "$0")/.."
L=results/temporal/logs
python -m boa_dit.temporal.dit setup > $L/setup_dit.log 2>&1
python -m boa_dit.temporal.dit a0 > $L/a0.log 2>&1
python -m boa_dit.temporal.rebuild > $L/rebuild.log 2>&1
python -m boa_dit.temporal.dit check32 > $L/check32.log 2>&1
echo Q1_DONE
