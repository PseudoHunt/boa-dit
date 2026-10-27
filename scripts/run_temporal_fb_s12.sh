#!/bin/bash
# rule_feedback.md stages 1-2 (single GPU, serial), gated on the stage 1-4 ceiling (logged total 7.54 h)
cd "$(dirname "$0")/.."
export TEMPORAL_CAP=7.54
L=results/temporal/logs
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est || { echo "CAP: skipped $log"; return 1; }
        "$@" > $L/$log.log 2>&1 || { echo "FAILED $log"; return 1; }; echo "done $log"; }
job 0.20 fb_stage1 python -m boa_dit.temporal.stage1
job 0.08 bp_calib_dit python -m boa_dit.temporal.dit_act calib &&
job 0.10 bp_b1_dit python -m boa_dit.temporal.dit_act b1
job 0.15 bp_calib_pixart python -m boa_dit.temporal.pixart_b calib &&
job 0.25 bp_b1_pixart python -m boa_dit.temporal.pixart_b b1
echo S12_DONE
