#!/bin/bash
# B′ gate (rule.md): calibration + B1′ on both beds; hard budget 0.6 A30-h on top of the 3.54 spent (TEMPORAL_CAP 4.15)
cd "$(dirname "$0")/.."
export TEMPORAL_CAP=4.15
L=results/temporal/logs
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est || { echo "CAP: skipped $log"; return 1; }
        "$@" > $L/$log.log 2>&1 || { echo "FAILED $log"; return 1; }; echo "done $log"; }
job 0.06 bp_calib_dit python -m boa_dit.temporal.dit_act calib &&
job 0.06 bp_b1_dit python -m boa_dit.temporal.dit_act b1 &&
job 0.22 bp_calib_pixart python -m boa_dit.temporal.pixart_b calib &&
job 0.22 bp_b1_pixart python -m boa_dit.temporal.pixart_b b1
echo GATE_DONE
