#!/bin/bash
# block 4 (rule_extension.md, recorded before results): dct4 at attention sites (512 seeds), E4 step count; commit + push
cd "$(dirname "$0")/.."
L=results/temporal/logs
while kill -0 50023 2>/dev/null; do sleep 30; done
export TEMPORAL_CAP=$(cat results/temporal/ext_ceiling.txt)
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est 2>/dev/null || { echo "CEILING: skipped $log"; return 1; }
        echo "START $log ($(python -m boa_dit.temporal.decide budget 0 2>&1 | sed 's/ + est.*//'))"; "$@" > $L/$log.log 2>&1; rc=$?
        grep -E "pilot|ABORT|latent|steps|R =" $L/$log.log | cut -c1-260 | sed 's/^/  /'
        [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log"; scripts/temporal_checkpoint.sh "block 4: $log (rc $rc)"; return $rc; }
job 0.30 ext_e6_attn_dct4 python -m boa_dit.temporal.ext e6 --seeds confirm --arms attn_dct4
job 0.80 ext_e4_stepcount python -m boa_dit.temporal.ext_e4
echo BLOCK4_DONE
