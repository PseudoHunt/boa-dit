#!/bin/bash
# extension block 2 (rule_extension.md, recorded before results); commit + push after every job; ceiling-gated
cd "$(dirname "$0")/.."
L=results/temporal/logs
while kill -0 44415 2>/dev/null; do sleep 30; done
export TEMPORAL_CAP=$(cat results/temporal/ext_ceiling.txt)
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est 2>/dev/null || { echo "CEILING: skipped $log"; return 1; }
        echo "START $log ($(python -m boa_dit.temporal.decide budget 0 2>&1 | sed 's/ + est.*//'))"; "$@" > $L/$log.log 2>&1; rc=$?
        grep -E "pilot|ABORT|latent|R =|psnr" $L/$log.log | cut -c1-260 | sed 's/^/  /'
        [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log"; scripts/temporal_checkpoint.sh "block 2: $log (rc $rc)"; return $rc; }
job 0.20 ext_e2_confirm python -m boa_dit.temporal.ext e2 --seeds confirm --arms none,full,dct4
job 0.25 ext_e5_confirm python -m boa_dit.temporal.ext e5 --seeds confirm --arms attn
for m in base feedback; do job 0.40 ext_e1_w16_$m python -m boa_dit.temporal.ext_w4a4 arm --mode $m --w16 --set screen; done
echo BLOCK2_DONE
