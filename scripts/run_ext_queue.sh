#!/bin/bash
# extension queue (rule_extension.md): waits for queue B, then for the user's ceiling (results/temporal/ext_ceiling.txt,
# an absolute logged total in A30-h); every job is gated on it and followed by a commit + push.
cd "$(dirname "$0")/.."
L=results/temporal/logs
while kill -0 32041 2>/dev/null; do sleep 30; done
CF=results/temporal/ext_ceiling.txt
[ -f $CF ] || echo "WAITING for the extension ceiling (results/temporal/ext_ceiling.txt)"
while [ ! -f $CF ]; do sleep 30; done
export TEMPORAL_CAP=$(cat $CF)
echo "extension ceiling: logged total $TEMPORAL_CAP A30-h"
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est 2>/dev/null || { echo "CEILING: skipped $log"; return 1; }
        echo "START $log ($(python -m boa_dit.temporal.decide budget 0 2>&1 | sed 's/ + est.*//'))"; "$@" > $L/$log.log 2>&1; rc=$?
        tail -n 12 $L/$log.log | grep -E "pilot|ABORT|latent|R =|e2e|psnr" | sed 's/^/  /'
        [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log (see $L/$log.log)"; scripts/temporal_checkpoint.sh "extension: $log (rc $rc)"; return $rc; }
job 0.05 ext_e1_build python -m boa_dit.temporal.ext_w4a4 build &&
for m in base feedback unif; do job 0.05 ext_e1_$m python -m boa_dit.temporal.ext_w4a4 arm --mode $m --set screen; done
job 0.05 ext_e2_screen python -m boa_dit.temporal.ext e2 --seeds screen
BEST=$(python -c "
import json; r=json.load(open('results/temporal/ext/e2_screen.json'))['arms']; c={k:v['rel_mse_mean'] for k,v in r.items() if k in ('mean','dct4','rand4')}
print(min(c,key=c.get) if c else '')" 2>/dev/null)
echo "best compressed arm on the E2 screen: ${BEST:-none}"
job 0.05 ext_e3_runtime python -m boa_dit.temporal.ext e3 --arms none,unif,full${BEST:+,$BEST}
echo "EXT_QUEUE_DONE (next: user picks a confirmation run or the E4/E5 diagnostic)"
