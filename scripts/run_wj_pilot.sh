#!/bin/bash
# weights-joint pilot (PROTOCOL.md): waits for the GPU (block 5 on temporal-oracles), then oracle -> audit + constructions ->
# screen arms -> controls for the most promising selective pair. Every job is gated on the 2.0 A30-h pilot cap
# (boa_dit.weights_joint.common.guard; exit 5 = unrun) and followed by a commit + push of branch weights-joint.
cd "$(dirname "$0")/.."
L=results/temporal/weights_joint/logs; mkdir -p $L
while pgrep -f "pixart_b s3 --kind single8 --feedback --compress" > /dev/null; do sleep 30; done
ckpt() { python scripts/make_wj.py > /dev/null 2>&1
         git add boa_dit/weights_joint results/temporal/weights_joint results/gpu_hours.csv scripts/run_wj_pilot.sh scripts/make_wj.py 2>/dev/null
         git commit -q -m "weights-joint checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && echo "  committed $(git log --oneline -1 | cut -c1-60)"
         GIT_TERMINAL_PROMPT=0 timeout 120 git push -q origin weights-joint 2>&1 | grep -v "^remote:" | tail -1; echo "  pushed"; }
job() { log=$1; shift; echo "START $log"; "$@" > $L/$log.log 2>&1; rc=$?
        grep -E "budget|UNRUN|pilot|symmetry|static_|alt_|audit|identical|latent|C/I|recon|Error" $L/$log.log | cut -c1-240 | sed 's/^/  /'
        [ $rc = 0 ] && echo "DONE $log" || echo "STOPPED($rc) $log"; ckpt "$log (rc $rc)"; return $rc; }
job oracle python -m boa_dit.weights_joint.oracle
job build python -m boa_dit.weights_joint.build || { echo "build failed or unrun: no joint arms"; exit 0; }
job screen python -m boa_dit.weights_joint.arms --arms A,indep,B_final1,C_final16,D_top16,E_rand16,F_all
BEST=$(python -c "
import json; a=json.load(open('results/temporal/weights_joint/arms.json'))['arms']; s=[k for k in ('B_final1','C_final16','D_top16','E_rand16') if k in a]
print(min(s, key=lambda k: a[k]['rel_mse_mean']) if s else '')" 2>/dev/null)
echo "most promising selective pair: ${BEST:-none}"
if [ -n "$BEST" ]; then
  job controls python -m boa_dit.weights_joint.arms --arms $BEST:A,$BEST:B,$BEST:mean
  job build_hp python -m boa_dit.weights_joint.build --hp $BEST && job hp python -m boa_dit.weights_joint.arms --arms $BEST:hp
fi
ckpt "pilot finished"
echo WJ_PILOT_DONE
