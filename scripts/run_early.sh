#!/bin/bash
# DNS jobs that may share the GPU with E3C's untimed eval/FID phase; holds the lock so E3C's time-matched training waits.
D=/home/boa-dit-dns; cd $D
LOCK=/home/boa-dit/ckpt/gpu_shared.lock
ck() { git add results/dns results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f6 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
step() { local n=$1; shift; "$@" > logs/f6_$n.log 2>&1; local rc=$?; grep -v -i -E "warn|it/s\]" logs/f6_$n.log | tail -8; ck "$n (rc $rc)"; return $rc; }
until [ -f /home/boa-dit-e3c/logs/f5_eval_a.log ]; do sleep 60; done
touch $LOCK; trap "rm -f $LOCK" EXIT
echo "early start $(date -u)"
step checks python -m boa_dit.dns.run checks
step calib python -m boa_dit.dns.run calib && step checks2 python -m boa_dit.dns.run checks
step traces python -m boa_dit.dns.run traces
rm -f $LOCK
echo "EARLY_DONE $(date -u)"
