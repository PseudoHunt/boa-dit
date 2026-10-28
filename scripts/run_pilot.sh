#!/bin/bash
# Part 3 PixArt pilot (user spec), run last; commit + push after every job.
D=/home/boa-dit-e3c; cd $D
QB=/home/boa-dit/ckpt/w4a4/q_hsvd_r4_b0.5_kv16_u2_rs.pt; PQ=/home/boa-dit/ckpt/e3c/pixart_pilot
ck() { git add results/e3c results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 pilot checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
step() { local n=$1; shift; "$@" > logs/f5_$n.log 2>&1; local rc=$?; grep -v -i -E "warn|it/s\]" logs/f5_$n.log | tail -8; ck "$n (rc $rc)"; python -m boa_dit.f5; return $rc; }
P="python -m boa_dit.e3c.pixart_refit"
step pilot_check $P check --pilot || { echo "PILOT STEP-1 CHECK FAILED: pilot uninformative (reported as such)"; exit 6; }
step pilot_cache $P cache --pilot && step pilot_profile $P pilot_profile || { echo "PILOT STOPPED"; exit 7; }
python -c "import json;import sys;sys.exit(0 if json.load(open('results/e3c/pixart_pilot/profile.json'))['K']==3 else 1)" || step pilot_cache_K2 $P cache --pilot
step pilot_final $P pilot_final
step pilot_gen_ref $P gen --name pilot_bf16_256 --q bf16 --lo 256 --hi 288
step pilot_gen_base $P gen --name pilot_base_256 --q $QB --lo 256 --hi 288
step pilot_gen_C1 $P gen --name pilot_C1_256 --q $PQ/q_pilot_P1.pt --lo 256 --hi 288
step pilot_gen_C2 $P gen --name pilot_C2_256 --q $PQ/q_pilot_P2.pt --lo 256 --hi 288
step pilot_score $P pilot_score
echo PILOT_DONE
