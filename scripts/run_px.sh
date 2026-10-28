#!/bin/bash
# PixArt Part 2 (results/e3c/PROTOCOL.md); commit + push after every job.
D=/home/boa-dit-e3c; cd $D
QB=/home/boa-dit/ckpt/w4a4/q_hsvd_r4_b0.5_kv16_u2_rs.pt; PQ=/home/boa-dit/ckpt/e3c/pixart
ck() { git add results/e3c results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
step() { local n=$1; shift; "$@" > logs/f5_$n.log 2>&1; local rc=$?; grep -v -i -E "warn|it/s\]" logs/f5_$n.log | tail -8; ck "$n (rc $rc)"; python -m boa_dit.f5; return $rc; }
P="python -m boa_dit.e3c.pixart_refit"
step px_check $P check || { echo "PX CHECK FAILED: Part 2 stops (reported as such)"; exit 6; }
step px_cache $P cache && step px_profile $P profile && step px_plan $P plan && step px_design $P design && step px_final $P final || { echo "PX TRAINING STOPPED"; exit 7; }
step px_gen_ref_c288 $P gen --name p5_bf16_c288 --q bf16 --lo 288 --hi 352
step px_gen_P0_c288 $P gen --name p5_P0_c288 --q $QB --lo 288 --hi 352
for a in P1 P2; do
  step px_gen_${a}_mjhq64 $P gen --name p5_${a}_mjhq64 --q $PQ/q_final_$a.pt --lo 0 --hi 64
  step px_gen_${a}_c288 $P gen --name p5_${a}_c288 --q $PQ/q_final_$a.pt --lo 288 --hi 352
done
step px_score $P score
step px_ir /tmp/claude-0/-home/9a42aff5-4872-4f32-ba52-5b677aa2b906/scratchpad/irenv/bin/python scripts/px_ir.py
echo PX_DONE
