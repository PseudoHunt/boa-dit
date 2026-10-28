#!/bin/bash
# E3 confirmation queue (results/e3c/PROTOCOL.md Part 1); commit + push after every job.
D=/home/boa-dit-e3c; cd $D
ck() { git add results/e3c results/deflate/quant/deflate_heig_r16_w4.json results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
step() {  # name cmd...
  local n=$1; shift
  "$@" > logs/f5_$n.log 2>&1; local rc=$?
  grep -v -i -E "warn|loading|it/s\]" logs/f5_$n.log | tail -6
  ck "$n (rc $rc)"; python -m boa_dit.f5
  return $rc
}
step build_h16 python -m boa_dit.deflate.quantize_dc --mode heig --r 16 --bits 4 --resume || echo "H16 BUILD FAILED"
for a in "C1t 0" "C2 0" "C1u 0" "C1u 1" "C1u 2" "C1t 1" "C1t 2" "C2 1" "C2 2"; do set -- $a
  step train_$1_s$2 python -m boa_dit.e3c.run train --arm $1 --seed $2
done
step mfit python -m boa_dit.e3c.run mfit
step eval_a python -m boa_dit.e3c.run eval
for a in C0 H16 C2_s0 C2M_s0 C1t_s0; do step fid_$a python -m boa_dit.e3c.run fid --arm $a; done
for a in "C2L 0" "C2L 1" "C2L 2" "C1L 0" "C1L 1" "C1L 2"; do set -- $a
  step train_$1_s$2 python -m boa_dit.e3c.run train --arm $1 --seed $2
done
step eval_b python -m boa_dit.e3c.run eval
step fid_C2L_s0 python -m boa_dit.e3c.run fid --arm C2L_s0
echo E3C_DONE
