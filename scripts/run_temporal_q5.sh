#!/bin/bash
# temporal phase, queue 5 (single GPU, after B2): B3 if B2 passes, then exploratory (1) W5/W6, (2) k=3/4, (3) b'.
# Every job is gated on the 4.0 A30-hour cap (python -m boa_dit.temporal.decide budget <estimate>).
cd "$(dirname "$0")/.."
L=results/temporal/logs
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est || { echo "CAP: skipped $log"; return 1; }
        "$@" > $L/$log.log 2>&1 || { echo "FAILED $log"; return 1; }; echo "done $log"; }
read B2 WIN < <(python -m boa_dit.temporal.decide b2)
echo "B2 $B2 $WIN"
if [ "$B2" = PASS ]; then
  W4=hsvd_r0_b0.5_kv16_rs
  job 0.12 b3_calibcache python -m boa_dit.temporal.w4build calib &&
  job 0.55 b3_quant python -m boa_dit.temporal.w4build quant &&
  job 0.18 b3_calib_static python -m boa_dit.temporal.pixart calib --weights $W4 &&
  { [ "$WIN" != calibrated ] || job 0.12 b3_dsolve python -m boa_dit.temporal.pixart dsolve --weights $W4; } &&
  job 0.22 b3_none python -m boa_dit.temporal.pixart arm --weights $W4 --dither none &&
  job 0.22 b3_$WIN python -m boa_dit.temporal.pixart arm --weights $W4 --dither $WIN
fi
# exploratory, in the user's order; stop at the first cap refusal
job 0.10 x_w5_build python -m boa_dit.temporal.rebuild --bits 5 &&
job 0.11 x_w5_eval python -m boa_dit.temporal.dit arm --arm w5 --n 512 &&
job 0.10 x_w6_build python -m boa_dit.temporal.rebuild --bits 6 &&
job 0.11 x_w6_eval python -m boa_dit.temporal.dit arm --arm w6 --n 512 &&
job 0.06 x_pair_extra python -m boa_dit.temporal.pair_extra &&
job 0.11 x_k3 python -m boa_dit.temporal.dit arm --arm k3 --n 512 &&
job 0.11 x_k4 python -m boa_dit.temporal.dit arm --arm k4 --n 512 &&
job 0.11 x_bu python -m boa_dit.temporal.dit arm --arm bu --n 512
echo Q5_DONE
