#!/bin/bash
# user swap (rule_distribution.md): PixArt DCT-16 feedback on both prompt sets -> score -> §5 with what is left; push after each job.
cd "$(dirname "$0")/.."
ck() { git add results/temporal/dist results/temporal/mem results/temporal/rule_distribution.md results/gpu_hours.csv boa_dit/temporal scripts >/dev/null 2>&1
       git commit -qm "c3 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin temporal-oracles && echo "PUSHED: $1"; }
python -m boa_dit.temporal.dist gen16 > logs/c3_dct16_gen.log 2>&1; rc=$?
grep -E "pilot|budget|done|Error" logs/c3_dct16_gen.log
ck "DCT-16 feedback images, both sets (rc $rc)"
[ $rc = 0 ] || { echo "gen16 FAILED"; exit 1; }
python -m boa_dit.temporal.dist score > logs/c3_score16.log 2>&1 && grep -v -i warn logs/c3_score16.log || { echo "score FAILED"; exit 1; }
ck "DCT-16 trade-off read (PSNR/LPIPS/latent, both sets)"
echo DCT16_DONE
for c in rank eta eval; do
  python -m boa_dit.temporal.mem $c > logs/c3_mem_$c.log 2>&1; rc=$?
  grep -v -i -E "warn|loading" logs/c3_mem_$c.log | tail -5
  ck "§5 mem $c (rc $rc)"
  [ $rc = 0 ] || { echo "mem $c FAILED"; break; }
done
echo C3_ALL_DONE
