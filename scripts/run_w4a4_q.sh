#!/usr/bin/env bash
# W4A4 queue (one GPU): items "quantize flags|arm name"; quantize (if missing) then screen (if missing).
cd "$(dirname "$0")/.." || exit 1
dev=$1; shift
for it in "$@"; do
  IFS='|' read flags name <<< "$it"
  [ -f ckpt/w4a4/q_$name.pt ] || python -m boa_dit.w4a4.quantize $flags --device $dev 2>&1 | grep -E "^block|^done|Error|Traceback" || { echo "FAIL quant $name"; continue; }
  [ -f results/w4a4/screen/$name.json ] || python -m boa_dit.w4a4.screen --arm $name --device $dev 2>&1 | grep -E "proxy|Error|Traceback"
done
echo QUEUE_DONE $dev
