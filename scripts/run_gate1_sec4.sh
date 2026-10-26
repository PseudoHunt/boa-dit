#!/usr/bin/env bash
# Gate 1 Section 4 queue (one GPU, end to end). Args: device, then items = quantize_ts flag strings, e.g. "--act_order".
cd "$(dirname "$0")/.." || exit 1
dev=$1; shift
for it in "$@"; do
  name=$(python -m boa_dit.twosided.quantize_ts $it --print_name)
  [ -f results/gate1/quant/$name.json ] || python -m boa_dit.twosided.quantize_ts $it --device $dev --resume || { echo "FAIL quant $name"; continue; }
  [ -f results/gate1/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage gate1 --carry --device $dev || echo "FAIL latent $name"
done
echo QUEUE_DONE $dev
