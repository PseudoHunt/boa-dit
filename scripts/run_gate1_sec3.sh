#!/usr/bin/env bash
# Gate 1 Section 3 queue (one GPU, end to end). Args: device, then items "mode:r:bits".
cd "$(dirname "$0")/.." || exit 1
dev=$1; shift
for it in "$@"; do
  IFS=: read mode r bits <<< "$it"
  name="deflate_${mode}_r${r}_w${bits}"
  [ -f results/deflate/quant/$name.json ] || python -m boa_dit.deflate.quantize_dc --mode $mode --r $r --bits $bits --device $dev --resume || { echo "FAIL quant $name"; continue; }
  while [ ! -f ckpt/fp_eval_latents.pt ]; do sleep 20; done
  [ -f results/gate1/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage gate1 --carry --device $dev || echo "FAIL latent $name"
done
echo QUEUE_DONE $dev
