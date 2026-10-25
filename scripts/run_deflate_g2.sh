#!/usr/bin/env bash
# One G2 queue per GPU. Args: device, then items "mode:r:bits:fid(0/1)".
dev=$1; shift
for it in "$@"; do
  IFS=: read mode r bits fid <<< "$it"
  name="deflate_${mode}_r${r}_w${bits}"
  [ -f results/deflate/quant/$name.json ] || python -m boa_dit.deflate.quantize_dc --mode $mode --r $r --bits $bits --device $dev --resume || { echo "FAIL quant $name"; continue; }
  [ -f results/deflate/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage deflate --carry --device $dev || echo "FAIL latent $name"
  if [ "$fid" = 1 ] && [ ! -f results/deflate/eval/fid2000_$name.json ]; then
    mkdir -p ckpt/deflate_grids
    python -m boa_dit.eval_fid --name $name --n 2000 --stage deflate --carry --grid ckpt/deflate_grids/$name.pt --device $dev || echo "FAIL fid $name"
  fi
done
echo QUEUE_DONE $dev
