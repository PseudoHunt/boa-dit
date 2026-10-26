cd /home/jl_fs/boa-dit
for v in pooled bucket; do
  name=deflate_heig_r8_w4_hk$v
  [ -f results/deflate/quant/$name.json ] || python -m boa_dit.deflate.quantize_dc --mode heig --r 8 --bits 4 --spectral configs/objective_hk_$v.json --device cuda:0 --resume 2>&1 | grep -E "^done|Error|Traceback"
  [ -f results/objective/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage objective --carry --device cuda:0 2>&1 | grep -E "rel_mse_mean|Error" | cut -c1-160
done
python -m boa_dit.twosided.downstream --all_blocks --arms deflate_heig_r8_w4_hkpooled,deflate_heig_r8_w4_hkbucket --out results/objective/per_block_hk2.json --device cuda:0 2>&1 | grep -E "Error|Traceback"
echo GPU0_HK_DONE
