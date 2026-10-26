cd /home/jl_fs/boa-dit
while pgrep -f "logs/obj_queue.sh" >/dev/null; do sleep 15; done
S=$(python -c "import json; print(json.load(open('configs/objective_hk.json'))['sigma_tokens'])")
name=deflate_heig_r8_w4_hk
[ -f results/deflate/quant/$name.json ] || python -m boa_dit.deflate.quantize_dc --mode heig --r 8 --bits 4 --blur_sigma $S --device cuda:0 --resume 2>&1 | grep -E "^done|Error|Traceback"
[ -f results/objective/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage objective --carry --device cuda:0 2>&1 | grep -E "rel_mse_mean|Error" | cut -c1-200
python -m boa_dit.twosided.downstream --all_blocks --arms $name --out results/objective/per_block_hk.json --device cuda:0 2>&1 | grep -E "Error|Traceback"
python -u -m boa_dit.objective.proxy --device cuda:0 --extra "CFG-solve=deflate_heig_r8_w4_cfgsolve,CFG-full=deflate_heig_r8_w4_cfgfull,H_K=$name" 2>&1 | grep -E "proxy|Spearman|Error|Traceback"
echo OBJ_QUEUE2_DONE
