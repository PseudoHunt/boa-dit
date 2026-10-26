cd /home/jl_fs/boa-dit
while pgrep -f "boa_dit.objective.proxy --device" >/dev/null; do sleep 15; done
python -u -m boa_dit.objective.d1 --device cuda:0 2>&1 | grep -v -i warn | tail -40
python -u -m boa_dit.objective.d2 --device cuda:0 2>&1 | grep -v -i warn | tail -20
for h in "cfg cur" "cfg cfg"; do set -- $h
  name=$(python -c "import boa_dit.deflate.quantize_dc as q, argparse; print(q.arm_name(argparse.Namespace(mode='heig', r='8', bits=4, damp=0.01, r_config=None, hess='$1', subspace_hess='$2')))")
  [ -f results/deflate/quant/$name.json ] || python -m boa_dit.deflate.quantize_dc --mode heig --r 8 --bits 4 --hess $1 --subspace_hess $2 --device cuda:0 --resume 2>&1 | grep -E "^done|Error|Traceback"
  [ -f results/objective/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage objective --carry --device cuda:0 2>&1 | grep -E "rel_mse_mean|Error" | cut -c1-200
done
python -m boa_dit.twosided.downstream --all_blocks --arms deflate_heig_r8_w4,deflate_heig_r8_w4_cfgsolve,deflate_heig_r8_w4_cfgfull --out results/objective/per_block.json --device cuda:0 2>&1 | grep -E "Error|Traceback"
echo OBJ_QUEUE_DONE
