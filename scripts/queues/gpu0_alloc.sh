cd /home/jl_fs/boa-dit
while pgrep -f "objective.layer_attrib --device" >/dev/null; do sleep 30; done
stop=$(python -c "import json; print(json.load(open('results/objective/layer_attrib.json')).get('flat_stop', True))")
if [ "$stop" = "True" ]; then echo "FLAT: top-10 share < 25%, allocation not run (pre-registered stop)"; echo GPU0_ALLOC_DONE; exit 0; fi
python -m boa_dit.objective.alloc_latent 2>&1 | grep -E "^latent|^energy|Error|Traceback"
for lab in latent energy; do
  name=deflate_heig_ralloc${lab}_w4
  [ -f results/deflate/quant/$name.json ] || python -m boa_dit.deflate.quantize_dc --mode heig --r star --r_config configs/objective_alloc_$lab.json --bits 4 --device cuda:0 --resume 2>&1 | grep -E "^done|Error|Traceback"
  [ -f results/objective/eval/latent_$name.json ] || python -m boa_dit.eval_latent_mse --name $name --stage objective --carry --device cuda:0 2>&1 | grep -E "rel_mse_mean|Error" | cut -c1-160
done
echo GPU0_ALLOC_DONE
