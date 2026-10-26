cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_alloc.sh" >/dev/null; do sleep 30; done
[ -f results/objective/eval/latent_deflate_heig_ralloclatent_w4.json ] && python -u -m boa_dit.objective.arm_spectra --device cuda:0 2>&1 | grep -E "latent|Error|Traceback"
echo GPU0_ALLOC_SPECTRA_DONE
