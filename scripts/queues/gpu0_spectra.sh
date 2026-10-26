cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_hk.sh" >/dev/null; do sleep 20; done
python -u -m boa_dit.objective.arm_spectra --device cuda:0 2>&1 | grep -E "latent|Error|Traceback"
echo GPU0_SPECTRA_DONE
