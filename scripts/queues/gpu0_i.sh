cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_h.sh" >/dev/null; do sleep 15; done
python -m boa_dit.twosided.downstream --all_blocks --device cuda:0
echo GPU0_I_DONE
