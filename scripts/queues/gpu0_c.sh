cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_b.sh" >/dev/null; do sleep 15; done
[ -f results/gate1/quant/ts_dc8_split_w4.json ] || python -m boa_dit.twosided.quantize_ts --split --device cuda:0 --resume
python -m boa_dit.twosided.compare_ckpt ts_dc8_split_w4 deflate_dc_r8_w4
echo GPU0_C_DONE
