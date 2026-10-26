cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_f.sh" >/dev/null; do sleep 15; done
[ -f ckpt/ckpt_ts_heig8_vv1_fc1v1_dsf_w4/model.pt ] && python -m boa_dit.twosided.downstream --device cuda:0
echo GPU0_G_DONE
