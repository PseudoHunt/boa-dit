cd /home/jl_fs/boa-dit
while pgrep -f "fid5k --arm deflate_dc_r8_w4" >/dev/null; do sleep 15; done
python -m boa_dit.twosided.fid5k --arm deflate_heig_r8_w4 --device cuda:1
echo GPU1_C_DONE
