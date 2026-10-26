cd /home/jl_fs/boa-dit
true


bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 --kv16 --fc2_unsigned --range_search --cfg|hsvd_r4_b0.5_kv16_u2_rs_cfg"
python -m boa_dit.w4a4.gen --arm hsvd_r4_b0.5_kv16_u2_rs_cfg --n 64 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-200
echo W4A4_CFGARM_DONE
