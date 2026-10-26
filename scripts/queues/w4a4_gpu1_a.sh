cd /home/jl_fs/boa-dit
bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4|hsvd_r4_b0.5"
python -m boa_dit.w4a4.screen --arm hsvd_r4_b0.5 --tag _rep2 --device cuda:1 2>&1 | grep -E "proxy|Error"
python -m boa_dit.w4a4.screen --arm fp --device cuda:1 2>&1 | grep -E "proxy|Error"
python -m boa_dit.w4a4.gen --arm bf16 --n 256 --device cuda:1 2>&1 | grep -E "arm|Error|Traceback"
python -m boa_dit.w4a4.gen --arm hsvd_r4_b0.5 --n 256 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback"
bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 0|hsvd_r0_b0.5" "--method gptq|gptq_r0" "--method rtn|rtn_r0"
echo W4A4_GPU1_A_DONE
