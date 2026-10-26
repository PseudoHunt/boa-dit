cd /home/jl_fs/boa-dit
while pgrep -f "logs/w4a4_gpu1_a.sh" >/dev/null; do sleep 30; done
python -m boa_dit.w4a4.rho --arm hsvd_r4_b0.5 --device cuda:1 2>&1 | grep -E "gate|block|Error|Traceback"
python -m boa_dit.w4a4.alloc
bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 --r_config configs/w4a4_alloc.json|hsvd_rw4a4_alloc_b0.5" "--method hsvd --r 4 --order err --target asym|hsvdB4_r4_b0.5"
echo W4A4_GPU1_B_DONE
