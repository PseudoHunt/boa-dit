cd /home/jl_fs/boa-dit
while pgrep -f "dc_runs/run_svdq.sh" >/dev/null; do sleep 30; done
if [ -d /root/ext/dc_runs/svdq_r32_model ]; then
  while [ $(ls ckpt/w4a4/img/bf16 2>/dev/null | wc -l) -lt 32 ] || pgrep -f "gen --arm bf16" >/dev/null; do sleep 30; done
  (cd /root/ext/deepcompressor/examples/diffusion && HF_HOME=/home/jl_fs/hf /root/dcvenv/bin/python /home/jl_fs/boa-dit/boa_dit/w4a4/dc_bridge.py --load /root/ext/dc_runs/svdq_r32_model --device cuda:0) 2>&1 | grep -v -i "warn\|pad>\|%|" | tail -20
else
  echo "deepcompressor produced no model: falling back to the SVDQuant reimplementation"
  bash scripts/run_w4a4_q.sh cuda:0 "--method svdq --r 32|svdq_r32"
  python -m boa_dit.w4a4.gen --arm svdq_r32 --n 256 --device cuda:0 2>&1 | grep -E "psnr|Error|Traceback"
fi
bash scripts/run_w4a4_q.sh cuda:0 "--method hsvd --r 4 --subspace heig|hsvd_r4_heig" "--method hsvd --r 4 --beta 0|hsvd_r4_b0" "--method hsvd --r 8|hsvd_r8_b0.5" "--method hsvd --r 8 --subspace heig|hsvd_r8_heig"
echo W4A4_GPU0_B_DONE
