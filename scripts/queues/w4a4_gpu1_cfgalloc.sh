cd /home/jl_fs/boa-dit
while ps -eo args | grep -v grep | grep -q "w4a4_gpu1_cfgarm.sh"; do sleep 30; done
B=hsvd_r4_b0.5_kv16_u2_rs
python -m boa_dit.w4a4.paired_img ${B}_cfg $B
for spec in "--cfg|hsvd_rw4a4_cfgalloc_b0.5_kv16_u2_rs_cfg" "|hsvd_rw4a4_cfgalloc_b0.5_kv16_u2_rs"; do
  IFS='|' read extra name <<< "$spec"
  bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 --r_config configs/w4a4_cfgalloc.json --kv16 --fc2_unsigned --range_search $extra|$name"
  python -m boa_dit.w4a4.gen --arm $name --n 64 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-160
  python -m boa_dit.w4a4.paired_img $name $B
done
g=$(python -c "from boa_dit.w4a4.paired_img import paired; print(paired('${B}_cfg', '$B')['dpsnr'])")
if python -c "import sys; sys.exit(0 if $g >= 0.2 else 1)"; then
  echo "CFG-amp: RUN (CFG gain $g dB >= 0.2)"
  bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 --kv16 --fc2_unsigned --range_search --cfg --cfg_w 7.5|${B}_cfgamp7.5"
  python -m boa_dit.w4a4.gen --arm ${B}_cfgamp7.5 --n 64 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-160
  python -m boa_dit.w4a4.paired_img ${B}_cfgamp7.5 $B
else
  echo "CFG-amp: SKIP (CFG gain $g dB < 0.2)"
fi
echo W4A4_CFGALLOC_DONE
