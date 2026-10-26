cd /home/jl_fs/boa-dit
B=hsvd_r4_b0.5_kv16_u2_rs
# 1. let the old chain finish the CFG arm's MJHQ-64 screen, then stop it before its (superseded) proportional CFG-alloc steps
until [ -f results/w4a4/eval/img64_${B}_cfg.json ]; do sleep 20; done
sleep 60
for p in $(pgrep -f "logs/w4a4_gpu1_cfgchain.sh"); do pkill -P $p; kill $p; done
python -m boa_dit.w4a4.paired_img ${B}_cfg $B
# 2. corrected greedy CFG-alloc ranks (H_cfg and per-branch control), then the two arms
python -m boa_dit.w4a4.cfg_alloc --device cuda:1 > logs/w4a4_cfg_alloc_greedy.log 2>&1 || exit 1
grep -v "^ " logs/w4a4_cfg_alloc_greedy.log | tail -3
for spec in "--r_config configs/w4a4_cfgalloc_greedy.json --cfg|hsvd_rw4a4_cfgalloc_greedy_b0.5_kv16_u2_rs_cfg" "--r_config configs/w4a4_cfgalloc_greedy_ctl.json|hsvd_rw4a4_cfgalloc_greedy_ctl_b0.5_kv16_u2_rs"; do
  IFS='|' read extra name <<< "$spec"
  bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 $extra --kv16 --fc2_unsigned --range_search|$name"
  python -m boa_dit.w4a4.gen --arm $name --n 64 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-160
  python -m boa_dit.w4a4.paired_img $name $B
done
# 3. CFG-amp, gated on the CFG arm (unchanged rule)
g=$(python -c "from boa_dit.w4a4.paired_img import paired; print(paired('${B}_cfg', '$B')['dpsnr'])")
if python -c "import sys; sys.exit(0 if $g >= 0.2 else 1)"; then
  echo "CFG-amp: RUN (CFG gain $g dB >= 0.2)"
  bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 --kv16 --fc2_unsigned --range_search --cfg --cfg_w 7.5|${B}_cfgamp7.5"
  python -m boa_dit.w4a4.gen --arm ${B}_cfgamp7.5 --n 64 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-160
  python -m boa_dit.w4a4.paired_img ${B}_cfgamp7.5 $B
else
  echo "CFG-amp: SKIP (CFG gain $g dB < 0.2)"
fi
echo W4A4_GREEDY_DONE
