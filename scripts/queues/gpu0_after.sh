cd /home/jl_fs/boa-dit
while ps -eo args | grep -v grep | grep -q -E "gpu0_alloc_spectra.sh|gpu0_alloc.sh|objective.layer_attrib"; do sleep 30; done
B=hsvd_r4_b0.5_kv16_u2_rs
# 1. SVDQuant r32 reproduction (deepcompressor), then its bridge eval (screen + MJHQ-256): read 1's protocol offset
[ -d /root/ext/dc_runs/svdq_r32_model ] || bash /root/ext/dc_runs/run_svdq.sh > /root/ext/dc_runs/run_svdq4.log 2>&1
grep -a -E "DC_DONE|^Mon" /root/ext/dc_runs/run_svdq4.log | tail -3
if [ -d /root/ext/dc_runs/svdq_r32_model ]; then
  (cd /root/ext/deepcompressor/examples/diffusion && HF_HOME=/home/jl_fs/hf /root/dcvenv/bin/python /home/jl_fs/boa-dit/boa_dit/w4a4/dc_bridge.py --load /root/ext/dc_runs/svdq_r32_model --device cuda:0) 2>&1 | grep -E "proxy|psnr|Error|Traceback" | cut -c1-200
fi
# 2. corrected base MJHQ-256 (read 1)
python -m boa_dit.w4a4.gen --arm $B --n 256 --device cuda:0 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-200
# 3. K and CFG+K
for spec in "--kfilter configs/w4a4_kfilter.json|${B}_k" "--cfg --kfilter configs/w4a4_kfilter.json|${B}_cfg_k"; do
  IFS='|' read extra name <<< "$spec"
  bash scripts/run_w4a4_q.sh cuda:0 "--method hsvd --r 4 --kv16 --fc2_unsigned --range_search $extra|$name"
  python -m boa_dit.w4a4.gen --arm $name --n 64 --device cuda:0 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-160
  python -m boa_dit.w4a4.paired_img $name $B
done
echo GPU0_AFTER_DONE
