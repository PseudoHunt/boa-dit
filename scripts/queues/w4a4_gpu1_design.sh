cd /home/jl_fs/boa-dit
python -u -m boa_dit.w4a4.caches --what embeds,design --device cuda:1 2>&1 | grep -E "^embedded|design|cache|Error|Traceback" | grep -v "pad>" | cut -c1-200
python -u -m boa_dit.w4a4.cfg_branch --arm hsvd_r4_b0.5_kv16_u2_rs --device cuda:1 2>&1 | grep -E "ratio|Error|Traceback"
python -u -m boa_dit.w4a4.band_gain --arm hsvd_r4_b0.5_kv16_u2_rs --device cuda:1 2>&1 | grep -E "step|bitexact|Error|Traceback"
echo W4A4_DESIGN_DONE
