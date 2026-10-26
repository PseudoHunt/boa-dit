cd /home/jl_fs/boa-dit
while pgrep -f "debug_base --what split --device cuda:1" >/dev/null; do sleep 20; done
python -u -m boa_dit.w4a4.debug_base --what split --variants kv16,u2 --device cuda:1 2>&1 | grep -E "kv16|u2:|full W4A4|Error|Traceback"
for flags in "--range_search" "--kv16 --fc2_unsigned --range_search"; do
  name=$(python -c "import argparse; from boa_dit.w4a4.quantize import arm_name; a=argparse.Namespace(method='hsvd', r=4, subspace='beta', beta=0.5, lam_A=0.05, suffix='', r_config=None, order='fwd', target='sym', kv16=False, fc2_unsigned=False, range_search=False)
for f in '$flags'.split(): setattr(a, f[2:], True)
print(arm_name(a))")
  bash scripts/run_w4a4_q.sh cuda:1 "--method hsvd --r 4 $flags|$name"
  python -m boa_dit.w4a4.gen --arm $name --n 64 --device cuda:1 2>&1 | grep -E "psnr|Error|Traceback" | cut -c1-200
done
echo W4A4_GPU1_DEBUG_DONE
