"""B3 setup (rule.md): the unchanged w4a4 drivers, logged under this phase's stage.
  calib : w4a4.caches --what calib (block-0 inputs of 32 calibration prompts)
  quant : w4a4.quantize --method hsvd --r 0 --kv16 --range_search  -> ckpt/w4a4/q_hsvd_r0_b0.5_kv16_rs.pt
Run: python -m boa_dit.temporal.w4build calib | quant
"""
import sys

from boa_dit.common import Run
from boa_dit.temporal.common import STAGE

W4_ARM = "hsvd_r0_b0.5_kv16_rs"


def main():
    cmd, dev = sys.argv[1], "cuda:0"
    tag = lambda stage, name, d, cfg: Run(STAGE, f"b3_{name}", d, cfg)
    if cmd == "calib":
        from boa_dit.w4a4 import caches as M
        sys.argv = [sys.argv[0], "--what", "calib", "--device", dev]
    else:
        from boa_dit.w4a4 import quantize as M
        sys.argv = [sys.argv[0], "--method", "hsvd", "--r", "0", "--kv16", "--range_search", "--device", dev]
    M.Run = tag
    M.main()


if __name__ == "__main__":
    main()
