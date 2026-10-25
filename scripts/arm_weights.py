"""Effective bucket weight of a weighting arm, computed from boa_dit.weights exactly as quantize.py uses it.

Share of weight on steps 0-14 (buckets 0-2) and on steps 25-49 (buckets 5-9), averaged over every block,
layer and row group, weighted by the number of rows each p applies to (dead RTN rows excluded).
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit import weights as W  # noqa: E402
from boa_dit.model import LINEARS  # noqa: E402

S, B = 50, 10


def effective(mode, K=1, lam=1.0, tau=1.0, clip=0.0, gain_tag="gain_iso_rho0.003_fp32", gain_stat="median"):
    gates = W.load_gates()
    gain = W.load_gain(gain_tag, S, B, gain_stat) if mode in ("gain", "full") else None
    tot, acc = 0, np.zeros(B)
    for blk in range(28):
        for ln in LINEARS:
            for rows, p in W.bucket_weights(mode, ln, blk, gates, gain, S, B, K, lam, None, tau=tau, clip=clip):
                if p is None:
                    continue
                acc += len(rows) * np.asarray(p)
                tot += len(rows)
    p = acc / tot
    return {"p_mean": p.tolist(), "share_steps_0_14": float(p[:3].sum()), "share_steps_25_49": float(p[5:].sum()),
            "max_over_min": float(p.max() / p.min())}


if __name__ == "__main__":
    arms = json.loads(sys.argv[1])
    print(json.dumps({k: effective(**v) for k, v in arms.items()}, indent=1))
