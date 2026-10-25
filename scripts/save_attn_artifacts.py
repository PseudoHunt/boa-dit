"""Gate 3 artifacts at k = 20 from the Gate 1 mean maps: V3 kernels (git), V2 windowed maps and V1 maps (git LFS)."""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.attn_share.kernel_support import kernels  # noqa: E402
from boa_dit.common import atomic_torch_save  # noqa: E402

K = 20
SRC = os.path.join(ROOT, "results", "attn_share", "maps")
DST = os.path.join(ROOT, "results", "attn_share", "artifacts")
META = {"k": K, "windows": [f"s{w}-{w + 4}" for w in range(0, K, 5)], "source": "Gate 1 mean maps (seeds 1000-1031)",
        "layout": "[window or step, block 0-27, head 0-15, ...]; maps are rows = query position i, cols = key j on the 16x16 grid"}
kap, v2 = {}, {}
for br in ("cond", "uncond"):
    M = torch.load(os.path.join(SRC, f"M_{br}.pt"), map_location="cpu", mmap=True)[:K]
    kap[br] = kernels(M, K)                                                         # [4, 28, 16, 31, 31] fp32
    v2[br] = torch.stack([M[w:w + 5].float().mean(0) for w in range(0, K, 5)]).half()   # [4, 28, 16, 256, 256]
    atomic_torch_save({"M": M.clone(), **META, "branch": br}, os.path.join(DST, f"V1_maps_{br}_s0-19.pt"))
atomic_torch_save({"kappa": kap, **META, "kernel": "31x31, index [dy + 15, dx + 15] = mean of M[i, j] over pairs with "
                   "pos_j - pos_i = (dy, dx); row-normalize kappa[offsets] at use"}, os.path.join(DST, "V3_kernels_k20.pt"))
atomic_torch_save({"maps": v2, **META}, os.path.join(DST, "V2_windowed_maps_k20.pt"))
print({f: os.path.getsize(os.path.join(DST, f)) for f in os.listdir(DST)})
