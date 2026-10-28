"""ImageReward v1.0 per image for the PixArt refit arms (results/e3c/PROTOCOL.md Part 2); runs in the ImageReward venv.
Run: $IRPY scripts/px_ir.py"""
import json, os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import ImageReward as RM
from PIL import Image
from boa_dit.common import Run, atomic_write_json
from boa_dit.e3c.pixart_refit import NAMES, OUT, SETS
from boa_dit.f5 import STAGE
from boa_dit.temporal import ext_w4a4 as EXT
from boa_dit.w4a4 import common as WC
dev = "cuda:0"
prompts = WC.load_embeds("mjhq1k")["prompts"]
res = {"model": "ImageReward-v1.0"}
with Run(STAGE, "e3c_px_ir", dev, {}) as run:
    m = RM.load("ImageReward-v1.0", device=dev)
    for setn, (lo, hi) in SETS.items():
        res[setn] = {}
        for arm in ("ref", "P0", "P1", "P2"):
            imgs, _ = EXT.load(NAMES[setn][arm], lo, hi)
            res[setn][arm] = [float(m.score(prompts[lo + k], [Image.fromarray(imgs[k].permute(1, 2, 0).numpy().astype("uint8"))])) for k in range(len(imgs))]
            print(f"  IR {setn} {arm}: {np.mean(res[setn][arm]):+.4f}", flush=True)
    res["run"] = run.record()
atomic_write_json(os.path.join(OUT, "ir.json"), res)
