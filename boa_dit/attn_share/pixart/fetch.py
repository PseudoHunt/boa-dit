"""Fetch PixArt-Sigma-XL-2-1024-MS into a local dir; the fp32 T5 shards are converted to fp16 one at a time (disk)."""
import json
import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file, save_file

REPO = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
DST = "/home/jl_fs/pixart_sigma_1024"

if __name__ == "__main__":
    snapshot_download(REPO, local_dir=DST, ignore_patterns=["text_encoder/model-*.safetensors", "asset/*"])
    idx = json.load(open(os.path.join(DST, "text_encoder", "model.safetensors.index.json")))
    for shard in sorted(set(idx["weight_map"].values())):
        out = os.path.join(DST, "text_encoder", shard)
        if os.path.exists(out + ".fp16done"):
            continue
        p = hf_hub_download(REPO, f"text_encoder/{shard}", local_dir=DST + "_tmp")
        sd = {k: v.half() for k, v in load_file(p).items()}
        save_file(sd, out + ".tmp", metadata={"format": "pt"})
        os.replace(out + ".tmp", out)
        os.remove(p)
        open(out + ".fp16done", "w").write("converted from fp32\n")
        print("converted", shard, flush=True)
    print("FETCH_DONE", flush=True)
