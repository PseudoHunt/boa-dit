"""Download the pinned PUBLIC upstream inputs (no project-trained artifacts are hosted anywhere; those are regenerated: handoff/REGENERATION.md).
  --what dit,vae,fidref[,pixart,mjhq]   (default dit,vae,fidref = what the DiT pilot needs)
Files land under the data root (BOA_DATA, default ./data); ./ckpt is a symlink to data/ckpt (the code's CKPT_DIR).
Run: python scripts/fetch_upstream.py [--what ...]
"""
import argparse
import hashlib
import json
import os
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("BOA_DATA", os.path.join(ROOT, "data"))
REF = json.load(open(os.path.join(ROOT, "handoff", "REFERENCE_HASHES.json")))["upstream"]
URLS = {"DiT-XL-2-256x256.pt": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt",
        "VIRTUAL_imagenet256_labeled.npz": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"}
HF = {"vae": ("stabilityai/sd-vae-ft-ema", "f04b2c4b98319346dad8c65879f680b1997b204a", None),
      "pixart": ("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", "e102b3591cc82e97071b8b4cb90d834d0c487207", "ext/pixart_sigma_1024")}
MJHQ = ("playgroundai/MJHQ-30K", "15b0a659e066e763d0e9a6cd8f00e25f8af5e084", "meta_data.json")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def fetch(name):
    dst = os.path.join(DATA, "ckpt", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not (os.path.exists(dst) and sha256(dst) == REF[name]):
        print(f"  downloading {URLS[name]}", flush=True)
        urllib.request.urlretrieve(URLS[name], dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    ok = sha256(dst) == REF[name]
    print(f"  {name}: sha256 {'OK' if ok else 'MISMATCH'}", flush=True)
    assert ok, f"upstream hash mismatch: {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="dit,vae,fidref")
    w = ap.parse_args().what.split(",")
    os.makedirs(os.path.join(DATA, "ckpt"), exist_ok=True)
    link = os.path.join(ROOT, "ckpt")
    if not os.path.lexists(link):
        os.symlink(os.path.join(DATA, "ckpt"), link)
    if "dit" in w:
        fetch("DiT-XL-2-256x256.pt")
    if "fidref" in w:
        fetch("VIRTUAL_imagenet256_labeled.npz")
    os.environ.setdefault("HF_HOME", os.environ.get("BOA_HF_HOME", os.path.join(DATA, "hf")))
    from huggingface_hub import hf_hub_download, snapshot_download
    for k in ("vae", "pixart"):
        if k in w:
            repo, rev, local = HF[k]
            snapshot_download(repo, revision=rev, local_dir=os.environ.get("BOA_PIXART_DIR", os.path.join(DATA, local)) if local else None)
            print(f"  {repo} @ {rev}: OK", flush=True)
    if "mjhq" in w:
        d = os.path.join(os.environ.get("BOA_EXT_DIR", os.path.join(DATA, "ext")), "mjhq")
        hf_hub_download(MJHQ[0], MJHQ[2], revision=MJHQ[1], repo_type="dataset", local_dir=d)
        print(f"  {MJHQ[0]} @ {MJHQ[1]}: {MJHQ[2]} OK", flush=True)
    print("FETCH_OK", flush=True)


if __name__ == "__main__":
    main()
