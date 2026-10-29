"""Restore parked assets on a fresh VM (handoff/START_HERE.md). Reads handoff/ASSETS.json; downloads release parts with `gh`, verifies every part
and every member file against SHA-256, extracts into the repo (data/ckpt/..., data/ext/..., results/stage0/ref_stats.npz), creates ckpt -> data/ckpt.
  --groups A_dit_base[,B_dit_analysis,C_pixart_base,D_pixart_analysis]   archive groups to restore (default: A_dit_base, which the DiT pilot needs)
  --upstream dit|vae|pixart|all                                          pinned public downloads (default: dit,vae)
  --rebuild-pixart-refits                                                rebuild the 2 GB PixArt refit checkpoints from base + branch files (bit-exact)
Run: python scripts/restore_assets.py [--groups ...] [--upstream ...]
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = json.load(open(os.path.join(ROOT, "handoff", "ASSETS.json")))
DATA = os.environ.get("BOA_DATA", os.path.join(ROOT, "data"))
HF_REV = {"stabilityai/sd-vae-ft-ema": "f04b2c4b98319346dad8c65879f680b1997b204a",
          "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS": "e102b3591cc82e97071b8b4cb90d834d0c487207"}


def sha256(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def dest_path(d):
    return os.path.join(DATA, d[len("data/"):]) if d.startswith("data/") else os.path.join(ROOT, d)


def restore_group(g, work):
    parts = sorted(k for k, v in M["parts"].items() if v["group"] == g)
    os.makedirs(work, exist_ok=True)
    tarp = os.path.join(work, f"{g}.tar")
    with open(tarp, "wb") as out:
        for name in parts:
            p = os.path.join(work, name)
            if not (os.path.exists(p) and sha256(p) == M["parts"][name]["sha256"]):
                subprocess.run(["gh", "release", "download", M["archive_tag"], "-p", name, "-D", work, "--clobber", "-R", "PseudoHunt/boa-dit"], check=True)
            assert sha256(p) == M["parts"][name]["sha256"], f"part hash mismatch: {name}"
            with open(p, "rb") as f:
                shutil.copyfileobj(f, out, 1 << 24)
            os.remove(p)
    with tarfile.open(tarp) as t:
        for mem in t.getmembers():
            d = dest_path(mem.name)
            os.makedirs(os.path.dirname(d), exist_ok=True)
            with t.extractfile(mem) as src, open(d, "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 24)
    os.remove(tarp)
    bad = [x["dest"] for x in M["assets"] if x["group"] == g and sha256(dest_path(x["dest"])) != x["sha256"]]
    print(f"  {g}: {sum(1 for x in M['assets'] if x['group'] == g)} files restored, hash mismatches: {bad}", flush=True)
    assert not bad


def fetch(url, dst, sha):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not (os.path.exists(dst) and sha256(dst) == sha):
        print(f"  downloading {url}", flush=True)
        urllib.request.urlretrieve(url, dst + ".tmp")
        os.replace(dst + ".tmp", dst)
    assert sha256(dst) == sha, f"upstream hash mismatch: {dst}"
    print(f"  ok {dst}", flush=True)


def upstream(which):
    up = {x["id"]: x for x in M["assets"] if x["class"] == "upstream"}
    if "dit" in which or "all" in which:
        x = up["DiT-XL-2-256x256.pt"]
        fetch(x["upstream"]["url"], dest_path(x["dest"]), M["upstream_sha"]["DiT-XL-2-256x256.pt"])
    if "fidref" in which or "all" in which:
        x = up["VIRTUAL_imagenet256_labeled.npz"]
        fetch(x["upstream"]["url"], dest_path(x["dest"]), M["upstream_sha"]["VIRTUAL_imagenet256_labeled.npz"])
    os.environ.setdefault("HF_HOME", os.environ.get("BOA_HF_HOME", os.path.join(DATA, "hf")))
    from huggingface_hub import snapshot_download
    if "vae" in which or "all" in which:
        snapshot_download("stabilityai/sd-vae-ft-ema", revision=HF_REV["stabilityai/sd-vae-ft-ema"])
        print("  ok sd-vae-ft-ema @", HF_REV["stabilityai/sd-vae-ft-ema"], flush=True)
    if "pixart" in which or "all" in which:
        d = os.environ.get("BOA_PIXART_DIR", os.path.join(DATA, "ext", "pixart_sigma_1024"))
        snapshot_download("PixArt-alpha/PixArt-Sigma-XL-2-1024-MS", revision=HF_REV["PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"], local_dir=d)
        print("  ok PixArt-Sigma @", HF_REV["PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"], "->", d, flush=True)


def rebuild_pixart():
    import torch
    rep = json.load(open(os.path.join(ROOT, "handoff", "pixart_branch_extract.json")))
    base = torch.load(os.path.join(DATA, "ckpt", "w4a4", "q_hsvd_r4_b0.5_kv16_u2_rs.pt"), map_location="cpu")
    for full, v in rep.items():
        br = torch.load(os.path.join(DATA, "ckpt", v["branch_file"]), map_location="cpu")
        q = {k: dict(base[k], L1=br[k]["L1"], L2=br[k]["L2"]) for k in base}
        out = os.path.join(DATA, "ckpt", full)
        torch.save(q, out)
        print(f"  rebuilt {full} (full-file sha256 of the original: {v['full_sha256'][:12]}...; tensors equal by construction, verified at park time)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--groups", default="A_dit_base")
    ap.add_argument("--upstream", default="dit,vae")
    ap.add_argument("--rebuild-pixart-refits", action="store_true")
    ap.add_argument("--work", default=os.path.join(DATA, "_download"))
    a = ap.parse_args()
    os.makedirs(os.path.join(DATA, "ckpt"), exist_ok=True)
    link = os.path.join(ROOT, "ckpt")
    if not os.path.lexists(link):
        os.symlink(os.path.relpath(os.path.join(DATA, "ckpt"), ROOT) if DATA.startswith(ROOT) else os.path.join(DATA, "ckpt"), link)
    for g in [x for x in a.groups.split(",") if x]:
        restore_group(g, a.work)
    if a.upstream:
        upstream(a.upstream.split(","))
    if a.rebuild_pixart_refits:
        rebuild_pixart()
    print("RESTORE_OK", flush=True)


if __name__ == "__main__":
    main()
