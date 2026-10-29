"""Back up new pilot checkpoints remotely: every file under ckpt/dit_joint/ not yet in handoff/dit_joint_assets.json is uploaded as a release asset
of tag `dit-joint-assets` (existing repo visibility), downloaded back and hash-verified, then recorded (name, bytes, sha256, stage).
Run: python scripts/backup_checkpoints.py [--stage S]"""
import argparse
import hashlib
import json
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TAG = "dit-joint-assets"
REC = os.path.join(ROOT, "handoff", "dit_joint_assets.json")
LIMIT = 1900 * 1024 * 1024


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="")
    a = ap.parse_args()
    src = os.path.join(ROOT, "ckpt", "dit_joint")
    if not os.path.isdir(src):
        return
    rec = json.load(open(REC)) if os.path.exists(REC) else {"tag": TAG, "files": {}}
    if subprocess.run(["gh", "release", "view", TAG], capture_output=True).returncode != 0:
        subprocess.run(["gh", "release", "create", TAG, "--title", TAG, "--notes", "DiT joint-feedback pilot checkpoints (see handoff/dit_joint_assets.json)"], check=True)
    for dp, _, fn in os.walk(src):
        for f in fn:
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, src)
            h = sha256(p)
            if rec["files"].get(rel, {}).get("sha256") == h:
                continue
            assert os.path.getsize(p) < LIMIT, f"{rel} exceeds the release-asset part limit: split it before backing up"
            name = rel.replace("/", "__")
            tmp = os.path.join(ROOT, "data", "_upload", name)                 # release asset names are flat: upload a copy named rel with / -> __
            os.makedirs(os.path.dirname(tmp), exist_ok=True)
            subprocess.run(["cp", p, tmp], check=True)
            subprocess.run(["gh", "release", "upload", TAG, tmp, "--clobber"], check=True)
            os.remove(tmp)
            vd = os.path.join(ROOT, "data", "_verify")
            os.makedirs(vd, exist_ok=True)
            subprocess.run(["gh", "release", "download", TAG, "-p", name, "-D", vd, "--clobber"], check=True)
            ok = sha256(os.path.join(vd, name)) == h
            os.remove(os.path.join(vd, name))
            rec["files"][rel] = {"asset": name, "bytes": os.path.getsize(p), "sha256": h, "stage": a.stage, "verified": ok}
            print(f"  backed up {rel}: verified {ok}", flush=True)
    json.dump(rec, open(REC, "w"), indent=1)


if __name__ == "__main__":
    main()
