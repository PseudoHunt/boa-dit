"""Verify a restored checkout (handoff/START_HERE.md): files and hashes, imports, the heig-8 W4 + carry model, a tiny inference smoke check,
and (on the SAME hardware/software only) bitwise agreement with archived latents.
  python scripts/verify_restore.py [--smoke] [--bitwise]
--smoke   : DiT-XL/2 + heig-8 W4 + rank-8 carry, DDIM-50, cfg 1.5, seeds 20000-20003; finite, sane RMS; writes handoff/verify_restore.json
--bitwise : compare the first 32 paired-eval latents with the archived E3C C0 latents (needs B_dit_analysis); expected bitwise ONLY on an A30 with the
            ENVIRONMENT.md stack. On other hardware, report the relative difference and re-establish same-machine controls instead of reusing old numbers.
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha256(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--bitwise", action="store_true")
    a = ap.parse_args()
    M = json.load(open(os.path.join(ROOT, "handoff", "ASSETS.json")))
    data = os.environ.get("BOA_DATA", os.path.join(ROOT, "data"))
    dp = lambda d: os.path.join(data, d[5:]) if d.startswith("data/") else os.path.join(ROOT, d)
    res = {"files": {}}
    for x in M["assets"]:
        if x["group"] == "A_dit_base" or x["id"] == "DiT-XL-2-256x256.pt":
            p = dp(x["dest"])
            want = x.get("sha256") or M["upstream_sha"].get(x["id"])
            res["files"][x["dest"]] = os.path.exists(p) and sha256(p) == want
    res["files_ok"] = all(res["files"].values())
    print("files ok:", res["files_ok"], [k for k, v in res["files"].items() if not v], flush=True)
    import torch
    res["torch"], res["cuda"], res["gpu"] = torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    from boa_dit.deflate.carry import install
    from boa_dit.model import labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
    from boa_dit.sample import ddim_sample
    res["imports_ok"] = True
    if a.smoke or a.bitwise:
        dev = "cuda:0"
        m = load_dit(dev)
        load_quant_state(m, os.path.join(ROOT, "ckpt", "ckpt_deflate_heig_r8_w4", "model.pt"))
        install(m, torch.load(os.path.join(ROOT, "ckpt", "temporal", "rebuild_quant", "deflate_heig_r8_w4", "carry.pt"), map_location="cpu"))
        d = make_diffusion(50)
        seeds = list(range(20000, 20032)) if a.bitwise else list(range(20000, 20004))
        with torch.no_grad():
            x = ddim_sample(m, d, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), 1.5).cpu()
        res["smoke"] = {"finite": bool(torch.isfinite(x).all()), "rms": float(x.pow(2).mean().sqrt()), "n": len(seeds)}
        print("smoke:", res["smoke"], flush=True)
        if a.bitwise:
            ref_f = os.path.join(ROOT, "ckpt", "e3c", "eval_latents.pt")
            if os.path.exists(ref_f):
                ref = torch.load(ref_f)["C0"][:32]
                res["bitwise_vs_archived_C0"] = {"equal": bool(torch.equal(x, ref)), "rel_diff": float((x - ref).norm() / ref.norm()),
                                                 "note": "bitwise expected only on A30 + the ENVIRONMENT.md stack"}
                print("bitwise vs archived E3C C0 (seeds 20000-20031):", res["bitwise_vs_archived_C0"], flush=True)
            else:
                res["bitwise_vs_archived_C0"] = "skipped: B_dit_analysis not restored"
    out = os.path.join(ROOT, "handoff", "verify_restore.json")
    json.dump(res, open(out, "w"), indent=1)
    print("VERIFY_DONE ->", out, flush=True)


if __name__ == "__main__":
    main()
