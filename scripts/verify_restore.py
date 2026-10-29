"""Verify a freshly bootstrapped checkout (handoff/START_HERE.md): upstream hashes, regenerated artifacts, imports, a tiny inference smoke check.
Regenerated files are compared with the old VM's hashes (handoff/REFERENCE_HASHES.json) and REPORTED. Bitwise identity is expected only on the same
GPU/software stack; a mismatch elsewhere is not a failure, but old baseline numbers must then not be reused (the pilot re-measures its controls).
Run: python scripts/verify_restore.py [--smoke]   -> handoff/verify_restore.json
"""
import argparse
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    ref = json.load(open(os.path.join(ROOT, "handoff", "REFERENCE_HASHES.json")))
    ck = os.path.join(ROOT, "ckpt")
    res = {"upstream": {}, "regenerated": {}}
    for name, h in ref["upstream"].items():
        p = os.path.join(ck, name)
        res["upstream"][name] = os.path.exists(p) and sha256(p) == h
    for dest in ("results/stage0/ref_stats.npz", "data/ckpt/ckpt_deflate_heig_r8_w4/model.pt", "data/ckpt/temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt",
                 "data/ckpt/temporal/dit_smooth_s.pt", "data/ckpt/temporal/dit_static_b_single8.pt"):
        p = os.path.join(ROOT, dest) if not dest.startswith("data/") else os.path.join(ck, dest[len("data/ckpt/"):])
        old = ref["generated"].get(dest, {}).get("sha256")
        res["regenerated"][dest] = {"exists": os.path.exists(p), "bitwise_equal_to_old_vm": (os.path.exists(p) and old is not None and sha256(p) == old)}
    import torch
    res["torch"], res["cuda"] = torch.__version__, torch.version.cuda
    res["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    git_carry = os.path.join(ROOT, "results", "deflate", "quant", "deflate_heig_r8_w4", "carry.pt")
    new_carry = os.path.join(ck, "temporal", "rebuild_quant", "deflate_heig_r8_w4", "carry.pt")
    if os.path.exists(new_carry):
        o, n = torch.load(git_carry, map_location="cpu"), torch.load(new_carry, map_location="cpu")
        res["heig8_carry_rebuild_vs_git"] = {"bitwise": all(torch.equal(o[k]["V"], n[k]["V"]) and torch.equal(o[k]["delta"], n[k]["delta"]) for k in o),
                                             "max_rel_delta_diff": max(float((o[k]["delta"].float() - n[k]["delta"].float()).norm() / o[k]["delta"].float().norm()) for k in o)}
    from boa_dit.deflate.carry import install
    from boa_dit.model import labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
    from boa_dit.sample import ddim_sample
    res["imports_ok"] = True
    if a.smoke:
        dev = "cuda:0"
        m = load_dit(dev)
        load_quant_state(m, os.path.join(ck, "ckpt_deflate_heig_r8_w4", "model.pt"))
        install(m, torch.load(new_carry, map_location="cpu"))
        seeds = list(range(20000, 20004))
        with torch.no_grad():
            x = ddim_sample(m, make_diffusion(50), noise_for_seeds(seeds, dev), labels_for_seeds(seeds), 1.5)
            img = __import__("boa_dit.model", fromlist=["decode"]).decode(load_vae(dev), x)
        res["smoke"] = {"finite": bool(torch.isfinite(x).all()), "latent_rms": float(x.pow(2).mean().sqrt()), "decoded_shape": list(img.shape), "n": len(seeds)}
    out = os.path.join(ROOT, "handoff", "verify_restore.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    ok = all(res["upstream"][k] for k in ("DiT-XL-2-256x256.pt",)) and all(v["exists"] for v in res["regenerated"].values()) and (not a.smoke or res["smoke"]["finite"])
    print("VERIFY_OK" if ok else "VERIFY_FAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
