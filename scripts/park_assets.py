"""Park the project's non-Git assets (handoff/README in handoff/START_HERE.md).
  plan     : walk the old data root, classify every file by the explicit rules below, write handoff/ASSETS.json (no hashes yet)
  extract  : PixArt refit checkpoints -> branch-only files (L1/L2), verified to rebuild the full checkpoint bit-exactly against the base
  pack     : archive groups -> tar parts (< 1.9 GB each) in a staging dir, SHA-256 of every member file and every part
  upload   : gh release upload of every part to the archive tag (existing repo visibility), then download each part back and re-hash
Run on the OLD VM only: python scripts/park_assets.py plan|extract|pack|upload --old-root /home/boa-dit --stage /home/park_stage
"""
import argparse
import fnmatch
import hashlib
import json
import os
import subprocess
import tarfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "handoff", "ASSETS.json")
TAG = "park-assets-2026-10-29"
PART = 1900 * 1024 * 1024

# (glob relative to <old-root>/ckpt unless it starts with '@' (absolute) or 'repo:' (repo-relative), group, class, role)
# classes: essential_resume_pilot | essential_resume_parked | essential_analysis | regenerable | upstream | scratch
RULES = [
    # ---- upstream (pinned public downloads; not re-hosted) ----
    ("DiT-XL-2-256x256.pt", None, "upstream", "DiT-XL/2 256 weights (facebookresearch/DiT release)"),
    ("VIRTUAL_imagenet256_labeled.npz", None, "upstream", "ADM ImageNet-256 FID reference batch (OpenAI guided-diffusion)"),
    # ---- DiT base for the joint pilot ----
    ("ckpt_deflate_heig_r8_w4/model.pt", "A_dit_base", "essential_resume_pilot", "heig-8 W4 DiT state (rebuild.py output; the W4 base of every DiT carry experiment)"),
    ("ckpt_deflate_heig_r8_w4/block*.pt", None, "scratch", "per-block resume files of the heig-8 build (superseded by model.pt)"),
    ("ckpt_deflate_heig_r8_w4/carry*.pt", None, "scratch", "per-block resume carry files of the heig-8 build (superseded by the rebuild carry)"),
    ("temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt", "A_dit_base", "essential_resume_pilot", "heig-8 rank-8 carry (bitwise equal to the git carry)"),
    ("temporal/dit_smooth_s.pt", "A_dit_base", "essential_resume_pilot", "DiT static-A8 smoothing vectors (temporal-oracles)"),
    ("temporal/dit_static_b_*.pt", "A_dit_base", "essential_resume_pilot", "DiT static-A8 activation tables (single6/single8/step8)"),
    ("temporal/fp_traj*.pt", "A_dit_base", "essential_analysis", "FP trajectory caches used by the temporal analyses"),
    ("temporal/stage1_delta_heig8.pt", "A_dit_base", "essential_analysis", "temporal stage-1 records"),
    ("fp_eval_*.pt", "A_dit_base", "essential_analysis", "FP eval latents/pixels used by eval_latent_mse"),
    ("repo:results/stage0/ref_stats.npz", "A_dit_base", "essential_resume_pilot", "pytorch-fid reference statistics of the ADM batch (sha a11af8b0...)"),
    # ---- DiT analysis ----
    ("ckpt_deflate_heig_r16_w4/model.pt", "B_dit_analysis", "essential_analysis", "heig-16 W4 DiT state (E3C comparator H16)"),
    ("e3c/carry_*.pt", "B_dit_analysis", "essential_analysis", "E3C trained carries (15 runs)"),
    ("e3c/eval_latents.pt", "B_dit_analysis", "essential_analysis", "E3C paired-eval latents (seeds 20000-20127; FP + all arms)"),
    ("e3c/m_C2_s0.pt", "B_dit_analysis", "essential_analysis", "M coefficients for C2 s0"),
    ("e3c/fid/*", "B_dit_analysis", "essential_analysis", "E3C FID-2k latents and per-image Inception features"),
    ("dns/cal_x0.partial", None, "scratch", "partial file of a finished run"),
    ("dns/*", "B_dit_analysis", "essential_analysis", "DNS calibration samples, C-DNS traces"),
    ("dns/*/*", "B_dit_analysis", "essential_analysis", "DNS / C-DNS / Q-Drift paired and FID latents, Inception features"),
    ("observability/*", "B_dit_analysis", "essential_analysis", "paired FP/heig-8 traces 7000-7063 / 7100-7163 (E1 and observability)"),
    ("followup/e1/*", "B_dit_analysis", "essential_analysis", "follow-up E1 latents/images"),
    ("followup/e3/*", "B_dit_analysis", "essential_analysis", "follow-up E3 cache, final carries, eval latents/images"),
    # ---- PixArt base ----
    ("w4a4/q_hsvd_r4_b0.5_kv16_u2_rs.pt", "C_pixart_base", "essential_resume_parked", "PixArt-Σ corrected H-SVDQuant-r4 W4A4 base tables (R, L1, L2, d, act_shift)"),
    ("w4a4/q_*.pt", None, "regenerable", "other W4A4 builder outputs (w4a4.quantize)"),
    ("w4a4/embeds_*.pt", "C_pixart_base", "essential_resume_parked", "T5 prompt embeddings (calib/design/screen/mjhq1k)"),
    ("@/root/ext/deepcompressor/examples/diffusion/prompts/qdiff.yaml", "C_pixart_base", "essential_resume_parked", "calibration/design prompt list (deepcompressor qdiff.yaml)"),
    ("@/root/ext/mjhq/meta_data.json", "C_pixart_base", "essential_resume_parked", "MJHQ-30K metadata (prompt order for screen/mjhq1k sets)"),
    ("w4a4/calib_block0.pt", None, "regenerable", "PixArt block-0 calibration cache"),
    # ---- PixArt analysis ----
    ("e3c/pixart/q_design_*.pt", None, "regenerable", "full PixArt refit checkpoints: rebuilt bit-exactly from the base + branch files (D_pixart_analysis)"),
    ("e3c/pixart/q_profile_*.pt", None, "scratch", "profiling runs"),
    ("e3c/pixart/diag_tmp_q.pt", None, "scratch", "temporary diagnostic file"),
    ("e3c/pixart/*", "D_pixart_analysis", "essential_analysis", "PixArt refit cache, diagnostic endpoints, image grids, branch files"),
    ("temporal/pixart/calib_hist.pt", None, "regenerable", "PixArt static-A8 calibration histogram"),
    ("temporal/pixart/*.pt", "D_pixart_analysis", "essential_analysis", "PixArt static-A8 tables, design cache"),
    ("temporal/pixart/img/*/*", "D_pixart_analysis", "essential_analysis", "PixArt generated images/latents (activation arms, bf16 references, MJHQ-64 base)"),
    ("followup/e2/**", "D_pixart_analysis", "essential_analysis", "follow-up E2 images"),
    ("w4a4/img/**", "D_pixart_analysis", "essential_analysis", "W4A4 phase images (bf16, fp16)"),
    # ---- regenerable DiT caches ----
    ("temporal/dit_calib_hist.pt", None, "regenerable", "DiT static-A8 calibration histogram"),
    ("temporal/heig8_layers.pt", None, "regenerable", "heig-8 per-layer PHP/scale/zero/V"),
    ("temporal/pair*.pt", None, "regenerable", "temporal pair analysis caches"),
    ("ckpt_deflate_heig_r8_w5/*", None, "regenerable", "heig-8 at W5 (exploratory comparator)"),
    ("ckpt_deflate_heig_r8_w6/*", None, "regenerable", "heig-8 at W6 (exploratory comparator)"),
    ("temporal/rebuild_quant/deflate_heig_r8_w5/*", None, "regenerable", "W5 carry"),
    ("temporal/rebuild_quant/deflate_heig_r8_w6/*", None, "regenerable", "W6 carry"),
    ("weights_joint/*", None, "regenerable", "weights-joint pilot caches (completed experiment; results in git)"),
    ("ckpt_deflate_heig_r16_w4/block*.pt", None, "scratch", "build leftovers"),
]
REGEN = {
    "temporal/dit_calib_hist.pt": "python -m boa_dit.temporal.c1 calib (temporal-oracles; ≈ 0.3 A30-h; see temporal/common.py)",
    "temporal/heig8_layers.pt": "python -m boa_dit.temporal.rebuild --device cuda:0 (≈ 0.1 A30-h)",
    "temporal/pair*.pt": "python -m boa_dit.temporal.pair (temporal-oracles; ≈ 0.5 A30-h)",
    "ckpt_deflate_heig_r8_w5/*": "python -m boa_dit.temporal.rebuild --bits 5 (≈ 0.1 A30-h)",
    "ckpt_deflate_heig_r8_w6/*": "python -m boa_dit.temporal.rebuild --bits 6 (≈ 0.1 A30-h)",
    "temporal/rebuild_quant/deflate_heig_r8_w5/*": "python -m boa_dit.temporal.rebuild --bits 5",
    "temporal/rebuild_quant/deflate_heig_r8_w6/*": "python -m boa_dit.temporal.rebuild --bits 6",
    "w4a4/calib_block0.pt": "python -m boa_dit.w4a4.caches (≈ 0.2 A30-h; needs the PixArt upstream model)",
    "w4a4/q_*.pt": "python -m boa_dit.w4a4.quantize --arm <name> (needs calib_block0.pt)",
    "temporal/pixart/calib_hist.pt": "python -m boa_dit.temporal.pixart calib (temporal-oracles; ≈ 0.3 A30-h)",
    "weights_joint/*": "python -m boa_dit.weights_joint.<module> per results/temporal/weights_joint/PROTOCOL.md (weights-joint branch)",
    "e3c/pixart/q_design_*.pt": "python scripts/restore_assets.py --rebuild-pixart-refits (base + branch files; bit-exact)",
}
UPSTREAM = {
    "DiT-XL-2-256x256.pt": {"url": "https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt"},
    "VIRTUAL_imagenet256_labeled.npz": {"url": "https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz"},
}


def sha256(p, buf=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(buf)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve(rule, old):
    g = rule[0]
    if g.startswith("@"):
        return [g[1:]] if os.path.exists(g[1:]) else []
    if g.startswith("repo:"):
        p = os.path.join(old, g[5:])
        return [p] if os.path.exists(p) else []
    out = []
    base = os.path.join(old, "ckpt")
    for dp, dn, fn in os.walk(base, followlinks=True):
        for f in fn:
            rel = os.path.relpath(os.path.join(dp, f), base)
            if fnmatch.fnmatch(rel, g.replace("**", "*")):
                out.append(os.path.join(dp, f))
    return out


def dest_of(path, old):
    if path.startswith(os.path.join(old, "ckpt")):
        return "data/ckpt/" + os.path.relpath(path, os.path.join(old, "ckpt"))
    if path.startswith(old):
        return os.path.relpath(path, old)                                      # repo-relative (e.g. results/stage0/ref_stats.npz)
    if "/root/ext/" in path:
        return "data/ext/" + path.split("/root/ext/")[1]
    return "data/other/" + os.path.basename(path)


def plan(a):
    seen, assets = set(), []
    for rule in RULES:
        for p in sorted(resolve(rule, a.old_root)):
            if p in seen:
                continue
            seen.add(p)
            rel = os.path.relpath(p, os.path.join(a.old_root, "ckpt")) if p.startswith(os.path.join(a.old_root, "ckpt")) else p
            assets.append({"id": rel, "old_path": p, "dest": dest_of(p, a.old_root), "group": rule[1], "class": rule[2], "role": rule[3],
                           "bytes": os.path.getsize(p), "regen": next((v for k, v in REGEN.items() if fnmatch.fnmatch(rel, k)), None),
                           "upstream": UPSTREAM.get(rel)})
    # unmatched files (must be empty: nothing silently left only on the old disk)
    base = os.path.join(a.old_root, "ckpt")
    unmatched = []
    for dp, dn, fn in os.walk(base, followlinks=True):
        for f in fn:
            p = os.path.join(dp, f)
            if p not in seen:
                unmatched.append(os.path.relpath(p, base))
    m = {"archive_tag": TAG, "old_root": a.old_root, "assets": assets, "unmatched": unmatched}
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    json.dump(m, open(MANIFEST, "w"), indent=1)
    by = {}
    for x in assets:
        k = (x["class"], x["group"])
        by[k] = by.get(k, 0) + x["bytes"]
    for k, v in sorted(by.items(), key=lambda kv: str(kv[0])):
        print(f"  {str(k):60s} {v / 1e9:8.2f} GB")
    print(f"  unmatched: {len(unmatched)}", unmatched[:20])


def extract(a):
    """PixArt refit checkpoints -> branch-only (L1/L2 bf16) files; verify base + branch == full checkpoint, tensor by tensor."""
    import torch
    m = json.load(open(MANIFEST))
    base = torch.load(os.path.join(a.old_root, "ckpt", "w4a4", "q_hsvd_r4_b0.5_kv16_u2_rs.pt"), map_location="cpu")
    outd = os.path.join(a.old_root, "ckpt", "e3c", "pixart", "branches")
    os.makedirs(outd, exist_ok=True)
    rep = {}
    for x in m["assets"]:
        if not fnmatch.fnmatch(x["id"], "e3c/pixart/q_design_*.pt"):
            continue
        q = torch.load(x["old_path"], map_location="cpu")
        br = {k: {"L1": q[k]["L1"], "L2": q[k]["L2"]} for k in q}
        rebuilt = {k: dict(base[k], L1=br[k]["L1"], L2=br[k]["L2"]) for k in base}
        ok = set(rebuilt) == set(q) and all(
            set(rebuilt[k]) == set(q[k]) and all((rebuilt[k][f] is None and q[k][f] is None) or
                                                 (rebuilt[k][f] is not None and q[k][f] is not None and torch.equal(rebuilt[k][f], q[k][f])) for f in q[k])
            for k in q)
        name = os.path.basename(x["id"]).replace("q_", "branch_")
        torch.save(br, os.path.join(outd, name))
        rep[x["id"]] = {"branch_file": f"e3c/pixart/branches/{name}", "rebuild_bit_exact": bool(ok), "full_sha256": sha256(x["old_path"])}
        print(f"  {x['id']}: rebuild bit-exact {ok}", flush=True)
    json.dump(rep, open(os.path.join(ROOT, "handoff", "pixart_branch_extract.json"), "w"), indent=1)
    assert all(v["rebuild_bit_exact"] for v in rep.values()), "a branch extract does not rebuild its checkpoint"


def pack(a):
    m = json.load(open(MANIFEST))
    os.makedirs(a.stage, exist_ok=True)
    groups = sorted({x["group"] for x in m["assets"] if x["group"]})
    for x in m["assets"]:
        if x["group"] and "sha256" not in x:
            x["sha256"] = sha256(x["old_path"])
    if not m.get("upstream_sha"):
        m["upstream_sha"] = {x["id"]: sha256(x["old_path"]) for x in m["assets"] if x["class"] == "upstream"}
    parts = {}
    for g in groups:
        tarp = os.path.join(a.stage, f"{g}.tar")
        if not os.path.exists(tarp + ".done"):
            with tarfile.open(tarp, "w") as t:
                for x in m["assets"]:
                    if x["group"] == g:
                        t.add(x["old_path"], arcname=x["dest"])
            open(tarp + ".done", "w").close()
        # split
        n, i = os.path.getsize(tarp), 0
        with open(tarp, "rb") as f:
            while True:
                b = f.read(PART)
                if not b:
                    break
                pp = f"{tarp}.part{i:02d}"
                if not os.path.exists(pp):
                    open(pp, "wb").write(b)
                parts[os.path.basename(pp)] = {"group": g, "bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()}
                i += 1
        os.remove(tarp)
        print(f"  {g}: {n / 1e9:.2f} GB in {i} parts", flush=True)
    m["parts"] = parts
    json.dump(m, open(MANIFEST, "w"), indent=1)
    with open(os.path.join(ROOT, "handoff", "SHA256SUMS"), "w") as f:
        for k, v in sorted(parts.items()):
            f.write(f"{v['sha256']}  {k}\n")
        for x in m["assets"]:
            if x.get("sha256"):
                f.write(f"{x['sha256']}  {x['dest']}\n")
        for k, v in m["upstream_sha"].items():
            f.write(f"{v}  data/ckpt/{k}\n")


def upload(a):
    m = json.load(open(MANIFEST))
    r = subprocess.run(["gh", "release", "view", TAG], capture_output=True, text=True)
    if r.returncode != 0:
        subprocess.run(["gh", "release", "create", TAG, "--title", TAG, "--notes",
                        "Parked project assets (see handoff/ASSETS.json on branch dit-joint-feedback). Tar parts; reassemble with scripts/restore_assets.py."], check=True)
    have = set(json.loads(subprocess.run(["gh", "release", "view", TAG, "--json", "assets"], capture_output=True, text=True, check=True).stdout)["assets"][i]["name"]
               for i in range(len(json.loads(subprocess.run(["gh", "release", "view", TAG, "--json", "assets"], capture_output=True, text=True, check=True).stdout)["assets"])))
    for name in sorted(m["parts"]):
        if name not in have:
            subprocess.run(["gh", "release", "upload", TAG, os.path.join(a.stage, name)], check=True)
            print(f"  uploaded {name}", flush=True)
    # verify by download + hash
    vd = os.path.join(a.stage, "verify")
    os.makedirs(vd, exist_ok=True)
    for name, v in sorted(m["parts"].items()):
        p = os.path.join(vd, name)
        subprocess.run(["gh", "release", "download", TAG, "-p", name, "-D", vd, "--clobber"], check=True)
        ok = sha256(p) == v["sha256"]
        v["verified_download"] = ok
        os.remove(p)
        print(f"  verify {name}: {ok}", flush=True)
    json.dump(m, open(MANIFEST, "w"), indent=1)
    assert all(v["verified_download"] for v in m["parts"].values())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "extract", "pack", "upload"])
    ap.add_argument("--old-root", default="/home/boa-dit")
    ap.add_argument("--stage", default="/home/park_stage")
    a = ap.parse_args()
    {"plan": plan, "extract": extract, "pack": pack, "upload": upload}[a.cmd](a)
