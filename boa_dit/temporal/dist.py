"""Consolidated §3 (rule_distribution.md): distributional metrics for the activation method.
  gen2   : D2 images on untouched prompts mjhq1k[64:96] (batch 1, seed = index): bf16 refs, single table, full feedback, DCT-4
  fid    : DiT FID-2k for one Stage-3 arm (D3/D4/D5), via the stage-3 entry point
  ir     : ImageReward per image for a set of arms (runs in the ImageReward venv; scoring only, logged under c3)
  score  : D1/D2 tables: paired bootstraps vs single table (PSNR, LPIPS, latent, ImageReward)
Run: python -m boa_dit.temporal.dist gen2 | score
     <irenv>/bin/python -m boa_dit.temporal.dist ir --set screen|untouched
"""
import argparse
import os
import time

import numpy as np
import torch

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.temporal.c3 import STAGE, guard, spent
from boa_dit.temporal.common import OUT

DD = os.path.join(OUT, "dist")
SETS = {"screen": (0, 64), "untouched": (64, 96)}
# (image dir, label); screen dirs are the Stage-3 arms, untouched dirs are written by gen2
ARMS = {"screen": [("bf16_b1", "bf16 reference"), ("s3_single8_none", "single table"), ("s3_single8_unif", "uniform dither (best stateless)"),
                   ("s3_single8_feedback", "full feedback"), ("s3_single8_feedback_dct4", "DCT-4 feedback")],
        "untouched": [("u_bf16_b1", "bf16 reference"), ("u_single8_none", "single table"), ("u_single8_feedback", "full feedback"),
                      ("u_single8_feedback_dct4", "DCT-4 feedback")]}
CHUNK = 8


def load_imgs(name, lo, hi):
    from boa_dit.temporal import pixart as PX
    ch = [torch.load(os.path.join(PX.img_dir(name), f"{i:05d}.pt")) for i in range(lo - lo % CHUNK, hi, CHUNK)]
    off = lo % CHUNK
    return torch.cat([c["img"] for c in ch])[off:off + hi - lo], torch.cat([c["lat"] for c in ch])[off:off + hi - lo]


@torch.no_grad()
def gen2(dev):
    """D2: untouched prompts, batch 1, one arm at a time; chunk files are keyed by absolute prompt index (resumable)."""
    from boa_dit.temporal import pixart as PX
    from boa_dit.w4a4 import common as WC
    lo, hi = SETS["untouched"]
    e = WC.load_embeds("mjhq1k")
    specs = [("u_bf16_b1", None), ("u_single8_none", dict(kind="single8", dither="none", feedback=False, compress=None)),
             ("u_single8_feedback", dict(kind="single8", dither="none", feedback=True, compress=None)),
             ("u_single8_feedback_dct4", dict(kind="single8", dither="none", feedback=True, compress="dct4"))]
    for name, spec in specs:
        d = PX.img_dir(name)
        os.makedirs(d, exist_ok=True)
        todo = [i for i in range(lo, hi, CHUNK) if not os.path.exists(os.path.join(d, f"{i:05d}.pt"))]
        if not todo:
            continue
        with Run(STAGE, f"dist_gen2_{name}", dev, {"prompts": [lo, hi], "spec": spec}) as run:
            pipe = WC.load_pipe(dev)
            if spec is not None:
                PX.install_fb(pipe.transformer, spec["kind"], spec["dither"], spec["feedback"], spec["compress"])
                PX.STATE.mode = "static"
            first = True
            for i in todo:
                imgs, lats = [], []
                for j in range(i, min(hi, i + CHUNK)):
                    torch.cuda.synchronize(); t0 = time.time()
                    sc = PX.StepCounter(pipe.transformer)
                    try:
                        lat = PX.sample_lat(pipe, e, [j], dev)
                    finally:
                        sc.remove()
                    imgs.append(PX.decode(pipe, lat)); lats.append(lat.float().cpu())
                    if first:
                        torch.cuda.synchronize(); s = time.time() - t0
                        proj = s * len(todo) * CHUNK / 3600
                        print(f"  pilot {name}: {s:.1f} s/image -> {proj:.3f} h for {len(todo) * CHUNK} images", flush=True)
                        if not guard(proj, f"gen2 {name}"):
                            raise SystemExit(4)
                        first = False
                atomic_torch_save({"img": torch.cat(imgs), "lat": torch.cat(lats)}, os.path.join(d, f"{i:05d}.pt"))
                print(f"  {name}: prompts {i}-{min(hi, i + CHUNK) - 1} done", flush=True)
            del pipe
            torch.cuda.empty_cache()
            run.record()


@torch.no_grad()
def ir(setname, dev):
    """ImageReward v1.0 per image (imported here so the main environment never loads it)."""
    import ImageReward as RM
    from PIL import Image
    from boa_dit.w4a4 import common as WC
    lo, hi = SETS[setname]
    prompts = WC.load_embeds("mjhq1k")["prompts"][lo:hi]
    with Run(STAGE, f"dist_ir_{setname}", dev, {"prompts": [lo, hi]}) as run:
        m = RM.load("ImageReward-v1.0", device=dev)
        res = {"set": setname, "prompts": [lo, hi], "model": "ImageReward-v1.0", "scores": {}}
        for name, lab in ARMS[setname]:
            imgs, _ = load_imgs(name, lo, hi)
            sc = []
            for k in range(len(imgs)):
                im = Image.fromarray(imgs[k].permute(1, 2, 0).numpy().astype("uint8"))
                sc.append(float(m.score(prompts[k], [im])))
            res["scores"][name] = sc
            print(f"  IR {setname} {lab}: mean {np.mean(sc):+.4f}", flush=True)
        res.update(run.record())
    atomic_write_json(os.path.join(DD, f"ir_{setname}.json"), res)


def boot_diff(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    g = np.random.default_rng(0)
    d = [np.mean(a[i] - b[i]) for i in (g.integers(0, len(a), len(a)) for _ in range(n))]
    return {"diff": float(np.mean(a - b)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def boot_red(a, b, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    g = np.random.default_rng(0)
    d = [1 - a[i].mean() / b[i].mean() for i in (g.integers(0, len(a), len(a)) for _ in range(n))]
    return {"reduction": float(1 - a.mean() / b.mean()), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def fidelity(setname, dev):
    """PSNR / LPIPS-VGG / latent rel. MSE per image vs the bf16 batch-1 references of the set."""
    import lpips
    lo, hi = SETS[setname]
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    ref_name = ARMS[setname][0][0]
    ri, rl = load_imgs(ref_name, lo, hi)
    out = {}
    for name, _ in ARMS[setname][1:]:
        qi, ql = load_imgs(name, lo, hi)
        ps, lp = [], []
        with torch.no_grad():
            for i in range(len(ri)):
                a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
        rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
        out[name] = {"psnr": ps, "lpips": lp, "latent": rel}
    return out


def score(dev):
    res = {}
    for setname in SETS:
        f = os.path.join(DD, f"ir_{setname}.json")
        if not os.path.exists(f):
            continue
        irs = __import__("json").load(open(f))["scores"]
        fid = fidelity(setname, dev)
        base = ARMS[setname][1][0]
        rows = {}
        for name, lab in ARMS[setname]:
            r = {"label": lab, "ir_mean": float(np.mean(irs[name])), "ir_per_image": irs[name]}
            if name in fid:
                r.update({k + "_mean": float(np.mean(v)) for k, v in fid[name].items()})
                r.update({k + "_per_image": v for k, v in fid[name].items()})
            if name != base and name in fid:
                r["vs_single"] = {"ir": boot_diff(irs[name], irs[base]), "psnr": boot_diff(fid[name]["psnr"], fid[base]["psnr"]),
                                  "lpips": boot_diff(fid[name]["lpips"], fid[base]["lpips"]), "latent": boot_red(fid[name]["latent"], fid[base]["latent"])}
            rows[name] = r
        res[setname] = {"prompts": list(SETS[setname]), "arms": rows}
    atomic_write_json(os.path.join(DD, "pixart_scores.json"), res)
    for s, v in res.items():
        print(f"== {s} mjhq1k[{v['prompts'][0]}:{v['prompts'][1]}]")
        for name, r in v["arms"].items():
            vs = r.get("vs_single")
            extra = "" if not vs else (f" | vs single: IR {vs['ir']['diff']:+.4f} [{vs['ir']['ci'][0]:+.4f}, {vs['ir']['ci'][1]:+.4f}], "
                                       f"PSNR {vs['psnr']['diff']:+.2f} [{vs['psnr']['ci'][0]:+.2f}, {vs['psnr']['ci'][1]:+.2f}], "
                                       f"LPIPS {vs['lpips']['diff']:+.4f} [{vs['lpips']['ci'][0]:+.4f}, {vs['lpips']['ci'][1]:+.4f}], "
                                       f"latent {-100 * vs['latent']['reduction']:+.1f}% [{-100 * vs['latent']['ci'][1]:+.1f}, {-100 * vs['latent']['ci'][0]:+.1f}]")
            print(f"  {r['label']:32s} IR {r['ir_mean']:+.4f}" + (f" PSNR {r['psnr_mean']:.2f} LPIPS {r['lpips_mean']:.4f} latent {r['latent_mean']:.5f}" if "psnr_mean" in r else "") + extra)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["gen2", "ir", "score"])
    ap.add_argument("--set", default="screen", choices=list(SETS))
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(DD, exist_ok=True)
    {"gen2": lambda: gen2(dev), "ir": lambda: ir(a.set, dev), "score": lambda: score(dev)}[a.cmd]()


if __name__ == "__main__":
    main()
