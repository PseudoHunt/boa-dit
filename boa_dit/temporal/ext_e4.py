"""Extension E4 (rule_extension.md, conditional): step count within PixArt-Σ. Single-table W16A8 static (stage-3 tables)
baseline and full feedback at 10 and 50 DPM-Solver++ steps (guidance 4.5), mjhq1k[:16], batch 1, seed = prompt index,
vs bf16 references at the same step count; the 20-step stage-3 pair is reported alongside where compatible.
The single table is step-count independent (one (Δ, z) for all steps), so its buffers are extended to any length.
Denoiser evaluations are counted (transformer calls). Run: python -m boa_dit.temporal.ext_e4
"""
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.temporal import pixart as PX
from boa_dit.temporal.common import OUT, STAGE, boot_reduction, ci_split
from boa_dit.w4a4 import common as WC

EXT = os.path.join(OUT, "ext")
N = 16


def extend(mods, T):
    for m in mods:
        for b in ("D", "Z", "O"):
            v = getattr(m, b)
            setattr(m, b, torch.cat([v, v[:1].expand(max(0, T - len(v)))]) if T > len(v) else v)


@torch.no_grad()
def sample_steps(pipe, steps, dev, name):
    d = PX.img_dir(name)
    os.makedirs(d, exist_ok=True)
    e = WC.load_embeds("mjhq1k")
    f = os.path.join(d, "all.pt")
    if os.path.exists(f):
        return torch.load(f)
    lats, imgs, calls = [], [], []
    for j in range(N):
        sc = PX.StepCounter(pipe.transformer)
        dt = pipe.transformer.dtype
        kw = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in WC.embeds_for(e, [j], dev).items()}
        lat = pipe(prompt=None, negative_prompt=None, num_inference_steps=steps, guidance_scale=WC.GUIDANCE,
                   generator=WC.generators([j], dev), output_type="latent", **kw).images
        calls.append(PX.STATE.step + 1)
        sc.remove()
        lats.append(lat.float().cpu()); imgs.append(PX.decode(pipe, lat))
    out = {"lat": torch.cat(lats), "img": torch.cat(imgs), "denoiser_calls": calls}
    atomic_torch_save(out, f)
    return out


@torch.no_grad()
def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(EXT, exist_ok=True)
    res = {"n": N, "prompts": "mjhq1k[:16]", "steps": {}}
    with Run(STAGE, "ext_e4_stepcount", dev, {"steps": [10, 50], "n": N}) as run:
        for steps in (10, 50):
            ref = sample_steps(WC.load_pipe(dev), steps, dev, f"e4_bf16_s{steps}")
            arms = {}
            for fb in (False, True):
                pipe = WC.load_pipe(dev)
                mods = PX.install_fb(pipe.transformer, "single8", "none", fb)
                extend(mods, steps)
                PX.STATE.mode = "static"
                arms["feedback" if fb else "single8"] = sample_steps(pipe, steps, dev, f"e4_{'fb' if fb else 'single8'}_s{steps}")
                PX.STATE.mode = "fp"
                del pipe
                torch.cuda.empty_cache()
            rel = {k: ((v["lat"] - ref["lat"]).flatten(1).pow(2).sum(1) / ref["lat"].flatten(1).pow(2).sum(1)).numpy() for k, v in arms.items()}
            psnr = {k: [float(10 * torch.log10(255.0 ** 2 / (ref["img"][i].float() - v["img"][i].float()).pow(2).mean().clamp_min(1e-10)))
                        for i in range(N)] for k, v in arms.items()}
            r = boot_reduction(rel["feedback"], rel["single8"])
            res["steps"][steps] = {"latent_rel_mse": {k: float(v.mean()) for k, v in rel.items()}, "psnr": {k: float(np.mean(v)) for k, v in psnr.items()},
                                   "feedback_vs_single": r, "denoiser_calls_per_image": arms["single8"]["denoiser_calls"][0],
                                   "latent_rel_mse_per_image": {k: v.tolist() for k, v in rel.items()}}
            print(f"  steps {steps}: single {rel['single8'].mean():.5f}, feedback {rel['feedback'].mean():.5f}, "
                  f"feedback vs single {-100 * r['reduction']:+.1f}% [{-100 * r['ci'][1]:+.1f}, {-100 * r['ci'][0]:+.1f}], "
                  f"calls/image {arms['single8']['denoiser_calls'][0]}", flush=True)
            atomic_write_json(os.path.join(EXT, "e4_stepcount.json"), {**res, "partial": True})
        res.update(run.record())
    atomic_write_json(os.path.join(EXT, "e4_stepcount.json"), res)


if __name__ == "__main__":
    main()
