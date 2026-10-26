"""PixArt-Σ D2(a): sampler band gain G(t, band) and the corrected base's error spectrum S(t, band).
bf16 FP model, design prompts 0-15 (held-out calibration prompts; seed = prompt index), the model's sampler (DPM-Solver++ 20, g 4.5). A manual copy
of the pipeline's denoising loop; at each probed step t (0, 2, .., 18) the latents and a deep copy of the scheduler
(its multistep history) are snapshotted, so each injection resumes from t instead of resampling from noise.
At step t a random-phase error confined to one of 4 equal-area radial bands of the 128x128 latent DFT (or white) is
added to the guided noise prediction (4 eps channels) with energy = the base's mean ||delta_g,t||^2 per sample
(cfg_branch.json, corrected base); FP continues. G = ||x_final_inj - x_final_FP||^2 / injected energy, mean over prompts.
S(t, band) from the corrected base's guided error on the design cache (prompts 0-15).
Run: python -m boa_dit.w4a4.band_gain --arm hsvd_r4_b0.5_kv16_u2_rs --device cuda:1
"""
import argparse
import copy
import json
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.w4a4.common import CK, GUIDANCE, STEPS, embeds_for, generators, load_embeds, load_pipe
from boa_dit.w4a4.screen import install, text_for

PROBE = tuple(range(0, 20, 2))
PROMPTS = list(range(16))


def band_masks(n=128):
    f = torch.fft.fftfreq(n)
    r = torch.sqrt(f[:, None] ** 2 + f[None, :] ** 2)
    q = torch.quantile(r.flatten(), torch.tensor([0.25, 0.5, 0.75]))
    edges = [-1.0] + q.tolist() + [10.0]
    return [((r > edges[i]) & (r <= edges[i + 1])) for i in range(4)]


@torch.no_grad()
def guided_eps(pipe, latents, t, kw):
    tr = pipe.transformer
    lat_in = torch.cat([latents] * 2)
    lat_in = pipe.scheduler.scale_model_input(lat_in, t)
    ts = t.expand(lat_in.shape[0]) if torch.is_tensor(t) else torch.full((lat_in.shape[0],), t, device=latents.device)
    o = tr(lat_in, encoder_hidden_states=kw["pe"], encoder_attention_mask=kw["pm"], timestep=ts,
           added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0]
    u, c = o.chunk(2)
    g = u + GUIDANCE * (c - u)
    return g.chunk(2, dim=1)[0]                                               # learned sigma: keep the eps half


@torch.no_grad()
def run_from(pipe, sched, latents, k0, timesteps, kw, inject=None):
    for k in range(k0, len(timesteps)):
        t = timesteps[k]
        eps = guided_eps(pipe, latents, t, kw)
        if inject is not None and k == k0:
            eps = eps + inject.to(eps.dtype)
        latents = sched.step(eps, t, latents, return_dict=False)[0]
    return latents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hsvd_r4_b0.5_kv16_u2_rs")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    cb = json.load(open(os.path.join(RESULTS, "w4a4", "cfg_branch.json")))
    assert cb["arm"] == a.arm, "run cfg_branch on the same arm first"
    energy = {r["step"]: r["g"] / 32 for r in cb["per_step"]}                 # per sample (32 prompts in cfg_branch)
    masks = band_masks()
    res = {"prompt_set": "design (held-out calibration prompts 32-63, rule_addendum_design.md)", "arm": a.arm, "probe_steps": list(PROBE), "bands": "4 equal-area radial bands of the 128x128 DFT", "G": {}, "S": {}}
    with Run("w4a4", f"band_gain_{a.arm}", dev, vars(a)) as run, torch.no_grad():
        pipe = load_pipe(dev)
        e = load_embeds("design")
        emb = embeds_for(e, PROMPTS, dev)
        kw = {"pe": torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]]).to(torch.bfloat16),
              "pm": torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])}
        pipe.scheduler.set_timesteps(STEPS, device=dev)
        timesteps = pipe.scheduler.timesteps
        g = generators(PROMPTS, dev)
        latents = torch.stack([torch.randn(4, 128, 128, generator=gi, device=dev) for gi in g]).to(torch.bfloat16)
        latents = latents * pipe.scheduler.init_noise_sigma
        snaps, x = {}, latents
        sched = copy.deepcopy(pipe.scheduler)
        for k, t in enumerate(timesteps):
            if k in PROBE:
                snaps[k] = (x.clone(), copy.deepcopy(sched))
            x = sched.step(guided_eps(pipe, x, t, kw), t, x, return_dict=False)[0]
        x_ref = x.float()
        k0 = PROBE[-1]
        chk = run_from(pipe, copy.deepcopy(snaps[k0][1]), snaps[k0][0], k0, timesteps, kw)
        res["continuation_bitexact"] = bool(torch.equal(chk.float(), x_ref))
        for k in PROBE:
            row = {}
            for bi, m in enumerate(masks + [None]):
                gen = torch.Generator().manual_seed(1000 * k + bi)
                z = torch.randn(len(PROMPTS), 4, 128, 128, generator=gen)
                if m is not None:
                    z = torch.fft.ifft2(torch.fft.fft2(z) * m).real
                z = z * torch.sqrt(torch.tensor(energy[k]) / z.pow(2).flatten(1).sum(1)).view(-1, 1, 1, 1)
                xk, sk = snaps[k]
                xf = run_from(pipe, copy.deepcopy(sk), xk.clone(), k, timesteps, kw, inject=z.to(dev)).float()
                row["white" if m is None else f"band{bi + 1}"] = ((xf - x_ref).flatten(1).pow(2).sum(1) / energy[k]).tolist()
            res["G"][k] = row
            print(f"step {k}: G " + " ".join(f"{kk}={np.mean(v):.3g}" for kk, v in row.items()), flush=True)
        # S: the arm's guided error spectrum on the design cache, prompts 0-15
        pipe2 = load_pipe(dev)
        install(pipe2.transformer, a.arm)
        c = torch.load(os.path.join(CK, "design_cache.pt"), map_location="cpu")
        for s in PROBE:
            idx = {(int(c["prompt"][j]), int(c["branch"][j])): j for j in range(len(c["t"])) if int(c["step"][j]) == s and int(c["prompt"][j]) < 16}
            rows = [idx[(p, b)] for p in PROMPTS for b in (0, 1)]
            pe, pm = text_for(e, c["prompt"][rows], c["branch"][rows], dev)
            o = pipe2.transformer(c["latent"][rows].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][rows].to(dev),
                                  added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0].float()[:, :4]
            d = (o - c["eps"][rows].to(dev).float()[:, :4]).double()
            dg = GUIDANCE * d[1::2] + (1 - GUIDANCE) * d[0::2]
            P = torch.fft.fft2(dg).abs().pow(2).sum((0, 1))
            res["S"][s] = [float(P[m.to(dev)].sum() / P.sum()) for m in masks]
        res["ratio_b1_b4"] = {k: float(np.mean(res["G"][k]["band1"]) / np.mean(res["G"][k]["band4"])) for k in PROBE}
        res.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "w4a4", "band_gain.json"), res)
    print("bitexact continuation:", res["continuation_bitexact"], "b1/b4:", {k: round(v, 2) for k, v in res["ratio_b1_b4"].items()}, flush=True)


if __name__ == "__main__":
    main()
