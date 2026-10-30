"""nk1: Nunchaku SDXL feasibility on the A30 (results/nunchaku_sdxl/PROTOCOL.md). Installation + profiling only.
Runs in the separate venv /home/nk_env (nunchaku 1.2.1 wheel, diffusers 0.36, transformers 4.55.2, torch 2.11.0+cu130).
Stages: example | profile
Run: /home/nk_env/bin/python scripts/nunchaku_sdxl_profile.py <stage>
"""
import csv
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "nunchaku_sdxl")
IMG = "/home/boa-dit/ckpt/nunchaku_sdxl"                                                 # images: not committed
SDXL = "stabilityai/stable-diffusion-xl-base-1.0"
NK_REPO, NK_FILE = "nunchaku-ai/nunchaku-sdxl", "svdq-int4_r32-sdxl.safetensors"
PROMPT = "A cinematic shot of a baby racoon wearing an intricate italian priest robe."
STAGE, DT, DEV = "nk1", torch.bfloat16, "cuda"
TS = (999, 750, 500, 250, 1)
NWARM, NTIME, NTRAJ = 5, 20, 3


def utc():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Ledger:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0, self.start = time.time(), utc()
        return self

    def __exit__(self, *exc):
        with open(os.path.join(ROOT, "results", "gpu_hours.csv"), "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            csv.writer(f).writerow([STAGE, self.name + ("" if exc[0] is None else "(FAILED)"), "cuda:0", self.start, utc(), f"{(time.time() - self.t0) / 3600:.4f}"])
        return False


def jdump(name, obj):
    with open(os.path.join(OUT, name + ".tmp"), "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(os.path.join(OUT, name + ".tmp"), os.path.join(OUT, name))


def nk_path():
    from huggingface_hub import hf_hub_download
    return hf_hub_download(NK_REPO, NK_FILE)


def example():
    """official examples/v1/sdxl.py as published (hub path as written), plus the same prompt with the FP UNet."""
    from diffusers import StableDiffusionXLPipeline
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
    os.makedirs(IMG, exist_ok=True)
    out = {"utc": utc()}
    with Ledger("example"):
        try:
            unet = NunchakuSDXLUNet2DConditionModel.from_pretrained("nunchaku-tech/nunchaku-sdxl/svdq-int4_r32-sdxl.safetensors")
            out["hub_path_as_published"] = "ok"
        except Exception as e:                                                       # renamed org: record and use the current repo id
            out["hub_path_as_published"] = f"failed: {type(e).__name__}: {str(e)[:200]}"
            unet = NunchakuSDXLUNet2DConditionModel.from_pretrained(nk_path())
        pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, unet=unet, torch_dtype=DT, use_safetensors=True, variant="fp16").to(DEV)
        g = torch.Generator(DEV).manual_seed(0)
        t0 = time.time()
        im = pipe(prompt=PROMPT, guidance_scale=5.0, num_inference_steps=50, generator=g).images[0]
        out["int4_pipeline_s_cold"] = time.time() - t0
        im.save(os.path.join(IMG, "sdxl_int4.png"))
        del pipe, unet
        torch.cuda.empty_cache()
        pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, torch_dtype=DT, use_safetensors=True, variant="fp16").to(DEV)
        g = torch.Generator(DEV).manual_seed(0)
        t0 = time.time()
        im2 = pipe(prompt=PROMPT, guidance_scale=5.0, num_inference_steps=50, generator=g).images[0]
        out["fp_pipeline_s_cold"] = time.time() - t0
        im2.save(os.path.join(IMG, "sdxl_fp.png"))
        a, b = np.asarray(im).astype(np.float64), np.asarray(im2).astype(np.float64)
        out["psnr_int4_vs_fp_same_seed"] = float(10 * np.log10(255 ** 2 / np.mean((a - b) ** 2)))
        out["images"] = {k: {"path": os.path.join(IMG, f), "bytes": os.path.getsize(os.path.join(IMG, f))} for k, f in (("int4", "sdxl_int4.png"), ("fp", "sdxl_fp.png"))}
        out["note"] = "installation check only (cold, single run, includes text encoding and VAE decode); not a timing result"
    jdump("example.json", out)
    print(json.dumps(out, indent=1))


def bytes_of(m):
    return int(sum(p.numel() * p.element_size() for p in m.parameters()) + sum(b.numel() * b.element_size() for b in m.buffers()))


@torch.no_grad()
def profile():
    from diffusers import AutoencoderKL, EulerDiscreteScheduler, StableDiffusionXLPipeline, UNet2DConditionModel
    from nunchaku.models.linear import SVDQW4A4Linear
    from nunchaku.models.unets.unet_sdxl import NunchakuSDXLUNet2DConditionModel
    torch.backends.cuda.matmul.allow_tf32 = False
    res = {"utc": utc(), "device": torch.cuda.get_device_name(0), "dtype": "bfloat16", "resolution": 1024, "batch_images": 1, "cfg_batch": 2, "guidance": 5.0}
    with Ledger("profile"):
        # text embeddings once, then the encoders are deleted
        pipe = StableDiffusionXLPipeline.from_pretrained(SDXL, unet=None, vae=None, torch_dtype=DT, use_safetensors=True, variant="fp16").to(DEV)
        pe, npe, pp, npp = pipe.encode_prompt(PROMPT, device=DEV, num_images_per_prompt=1, do_classifier_free_guidance=True)
        sched_cfg = pipe.scheduler.config
        del pipe
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc = torch.cat([npe, pe]).to(DT)
        added = {"text_embeds": torch.cat([npp, pp]).to(DT), "time_ids": torch.tensor([[1024, 1024, 0, 0, 1024, 1024]] * 2, device=DEV, dtype=DT)}
        fp = UNet2DConditionModel.from_pretrained(SDXL, subfolder="unet", torch_dtype=DT, use_safetensors=True, variant="fp16").to(DEV).eval()
        nk = NunchakuSDXLUNet2DConditionModel.from_pretrained(nk_path(), torch_dtype=DT).to(DEV).eval()
        vae = AutoencoderKL.from_pretrained(SDXL, subfolder="vae", torch_dtype=DT, use_safetensors=True, variant="fp16").to(DEV)
        torch.cuda.synchronize()
        res["resident_bytes"] = {"fp_unet": bytes_of(fp), "int4_unet": bytes_of(nk), "vae": bytes_of(vae)}
        res["mem_after_load"] = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved()}
        nl = [m for m in nk.modules() if isinstance(m, SVDQW4A4Linear)]
        res["int4_modules"] = {"SVDQW4A4Linear_count": len(nl), "other_linear_count": sum(isinstance(m, torch.nn.Linear) for m in nk.modules()),
                               "conv2d_count": sum(isinstance(m, torch.nn.Conv2d) for m in nk.modules()),
                               "param_dtypes_first_module": {n: str(p.dtype) for n, p in list(nl[0].named_parameters()) + list(nl[0].named_buffers())} if nl else {}}
        res["attn_processors"] = {"fp": sorted({type(p).__name__ for p in fp.attn_processors.values()}),
                                  "int4": sorted({type(m.processor).__name__ for m in nk.modules() if hasattr(m, "processor") and m.processor is not None})}
        g = torch.Generator(DEV).manual_seed(0)
        lat = torch.randn(1, 4, 128, 128, generator=g, device=DEV, dtype=DT)

        def call(m, t, x=lat):
            xin = torch.cat([x, x])
            return m(xin, t, encoder_hidden_states=enc, added_cond_kwargs=added, return_dict=False)[0]

        # numerical sanity: INT4 vs FP output at one t (not a quality metric)
        o_f, o_n = call(fp, torch.tensor(500, device=DEV)).float(), call(nk, torch.tensor(500, device=DEV)).float()
        res["sanity_rel_err_t500"] = float((o_n - o_f).pow(2).sum() / o_f.pow(2).sum())
        # per-call latency
        per = {"fp": {}, "int4": {}}
        allv = {"fp": [], "int4": []}
        for name, m in (("fp", fp), ("int4", nk)):
            for t in TS:
                tt = torch.tensor(t, device=DEV)
                for _ in range(NWARM):
                    call(m, tt)
                torch.cuda.synchronize()
                ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(NTIME)]
                for a, b in ev:
                    a.record(); call(m, tt); b.record()
                torch.cuda.synchronize()
                v = [a.elapsed_time(b) for a, b in ev]
                per[name][str(t)] = {"ms_median": float(np.median(v)), "ms_min": float(np.min(v)), "ms_max": float(np.max(v))}
                allv[name] += v
                print(f"  {name} t={t}: {np.median(v):.1f} ms [{np.min(v):.1f}, {np.max(v):.1f}]", flush=True)
        res["per_call_ms"] = per
        res["r_per_t"] = {str(t): per["int4"][str(t)]["ms_median"] / per["fp"][str(t)]["ms_median"] for t in TS}
        res["r_pooled"] = float(np.median(allv["int4"]) / np.median(allv["fp"]))
        res["per_call_peak"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        # trajectory latency: EulerDiscrete 50 steps, CFG 5.0, latent output (no decode)
        sch = EulerDiscreteScheduler.from_config(sched_cfg)

        def traj(m):
            sch.set_timesteps(50, device=DEV)
            x = torch.randn(1, 4, 128, 128, generator=torch.Generator(DEV).manual_seed(1), device=DEV, dtype=torch.float32) * sch.init_noise_sigma
            for t in sch.timesteps:
                xin = sch.scale_model_input(torch.cat([x, x]), t).to(DT)
                o = m(xin, t, encoder_hidden_states=enc, added_cond_kwargs=added, return_dict=False)[0].float()
                u, c = o.chunk(2)
                x = sch.step(u + 5.0 * (c - u), t, x).prev_sample
            return x
        tr = {}
        for name, m in (("fp", fp), ("int4", nk)):
            traj(m)
            w = []
            for _ in range(NTRAJ):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                xl = traj(m)
                torch.cuda.synchronize(); w.append(time.perf_counter() - t0)
            tr[name] = {"s_median": float(np.median(w)), "s_min": float(np.min(w)), "s_max": float(np.max(w)), "finite": bool(torch.isfinite(xl).all())}
            print(f"  trajectory {name}: {np.median(w):.2f} s", flush=True)
        res["trajectory_50_euler_s"] = tr
        res["trajectory_ratio"] = tr["int4"]["s_median"] / tr["fp"]["s_median"]
        res["peak"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        # kernel verification
        from torch.profiler import ProfilerActivity, profile as tprof
        kern = {}
        for name, m in (("fp", fp), ("int4", nk)):
            with tprof(activities=[ProfilerActivity.CUDA]) as prof:
                call(m, torch.tensor(500, device=DEV)); torch.cuda.synchronize()
            ev = [e for e in prof.key_averages() if e.device_time_total > 0]
            tot = sum(e.device_time_total for e in ev)
            rows = sorted(({"name": e.key[:160], "cuda_us": e.device_time_total, "count": e.count} for e in ev), key=lambda r: -r["cuda_us"])
            cls = {"w4a4_gemm_or_quant": r"w4a4|int4|quantize|svdq|nunchaku|gemm_w4|lora", "conv": r"conv|implicit_gemm|cudnn|winograd|sm80_xmma_fprop|nhwc", "attention": r"flash|fmha|attention|sdpa|softmax",
                   "norm": r"norm"}
            share = {k: 0.0 for k in cls}
            share["other"] = 0.0
            for r in rows:
                n = r["name"].lower()
                hit = next((k for k, p in cls.items() if re.search(p, n)), "other")
                share[hit] += r["cuda_us"] / tot
            kern[name] = {"total_cuda_ms": tot / 1000, "share": share, "top30": rows[:30]}
            print(f"  kernels {name}: total {tot / 1000:.1f} ms; shares {({k: round(v, 3) for k, v in share.items()})}", flush=True)
        jdump("kernels.json", kern)
        res["kernel_share"] = {k: v["share"] for k, v in kern.items()}
        fit = res["peak"]["reserved"] <= 22e9
        r = res["r_pooled"]
        res["decision"] = {"fit_le_22GB_reserved": bool(fit), "r_pooled": r,
                           "band": "comfortable (r <= 0.60)" if r <= 0.60 else ("marginal (0.60 < r < 0.74)" if r < 0.74 else "no margin (r >= 0.74)"),
                           "anchored_K4_estimate_over_fp": r + 13 / 50,
                           "eligible_for_anchor_follow_up": bool(fit and r <= 0.60)}
    jdump("profile.json", res)
    print(json.dumps(res["decision"], indent=1), flush=True)


def env():
    fr = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout
    import diffusers, transformers, importlib.metadata as md
    from huggingface_hub import HfApi
    api = HfApi()
    txt = [f"utc {utc()}", f"python {sys.version.split()[0]} (venv {sys.prefix})", f"torch {torch.__version__}, cuda {torch.version.cuda}", f"diffusers {diffusers.__version__}",
           f"transformers {transformers.__version__}", f"nunchaku {md.version('nunchaku')}",
           "wheel nunchaku-1.2.1+cu13.0torch2.11-cp310-cp310-linux_x86_64.whl sha256 82026d553f44390e416bb632198ba35787d8453821c0dd268f124eee722ef5f3 (matches the GitHub release digest)",
           f"{SDXL} revision {api.model_info(SDXL).sha}", f"{NK_REPO} revision {api.model_info(NK_REPO).sha}",
           f"driver {subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()}", "", "pip freeze:", fr]
    with open(os.path.join(OUT, "env.txt"), "w") as f:
        f.write("\n".join(txt))
    print("\n".join(txt[:10]))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"example": example, "profile": profile, "env": env}[sys.argv[1]]()
