# Nunchaku SDXL feasibility on the A30 (nk1): pre-registration

Written 2026-10-30, **before any installation or measurement**. Branch `nunchaku-sdxl-feasibility` (from weight-anchor-controls 83f657c),
results in `results/nunchaku_sdxl/`, code `scripts/nunchaku_sdxl_profile.py`, stage tag `nk1`. Deviations are appended below the marker, dated.

## Question
Does an existing accelerated low-bit implementation give a measured low-bit / FP denoiser call-time ratio r small enough to leave room for
K = 4 FP anchors on this A30?
- The anchored arm (wac1's R) costs 50 low-bit calls + 13 FP calls per 50-step trajectory, so T_R / T_FP ≈ r + 13/50. **Break-even is r = 0.74.**
- This is **installation and profiling only**: no anchor-error measurement, no PixArt port, no kernel work.

## Scope and budget
- **Cap:** 0.5 A30-hour in `gpu_hours.csv` rows tagged `nk1`, including installation smoke tests and failures.
- **Downloads:** public checkpoints from Hugging Face. **Nothing is uploaded.**
  - SDXL base `stabilityai/stable-diffusion-xl-base-1.0` (fp16 variant);
  - `nunchaku-ai/nunchaku-sdxl` (formerly `nunchaku-tech`), `svdq-int4_r32-sdxl.safetensors`.
  - Model files stay in the HF cache and are never committed.
- **Environment:** a separate conda env `nunchaku`, cloned from `py3.10` (torch 2.11.0+cu130, Python 3.10).
  - Nunchaku is installed **only** from the official GitHub release wheel `nunchaku-1.2.1+cu13.0torch2.11-cp310-cp310-linux_x86_64.whl` (v1.2.1, 2026-01-25).
    The PyPI package named `nunchaku` is an unrelated project and is not installed.
  - Pins follow Nunchaku's CI extras: diffusers 0.36, transformers 4.55.2, accelerate 1.9, peft 0.17, huggingface-hub 0.34.
  - If a pin cannot be satisfied, the actual versions are recorded as a deviation.
  - The full `pip freeze` is committed.

## Steps
1. **Official example:** run `examples/v1/sdxl.py` as published (bf16 pipeline, guidance 5.0, 50 steps), plus the same prompt with the FP UNet. Save both images (not committed, sizes recorded) as an installation check only.
2. **Dual residency:**
   - load the FP SDXL UNet (bf16, as in the official example) and the Nunchaku INT4 UNet together on the GPU, with the VAE;
   - precompute the text embeddings once (both text encoders), then delete the encoders before timing;
   - no CPU offloading;
   - record resident bytes and `max_memory_allocated` / `max_memory_reserved`.
3. **Approximation and attention settings:**
   - no caching accelerators (no first-block cache, TeaCache, DeepCache), no LoRA, no torch.compile, no offloading;
   - attention as shipped: Nunchaku's own FA2 processor for the INT4 UNet, diffusers' default `AttnProcessor2_0` (PyTorch SDPA) for the FP UNet;
   - the processors are recorded. If SDPA's flash backend is unavailable for the FP UNet, that is recorded rather than changed.
4. **Per-call latency:**
   - one UNet call = the CFG-batched forward (batch 2: uncond + cond) at 1024² (latent 4×128×128), batch 1 image, bf16;
   - timesteps t ∈ {999, 750, 500, 250, 1} with the same embeddings and added conditions (`text_embeds`, `time_ids` for 1024² / no crop);
   - per (model, t): 5 warm-up calls, then 20 timed calls with `torch.cuda.synchronize()` + CUDA events; median and range reported;
   - r = median INT4 call / median FP call, per t and pooled (median over all timed calls).
5. **Trajectory latency** (context, not the decision): the pipeline's denoising loop (EulerDiscrete, the SDXL default; 50 steps; guidance 5.0;
   precomputed embeddings; `output_type="latent"`, so no decode) for FP and INT4, 3 warm repetitions each.
6. **Kernel verification:**
   - a `torch.profiler` trace of one INT4 UNet call records the CUDA kernel names and the share of CUDA time in Nunchaku's W4A4 GEMM / quantize kernels;
   - count the `SVDQW4A4Linear` modules and confirm they are the layers executing;
   - the same trace for the FP call, for comparison;
   - the share of CUDA time in non-quantized parts (convolutions, norms, attention) is reported, since SDXL keeps its convolutions in bf16.

## Decision (pre-registered)
- **Fit:** both UNets + VAE resident with peak reserved memory ≤ 22 GB (of 24 GB).
- **Ratio** (pooled median r):
  - r ≤ 0.60: **comfortable margin** for K = 4 (anchored ≈ ≤ 0.86 of FP);
  - 0.60 < r < 0.74: **marginal**;
  - r ≥ 0.74: **no margin** for K = 4.
  - The thresholds are my reading of "comfortably below 0.74", fixed here before measurement.
- **Outcome:** if fit and comfortable, report that anchor-error measurement on Nunchaku SDXL is **eligible** as a separately pre-registered follow-up. It is **not run here.**
- **Scope of the measurement:** these are A30 latencies with this software stack. They are not generalized to other GPUs or to FLUX/Sana.
  Nunchaku SDXL is W4A4 (SVDQuant) and **not** the W4A16 bed of wa1/wac1, so the transfer of error reuse is an open question.

## Deliverables
In `results/nunchaku_sdxl/`:
- PROTOCOL.md;
- env.txt (versions, `pip freeze`, wheel sha256, checkpoint revisions);
- profile.json;
- kernels.json (top kernels, W4A4 share);
- REPORT.md.

The GPU ledger is `results/gpu_hours.csv`. Commit and push after each step.

---

## Deviations and results
(appended below, dated)
- 2026-10-30, deviation (environment):
  - `conda create --clone` stopped with CondaToSNonInteractiveError (Anaconda channel Terms of Service not accepted).
  - I did not accept the terms on the user's behalf. The separate environment is a Python venv (`/home/nk_env`, Python 3.10 from py3.10) with torch 2.11.0+cu130 from the PyTorch index.
  - The pins are as registered.
- 2026-10-30, results:
  - **Fit:** peak reserved 8.81 GB.
  - **Pooled r = 0.932** (per t 0.932–0.934); the 50-step trajectory ratio is 0.930.
  - **Kernels:** W4A4 GEMM kernels are executing, at 40.6% of INT4 CUDA time.
  - **Band:** "no margin". The anchor follow-up is not eligible and was not run.
  - **GPU:** 0.050 h.
