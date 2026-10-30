# Nunchaku SDXL feasibility on the A30 (nk1): report

Protocol: `PROTOCOL.md` (pre-registered 2026-10-30, before installation). Raw numbers: `profile.json`, `kernels.json`, `example.json`; environment: `env.txt`.
**GPU: 0.050 A30-h** (example 0.0126 + profile 0.0371). Installation and profiling only; no anchor-error measurement was run.

## Setup
- **Software:**
  - Nunchaku **1.2.1**, the official GitHub release wheel `cu13.0torch2.11-cp310`; its sha256 matches the release digest.
  - A separate venv `/home/nk_env`: torch 2.11.0+cu130, diffusers 0.36.0, transformers 4.55.2 (Nunchaku's CI pins).
  - The PyPI package called `nunchaku` is unrelated and was not installed.
- **Checkpoints:**
  - SDXL base 1.0 (fp16 variant, revision 46216598…), run in bf16 as in the official example;
  - `nunchaku-ai/nunchaku-sdxl` `svdq-int4_r32-sdxl.safetensors` (revision 3d7636b1…; the old `nunchaku-tech/…` hub path still resolves).
- **INT4 UNet:**
  - **420 `SVDQW4A4Linear`** modules: packed int8-stored INT4 `qweight`, bf16 scales, rank-32 bf16 low-rank branch. This is **W4A4**, not the W4A16 of wa1/wac1.
  - 183 other linears and **all 51 convolutions stay bf16**.
  - Attention: Nunchaku's FA2 processor; the FP UNet uses diffusers' `AttnProcessor2_0` (PyTorch SDPA, which ran the flash kernel).
- **Disabled:** caching, LoRA, `torch.compile` and offloading. TF32 matmul is off.
- **Official example:** ran as published (bf16, guidance 5.0, 50 steps). The INT4 and FP images for the same seed are coherent and similar, at 23.1 dB PSNR from each other.
  This is an installation check only.

## Measurements (batch 1 image, 1024², CFG batch 2, bf16, warm)
| quantity | FP (bf16) UNet | INT4 Nunchaku UNet | ratio INT4/FP |
|---|---:|---:|---:|
| one UNet call (CFG-batched), median of 20 per t, CUDA events | 199.1–200.0 ms | 186.0–186.7 ms | **0.932 pooled** (0.932–0.934 per t; t ∈ {999, 750, 500, 250, 1}) |
| 50-step Euler trajectory, latent output (no decode), median of 3 | 10.04 s | 9.34 s | 0.930 |
| resident model bytes | 5.13 GB | 2.56 GB | — |
| CUDA time in the profiled call | 197.3 ms | 182.8 ms | — |

- **Memory:** both UNets + VAE (0.17 GB) were resident at once with no offloading. Peak allocated 8.34 GB, peak reserved **8.81 GB** of 24 GB.
- **Kernels:** the packed low-bit kernels **are executing**.
  - `nunchaku::kernels::GEMM_W4A4<…>` accounts for 840 launches and 74 ms, **40.6% of the INT4 call's CUDA time**.
  - The FP call has no such kernels; its linear layers run as `ampere_bf16_s16816gemm` (≈ 92 ms of its top-30 kernels).
- **Where the time goes:**
  - Replacing the quantized linears cut linear-layer GEMM time only from ≈ 92 ms to ≈ 80 ms: 74 ms of W4A4 kernels plus the remaining bf16 linears.
  - Convolutions (≈ 20%), attention (≈ 11%), norms and elementwise kernels are unchanged at about 100 ms per call.
  - So even infinitely fast GEMMs would leave r ≈ 0.5 on this UNet.

## Pre-registered decision
- **Fit:** yes (peak reserved 8.8 GB ≤ 22 GB).
- **Ratio:** r = **0.932 → "no margin" (r ≥ 0.74)**. An anchored K = 4 sampler would cost about r + 13/50 ≈ **1.19× FP-50**.
- **Anchor-error measurement on Nunchaku SDXL: not eligible.** It was not run.

## Reading, kept narrow
- **On this A30 with this stack**, Nunchaku's SDXL W4A4 UNet is only **≈ 7% faster per denoiser call** than the bf16 UNet at 1024², batch 1.
  - The W4A4 GEMMs themselves are only modestly faster than bf16 tensor-core GEMMs at these shapes.
  - A large share of SDXL's time sits in unquantized convolutions, attention and elementwise kernels.
  - That falls far short of the ≈ 3× FLUX figure published for an RTX 4090.
- **No speed margin for K = 4 anchors here.** The existing accelerated implementation does not supply the margin the anchoring construction would need on this GPU and model.
- **Untested, not ruled out:** other GPUs (e.g. Ada/Blackwell with better low-bit throughput), larger batch sizes, FLUX or Sana (transformer-only, so less unquantized conv time), and newer Nunchaku versions.
  None of these were measured, and nothing here generalizes to them.
- **Error reuse:** this result says nothing about whether W4A4 error is reusable. That question was not reached.

## Reproduce
```
/root/miniconda3/envs/py3.10/bin/python -m venv /home/nk_env
/home/nk_env/bin/pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu130
gh release download v1.2.1 -R nunchaku-tech/nunchaku -p 'nunchaku-1.2.1+cu13.0torch2.11-cp310-cp310-linux_x86_64.whl'
/home/nk_env/bin/pip install ./nunchaku-1.2.1+cu13.0torch2.11-cp310-cp310-linux_x86_64.whl diffusers==0.36.0 transformers==4.55.2 accelerate==1.9.0 peft==0.17.0 huggingface-hub==0.34.4
/home/nk_env/bin/python scripts/nunchaku_sdxl_profile.py env|example|profile
```
The conda clone in the protocol was replaced by a venv, because `conda create` required accepting Anaconda's channel Terms of Service.
The images are in `ckpt/nunchaku_sdxl/` (not committed). No model or image was uploaded anywhere.
