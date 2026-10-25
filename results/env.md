# Environment (Stage 0)

- Host: JarvisLabs, 2x NVIDIA A30 24 GB, driver 580.126.20 (nvidia-smi CUDA 13.0), 32 vCPU, 503 GB RAM
- Python 3.10.20; torch 2.11.0+cu130 (CUDA 13.0); torchvision 0.26.0; timm 1.0.30; diffusers 0.38.0;
  accelerate 1.15.0; scipy 1.15.3; scikit-learn 1.7.2; pytorch-fid 0.3.0; numpy 2.2.6 — pinned in `requirements.txt`.
  All pip installs succeeded; nothing blocked.
- Submodules: `external/DiT` = facebookresearch/DiT @ ed81ce2229091fd4ecc9a223645f95cf379d582b;
  `external/boa` = PseudoHunt/boa-kronecker-gap @ 91813d573289841db96fd98bdea5ea256dc32abd.
- Checkpoint: `DiT-XL-2-256x256.pt` from dl.fbaipublicfiles.com/DiT/models (same file `download.py` fetches; loaded
  directly instead of via `find_model` to avoid the `pretrained_models/` cwd assumption). VAE: `stabilityai/sd-vae-ft-ema`.
  FID ref: `VIRTUAL_imagenet256_labeled.npz` (10,000 images); Inception stats cached in `results/stage0/ref_stats.npz`.

## Numerics
- Model weights fp32; forward under `torch.autocast(cuda, float16)` (pure `.half()` breaks: DiT's timestep
  embedder feeds an fp32 frequency vector into its MLP). Sampler arithmetic and latents fp32.
- TF32 disabled for matmul and cuDNN.
- **Determinism:** two runs of seeds 0-7 match bit-for-bit (`results/stage0/sample_run.json`).
- **Batch-layout dependence:** the same seed sampled at batch 8 vs batch 16 differs (max |Δ| 0.057, relative latent
  MSE 4.2e-6 mean / 7.1e-6 max) — fp16 GEMM kernel selection depends on shape. This is the FP noise floor; every
  matched-seed comparison in this repo uses a fixed batch layout (bs = 32 images, seeds in order).
- **Speed:** 0.50 s/image at batch 32 (both CFG branches, 50 DDIM steps, one A30) — slower than the handoff's
  0.2-0.4 s estimate. Consequence: 512-seed latent eval ≈ 4.3 min, FID-2k ≈ 17 min, FID-5k ≈ 42 min per run.
- **Resume test:** a 4-batch sampling run was SIGKILLed after batch 2 and resumed; only batches 3-4 ran and the result
  is bit-identical to an uninterrupted run.
