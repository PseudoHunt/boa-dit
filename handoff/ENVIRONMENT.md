# Environment (recorded on the old VM, 2026-10-29)

## Hardware and system
- GPU: 1× NVIDIA A30, 24 GiB (24576 MiB). Driver 580.126.20. `nvcc` 12.6 present; the PyTorch wheel bundles its own CUDA 13.0 runtime.
- CPU: 16 cores; RAM: 503 GiB (the PixArt calibration cache and the FID bootstraps used tens of GB of RAM; 64 GiB is enough for the DiT pilot).
- OS: Ubuntu 22.04.5 LTS. Disk: the old data root used 85 GB, of which 58 GB are regenerable caches. A fresh VM needs about **60 GB** for the
  DiT pilot (upstream 4.7 GB + DiT archive groups ≈ 7.7 GB + new checkpoints/latents ≈ 20 GB + headroom), and about **150 GB** to restore everything, PixArt included.
- **VRAM needed by the DiT pilot:** E3's rank-8 carry rollout training (K = 5, B = 4) peaked at **9.6 GiB** on the A30. With static A8 fake-quant and DCT-4 state
  the pilot needs more. Budget ≥ 16 GiB and measure in the pilot's profile step (results/dit_joint/PROTOCOL.md). PixArt work peaked at 6.1 GiB (B = 1).

## Python
- Python 3.10.20 (miniconda env `py3.10` on the old VM; on Ubuntu 22.04, system `python3` = 3.10 works).
- **Main environment:** `requirements-lock.txt` (pip freeze). Key versions:
  - torch 2.11.0+cu130, torchvision 0.26.0;
  - diffusers 0.38.0, transformers 5.7.0, huggingface_hub 1.33.0, accelerate 1.15.0, safetensors 0.8.0rc0, sentencepiece 0.2.2;
  - numpy 2.2.6, scipy 1.15.3, timm 0.6.13;
  - pytorch-fid 0.3.0, lpips 0.1.4 (VGG), pypdf 6.21.3.
  - Install torch from the cu130 index: `pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130`, then `pip install -r requirements-lock.txt`.
- **ImageReward environment** (PixArt ImageReward only; not needed by the DiT pilot): a venv **with system site packages** over the main env, plus
  `requirements-imagereward-overrides.txt`: transformers 4.44.2, fairscale 0.4.13, and the image-reward 1.5 wheel committed at
  `handoff/wheels/image_reward-1.5-py3-none-any.whl` (42 KB, built from the public ImageReward 1.5 source). The scripts reference it as `$IRPY`.
- xformers: **not installed**. Attention runs through PyTorch SDPA (flash and mem-efficient backends enabled by default).

## Numerical settings (as used in every reported number)
- DiT sampling (`boa_dit/sample.py::ddim_sample`): fp16 autocast forward and fp32 state; DDIM-50 (guided-diffusion respacing `ddim50`), eta 0, cfg 1.5.
  CFG is applied to eps channels 0–2 only; channel 3 and the variance channels come from the cond half (the official DiT `forward_with_cfg`).
- TF32: `torch.backends.cuda.matmul.allow_tf32 = False` (PyTorch default); cuDNN TF32 = True (default). Nothing in the code changes them.
- Batch sizes: DiT sampling 32 per batch, and results are **batch-size dependent at the bit level** (fp16 kernels): 4-image and 32-image batches differ
  at about 1e-3, 32-image batches are bitwise reproducible. Every paired comparison uses aligned batches of 32. Training B = 4 (DiT carry), B = 1 (PixArt).
- PixArt: bf16 pipeline, DPM-Solver++(2M) 20 steps, guidance 4.5, batch 1, seed = prompt index. W4A4 fake-quant in fp32, branch in bf16.
- FID: pytorch-fid InceptionV3 pool3 (weights `pt_inception-2015-12-05-6726825d.pth`) vs `results/stage0/ref_stats.npz`
  (ADM VIRTUAL_imagenet256 batch, 10,000 images; sha256 a11af8b0f80b2d77640c8dcef76674eebcff02c065a11bf4261e95cef635fab4).
- LPIPS: `lpips.LPIPS(net="vgg")` (torchvision vgg16-397923af.pth).

## Submodules
- `external/DiT` → https://github.com/facebookresearch/DiT.git @ ed81ce2229091fd4ecc9a223645f95cf379d582b
- `external/boa` → https://github.com/PseudoHunt/boa-kronecker-gap.git @ 91813d573289841db96fd98bdea5ea256dc32abd (the BoA quantizer used by the heig builder)

## Upstream models (pinned; not re-hosted; restore with `scripts/restore_assets.py --upstream`)
| asset | source | revision / sha256 |
|---|---|---|
| DiT-XL-2-256x256.pt | https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt | sha256 in SHA256SUMS |
| VIRTUAL_imagenet256_labeled.npz | https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz | sha256 in SHA256SUMS (only needed to recompute ref_stats, which is archived) |
| VAE (DiT) | HF `stabilityai/sd-vae-ft-ema` | revision f04b2c4b98319346dad8c65879f680b1997b204a |
| PixArt-Σ 1024 (transformer, VAE, T5 text encoder, tokenizer, scheduler) | HF `PixArt-alpha/PixArt-Sigma-XL-2-1024-MS` | revision e102b3591cc82e97071b8b4cb90d834d0c487207 (the old VM used a local copy at /home/jl_fs/pixart_sigma_1024 with the same layout) |
| MJHQ-30K metadata | HF dataset `playgroundai/MJHQ-30K` | revision 15b0a659e066e763d0e9a6cd8f00e25f8af5e084; `meta_data.json` is also archived (C_pixart_base) |
| ImageReward-v1.0 | HF `THUDM/ImageReward` (downloaded by `ImageReward.load` into ~/.cache/ImageReward) | ImageReward.pt 1,786,880,927 bytes |
| InceptionV3 (FID) | pytorch-fid release `pt_inception-2015-12-05-6726825d.pth` | downloaded automatically by pytorch-fid |
| VGG16 (LPIPS) | torchvision `vgg16-397923af.pth` | downloaded automatically |

## Human logins needed (no secrets are stored in the repo)
- GitHub: an authorized account with read access to `PseudoHunt/boa-dit` (private or public, as set by the owner) and release-asset download (`gh auth login`).
- Hugging Face: the models above are public. A token is only needed if HF rate-limits anonymous downloads (`huggingface-cli login`).
