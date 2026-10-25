"""Load DiT-XL/2 + VAE; hooks on adaLN outputs and on block-linear inputs."""
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR
from models import DiT_XL_2  # external/DiT
from diffusion import create_diffusion  # external/DiT

NULL_CLASS = 1000
VAE_SCALE = 0.18215
LINEARS = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def load_dit(device, dtype=torch.float32):
    # fp32 weights; the forward runs under fp16 autocast (see sample.ddim_sample).
    model = DiT_XL_2(input_size=32)
    sd = torch.load(os.path.join(CKPT_DIR, "DiT-XL-2-256x256.pt"), map_location="cpu", weights_only=True)
    if "ema" in sd:
        sd = sd["ema"]
    model.load_state_dict(sd)
    model.eval().requires_grad_(False)
    return model.to(device=device, dtype=dtype)


@torch.no_grad()
def load_quant_state(model, path):
    """Load a saved state dict via CPU and verify it landed.

    GPU peer-to-peer copies on this host are silently wrong (cuda:1 -> cuda:0 `copy_` returns bad data with no
    error), and torch.load without map_location restores tensors onto the GPU they were saved from. Always go
    through CPU, then check every tensor actually matches.
    """
    sd = torch.load(path, map_location="cpu")
    model.load_state_dict(sd)
    for k, v in model.state_dict().items():
        assert torch.equal(v.cpu(), sd[k]), f"state dict did not load correctly: {k}"
    return model


def load_vae(device):
    from diffusers.models import AutoencoderKL
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").to(device).eval()
    vae.requires_grad_(False)
    return vae


def make_diffusion(steps=50):
    return create_diffusion(timestep_respacing=f"ddim{steps}")


def labels_for_seeds(seeds):
    return np.array([int(np.random.default_rng(s).integers(0, 1000)) for s in seeds], dtype=np.int64)


def noise_for_seeds(seeds, device):
    out = []
    for s in seeds:
        g = torch.Generator(device=device).manual_seed(int(s))
        out.append(torch.randn(1, 4, 32, 32, generator=g, device=device, dtype=torch.float32))
    return torch.cat(out)


def block_linear(block, name):
    mod = block
    for part in name.split("."):
        mod = getattr(mod, part)
    return mod


@torch.no_grad()
def decode(vae, latents, bs=16):
    imgs = []
    for i in range(0, len(latents), bs):
        x = vae.decode(latents[i:i + bs].float() / VAE_SCALE).sample
        imgs.append(((x.clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu())
    return torch.cat(imgs)  # [N,3,256,256] uint8
