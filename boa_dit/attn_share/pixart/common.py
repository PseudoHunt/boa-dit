"""PixArt-Σ 1024 setup shared by the diagnostic scripts: paths, prompts, pipeline loading, prompt-embedding cache,
step counter, and the patched attention processors.

Batch layout inside the pipeline (diffusers PixArtSigmaPipeline): rows [0, B) = uncond (empty prompt),
rows [B, 2B) = cond — the opposite of the DiT-XL/2 harness. One transformer call per sampler step.
"""
import os

import torch
import torch.nn.functional as F

from boa_dit.common import CKPT_DIR, RESULTS, atomic_torch_save

MODEL_DIR = "/home/jl_fs/pixart_sigma_1024"
OUT = os.path.join(RESULTS, "attn_share", "pixart")
CK = os.path.join(CKPT_DIR, "pixart")
PROMPTS = ["a red double-decker bus driving down a city street",
           "a bowl of oranges on a wooden kitchen table",
           "a golden retriever lying on green grass in a park",
           "a snowy mountain range under a clear blue sky",
           "a woman in a yellow raincoat holding an umbrella on a rainy street",
           "a sailboat on a calm lake at sunset",
           "a close-up of a bicycle leaning against a brick wall",
           "a plate of sushi on a black table with chopsticks"]
SEEDS = (0, 1)
IMAGES = [(p, s) for p in range(len(PROMPTS)) for s in SEEDS]       # image m = (prompt m // 2, seed m % 2)
G = 64                                                               # 64 x 64 token grid at 1024 px


class _Ctx:
    step = -1     # sampler step of the current transformer call (0 = first, noisiest)
    B = None      # images per branch in the current call


CTX = _Ctx()


def load_pipe(dev, with_text_encoder=False):
    from diffusers import PixArtSigmaPipeline
    if with_text_encoder:   # transformers 5 ignores diffusers' torch_dtype for T5 (would load fp32, 19 GB): load explicitly
        from transformers import T5EncoderModel
        kw = {"text_encoder": T5EncoderModel.from_pretrained(os.path.join(MODEL_DIR, "text_encoder"), dtype=torch.float16)}
    else:
        kw = {"text_encoder": None}
    pipe = PixArtSigmaPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.float16, **kw).to(dev)
    pipe.set_progress_bar_config(disable=True)

    def pre(mod, args, kwargs):
        CTX.step += 1
        CTX.B = args[0].shape[0] // 2
    pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True)
    return pipe


@torch.no_grad()
def prompt_embeds(dev):
    """Encode the 8 prompts + empty prompt once (T5 loaded only here), cached to ckpt/pixart/prompt_embeds.pt."""
    path = os.path.join(CK, "prompt_embeds.pt")
    if not os.path.exists(path):
        os.makedirs(CK, exist_ok=True)
        pipe = load_pipe(dev, with_text_encoder=True)
        pe, pm, ne, nm = pipe.encode_prompt(PROMPTS, do_classifier_free_guidance=True, negative_prompt="",
                                            device=dev, max_sequence_length=300)
        atomic_torch_save({"prompt_embeds": pe.cpu(), "prompt_attention_mask": pm.cpu(),
                           "negative_prompt_embeds": ne[:1].cpu(), "negative_prompt_attention_mask": nm[:1].cpu(),
                           "text_encoder_dtype": str(pipe.text_encoder.dtype)}, path)
        del pipe
        torch.cuda.empty_cache()
    return torch.load(path, map_location="cpu")


def embeds_for(images, emb, dev):
    """Embedding kwargs for a list of (prompt, seed) images."""
    idx = [p for p, _ in images]
    n = len(images)
    return {"prompt_embeds": emb["prompt_embeds"][idx].to(dev), "prompt_attention_mask": emb["prompt_attention_mask"][idx].to(dev),
            "negative_prompt_embeds": emb["negative_prompt_embeds"].expand(n, -1, -1).to(dev),
            "negative_prompt_attention_mask": emb["negative_prompt_attention_mask"].expand(n, -1).to(dev)}


def generators(images, dev):
    return [torch.Generator(dev).manual_seed(s) for _, s in images]


class SelfAttnProcessor:
    """attn1 replacement. mode None: SDPA exactly as AttnProcessor2_0 (no mask for self-attention in PixArt).
    'stats' at steps in STEPS: per head, P = softmax(q k^T scale) in fp32, cb(block, step, head, P_cond, P_uncond),
    output P v. 'share' at steps < k: out = (floor * sum_j v_j + (kres * v)_i) / (floor * N + (kres * 1)_i) via FFT,
    from kernels kfn(block, step) -> (floor [2, H], kres_fft [2, H, S, S//2+1]) for (uncond, cond)."""

    def __init__(self, block):
        self.block = block
        self.mode, self.steps, self.cb, self.k, self.kfn = None, (), None, 0, None

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None, **kw):
        assert encoder_hidden_states is None and attention_mask is None
        B, N, _ = hidden_states.shape
        H = attn.heads
        s = CTX.step
        if self.mode == "share" and s < self.k:
            v = attn.to_v(hidden_states)
            hd = v.shape[-1] // H
            out = share_out(v.view(B, N, H, hd), *self.kfn(self.block, s))
            return attn.to_out[1](attn.to_out[0](out.reshape(B, N, H * hd).to(v.dtype)))
        q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        hd = k.shape[-1] // H
        q, k, v = (t.view(B, N, H, hd).transpose(1, 2) for t in (q, k, v))
        if self.mode == "stats" and s in self.steps:
            out = torch.empty_like(q)
            for h in range(H):
                P = torch.softmax((q[:, h].float() @ k[:, h].float().transpose(-1, -2)) * attn.scale, dim=-1)
                b = B // 2
                self.cb(self.block, s, h, P[b:], P[:b])
                out[:, h] = (P.to(v.dtype) @ v[:, h])
                del P
        elif self.mode == "explicit":
            out = torch.empty_like(q)
            for h in range(H):
                P = torch.softmax((q[:, h].float() @ k[:, h].float().transpose(-1, -2)) * attn.scale, dim=-1)
                out[:, h] = P.to(v.dtype) @ v[:, h]
        else:
            out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(B, N, H * hd).to(q.dtype)
        return attn.to_out[1](attn.to_out[0](out))


FFT_S = 192   # >= 64 + 127 - 1: linear (not circular) convolution on the 64 x 64 grid with a 127 x 127 kernel


def kernel_fft(kres):
    """kres [..., 127, 127] (index [dy + 63, dx + 63] = weight on key at offset pos_j - pos_i = (dy, dx)) ->
    rfft2 of the flipped kernel, so out_i = sum_j kres[pos_j - pos_i] v_j is a plain convolution."""
    return torch.fft.rfft2(kres.flip(-1, -2).float(), s=(FFT_S, FFT_S))


def conv_fft(x, kf):
    """x [B, H, C, 64, 64], kf [B or 1, H, S, S//2+1] -> [B, H, C, 64, 64] (cross-correlation with kres)."""
    y = torch.fft.irfft2(torch.fft.rfft2(x.float(), s=(FFT_S, FFT_S)) * kf[:, :, None], s=(FFT_S, FFT_S))
    return y[..., 63:63 + G, 63:63 + G]


def share_out(v, floor, kf):
    """v [B, N, H, hd]; floor [2, H] and kf [2, H, ...] for (uncond, cond); rows [0, B/2) uncond, [B/2, B) cond."""
    B, N, H, hd = v.shape
    b = B // 2
    br = torch.cat([torch.zeros(b, dtype=torch.long), torch.ones(b, dtype=torch.long)]).to(v.device)
    x = v.float().permute(0, 2, 3, 1).reshape(B, H, hd, G, G)
    fl, k = floor[br], kf[br]                                                      # [B, H], [B, H, S, S/2+1]
    num = conv_fft(x, k) + fl[:, :, None, None, None] * x.sum((-1, -2), keepdim=True)
    den = conv_fft(torch.ones(1, H, 1, G, G, device=v.device), kf)[br] \
        + fl[:, :, None, None, None] * N
    out = num / den
    return out.reshape(B, H, hd, N).permute(0, 3, 1, 2)                            # [B, N, H, hd]


def patch_self_attention(pipe):
    procs = []
    for l, blk in enumerate(pipe.transformer.transformer_blocks):
        p = SelfAttnProcessor(l)
        blk.attn1.set_processor(p)
        procs.append(p)
    return procs


@torch.no_grad()
def decode_images(pipe, latents):
    """VAE-decode one image at a time (a 16 x 1024 px batch does not fit); returns PIL images."""
    out = []
    for z in latents:
        x = pipe.vae.decode(z[None].to(pipe.vae.dtype) / pipe.vae.config.scaling_factor, return_dict=False)[0]
        out += pipe.image_processor.postprocess(x, output_type="pil")
    return out
