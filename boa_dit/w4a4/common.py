"""W4A4 PixArt-Σ phase: paths, prompt sets, T5 embedding cache, bf16 pipeline, the quantized-layer list.
See results/w4a4/rule.md. CFG batch layout in the diffusers pipeline: rows [0, B) uncond, [B, 2B) cond.
"""
import json
import os
import random

import torch

from boa_dit.common import CKPT_DIR, RESULTS, atomic_torch_save

MODEL_DIR = "/home/jl_fs/pixart_sigma_1024"
OUT = os.path.join(RESULTS, "w4a4")
CK = os.path.join(CKPT_DIR, "w4a4")
QDIFF = "/root/ext/deepcompressor/examples/diffusion/prompts/qdiff.yaml"
MJHQ_META = "/root/ext/mjhq/meta_data.json"
STEPS, GUIDANCE, MAX_SEQ = 20, 4.5, 300
LAYERS = ("attn1.to_q", "attn1.to_k", "attn1.to_v", "attn1.to_out.0", "attn2.to_q", "attn2.to_k", "attn2.to_v",
          "attn2.to_out.0", "ff.net.0.proj", "ff.net.2")


def calib_prompts(n=32):
    """deepcompressor's 128 (load_dataset_yaml: random.Random(0).shuffle of the yaml keys, first 128, sorted), first n.
    yaml parses some unquoted keys as ints, so sorting uses key=str."""
    import yaml
    meta = yaml.safe_load(open(QDIFF))
    names = list(meta.keys())
    random.Random(0).shuffle(names)
    names = sorted(names[:128], key=str)[:n]
    return [str(meta[k]).strip() for k in names]


def mjhq_sets():
    """screen (64) and image eval (1K, with MJHQ-256 = its first 256): disjoint fixed subsets of the sorted keys."""
    meta = json.load(open(MJHQ_META))
    keys = sorted(meta)
    screen = random.Random(0).sample(keys, 64)
    rest = sorted(set(keys) - set(screen))
    img = random.Random(1).sample(rest, 1000)
    return {"screen": [meta[k]["prompt"] for k in screen], "mjhq1k": [meta[k]["prompt"] for k in img],
            "screen_keys": screen, "mjhq1k_keys": img}


def prompt_sets():
    m = mjhq_sets()
    return {"calib": calib_prompts(32), "screen": m["screen"], "mjhq1k": m["mjhq1k"]}


@torch.no_grad()
def embed_all(dev):
    """T5 once (bf16, no-grad) for every prompt set + the empty negative prompt; cached to ckpt/w4a4/embeds_<set>.pt."""
    sets = prompt_sets()
    todo = [s for s in sets if not os.path.exists(os.path.join(CK, f"embeds_{s}.pt"))]
    if not todo:
        return
    from diffusers import PixArtSigmaPipeline
    from transformers import T5EncoderModel
    os.makedirs(CK, exist_ok=True)
    te = T5EncoderModel.from_pretrained(os.path.join(MODEL_DIR, "text_encoder"), dtype=torch.bfloat16).to(dev)
    p = PixArtSigmaPipeline.from_pretrained(MODEL_DIR, text_encoder=te, transformer=None, vae=None)
    for s in todo:
        pe, pm, ne, nm = [], [], None, None
        for i in range(0, len(sets[s]), 16):
            a, b, c, d = p.encode_prompt(sets[s][i:i + 16], do_classifier_free_guidance=True, negative_prompt="",
                                         device=dev, max_sequence_length=MAX_SEQ)
            pe.append(a.cpu()); pm.append(b.cpu()); ne, nm = c[:1].cpu(), d[:1].cpu()
        atomic_torch_save({"prompts": sets[s], "prompt_embeds": torch.cat(pe), "prompt_attention_mask": torch.cat(pm),
                           "negative_prompt_embeds": ne, "negative_prompt_attention_mask": nm},
                          os.path.join(CK, f"embeds_{s}.pt"))
        print(f"embedded {s}: {len(sets[s])}", flush=True)
    del p, te
    torch.cuda.empty_cache()


def load_embeds(s):
    return torch.load(os.path.join(CK, f"embeds_{s}.pt"), map_location="cpu")


def embeds_for(e, idx, dev):
    n = len(idx)
    return {"prompt_embeds": e["prompt_embeds"][idx].to(dev), "prompt_attention_mask": e["prompt_attention_mask"][idx].to(dev),
            "negative_prompt_embeds": e["negative_prompt_embeds"].expand(n, -1, -1).to(dev),
            "negative_prompt_attention_mask": e["negative_prompt_attention_mask"].expand(n, -1).to(dev)}


def load_pipe(dev):
    from diffusers import PixArtSigmaPipeline
    pipe = PixArtSigmaPipeline.from_pretrained(MODEL_DIR, torch_dtype=torch.bfloat16, text_encoder=None).to(dev)
    pipe.set_progress_bar_config(disable=True)
    pipe.transformer.requires_grad_(False)
    return pipe


def generators(idx, dev):
    return [torch.Generator(device=dev).manual_seed(int(i)) for i in idx]


def sample(pipe, e, idx, dev, output_type="latent", callback=None):
    """bf16 sampling for prompts idx (seed = prompt index), the model's defaults (DPM-Solver++ 20, g 4.5)."""
    dt = pipe.transformer.dtype
    kw = {k: (v.to(dt) if v.is_floating_point() else v) for k, v in embeds_for(e, idx, dev).items()}
    return pipe(prompt=None, negative_prompt=None, num_inference_steps=STEPS, guidance_scale=GUIDANCE,
                generator=generators(idx, dev), output_type=output_type, **kw).images


def block_linear(block, name):
    m = block
    for part in name.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m
