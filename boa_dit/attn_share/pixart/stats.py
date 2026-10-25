"""PixArt-Σ 1024, §1–2: model facts, processor parity, and the self-attention statistics sweep (+ cross-attn control).

  facts  : verify the model against the loaded objects -> results/attn_share/pixart/model_facts.json
  parity : explicit per-head fp32-softmax processor vs default SDPA, all block outputs, one transformer call
  stats  : 16 images (8 prompts x seeds 0, 1), 20 steps; at steps 0-9, per (block, head): within / same / diff prompt,
           entropy, KL ratio, branch cosine, 127x127 offset kernel (+ floor decomposition). Block 13 attn2 (heads 0, 7,
           15) gives the cross-attention control. Saves the FP images, a grid, and kappa (ckpt/pixart/kappa.pt).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch

from boa_dit.attn_share.gate1 import kernel_cos, offsets, pair_within, row_cos
from boa_dit.attn_share.pixart.common import (CK, CTX, G, IMAGES, MODEL_DIR, OUT, PROMPTS, embeds_for, generators,
                                              load_pipe, patch_self_attention, prompt_embeds, decode_images)
from boa_dit.common import Run, atomic_torch_save, atomic_write_json

S_STATS = 10
CROSS_BLOCK, CROSS_HEADS = 13, (0, 7, 15)
KN = 2 * G - 1                                                                   # 127


def support_side(kap, q=0.95):
    """Smallest odd side of a centered square holding >= q of kap's mass; kap [..., KN, KN] (non-negative)."""
    kap = kap / kap.sum((-1, -2), keepdim=True)
    c = KN // 2
    side = torch.full(kap.shape[:-2], KN, device=kap.device)
    for r in range(c, -1, -1):
        m = kap[..., c - r:c + r + 1, c - r:c + r + 1].sum((-1, -2))
        side = torch.where(m >= q, torch.full_like(side, 2 * r + 1), side)
    return side


def floor_stats(kap):
    """floor = min / median of kappa; share of mass; 95% side of (kappa - floor)+."""
    flat = kap.flatten(-2)
    out = {}
    for how in ("min", "median"):
        fl = flat.min(-1).values if how == "min" else flat.median(-1).values
        out[f"floor_share_{how}"] = fl * flat.shape[-1] / flat.sum(-1)
        out[f"side_{how}"] = support_side((kap - fl[..., None, None]).clamp_min(0)).float()
    return out


@torch.no_grad()
def facts(dev):
    import diffusers
    pipe = load_pipe(dev)
    tr = pipe.transformer
    cfg = dict(tr.config)
    blk = tr.transformer_blocks[0]
    f = {"diffusers": diffusers.__version__, "model_dir": MODEL_DIR, "transformer_class": type(tr).__name__,
         "n_blocks": len(tr.transformer_blocks), "hidden": cfg["num_attention_heads"] * cfg["attention_head_dim"],
         "heads": blk.attn1.heads, "head_dim": blk.attn1.inner_dim // blk.attn1.heads, "patch_size": cfg["patch_size"],
         "sample_size": cfg["sample_size"], "in_channels": cfg["in_channels"], "out_channels": cfg["out_channels"],
         "norm_type": cfg.get("norm_type"), "caption_channels": cfg.get("caption_channels"),
         "attn1_processor_default": type(blk.attn1.processor).__name__,
         "attn2_present": blk.attn2 is not None, "attn1_scale": blk.attn1.scale,
         "scheduler": type(pipe.scheduler).__name__, "scheduler_config": {k: v for k, v in dict(pipe.scheduler.config).items()
                                                                         if not k.startswith("_")},
         "vae": type(pipe.vae).__name__, "vae_scaling_factor": pipe.vae.config.scaling_factor,
         "dtype": str(tr.dtype), "pipeline_defaults": {"num_inference_steps": 20, "guidance_scale": 4.5,
                                                       "max_sequence_length": 300, "negative_prompt": ""},
         "cfg_batch_layout": "rows [0, B) uncond (empty prompt), rows [B, 2B) cond (pipeline: cat([negative, prompt]))"}
    f["latent_hw"] = cfg["sample_size"]
    f["image_tokens"] = (cfg["sample_size"] // cfg["patch_size"]) ** 2
    te = json.load(open(os.path.join(MODEL_DIR, "text_encoder", "config.json")))
    f["text_encoder"] = {"arch": te.get("architectures"), "d_model": te.get("d_model"), "num_layers": te.get("num_layers"),
                         "weights": "fp32 shards converted to fp16 on download (disk limit); loaded in fp16"}
    free, total = torch.cuda.mem_get_info(dev)
    f["gpu_free_gb_after_text_encoder_dropped"] = free / 2 ** 30
    f["gpu_total_gb"] = total / 2 ** 30
    atomic_write_json(os.path.join(OUT, "model_facts.json"), f)
    print(json.dumps(f, indent=1, default=str))


@torch.no_grad()
def parity(dev):
    emb = prompt_embeds(dev)
    pipe = load_pipe(dev)
    procs = patch_self_attention(pipe)
    tr = pipe.transformer
    imgs = IMAGES[:2]
    e = embeds_for(imgs, emb, dev)
    gen = torch.Generator(dev).manual_seed(0)
    x = torch.randn(2, 4, 128, 128, generator=gen, device=dev, dtype=torch.float16)
    xx = torch.cat([x, x])
    enc = torch.cat([e["negative_prompt_embeds"], e["prompt_embeds"]])
    msk = torch.cat([e["negative_prompt_attention_mask"], e["prompt_attention_mask"]])
    t = torch.full((4,), 999, device=dev)
    outs = {}
    hooks = [b.register_forward_hook(lambda m, i, o, l=l: outs.__setitem__(l, o.float().clone()))
             for l, b in enumerate(tr.transformer_blocks)]
    ac = {"resolution": None, "aspect_ratio": None}

    def run(mode):
        for p in procs:
            p.mode = mode
        CTX.step = -1
        tr(xx, encoder_hidden_states=enc, encoder_attention_mask=msk, timestep=t, added_cond_kwargs=ac)
        return {l: v.clone() for l, v in outs.items()}

    ref, exp = run(None), run("explicit")
    from torch.nn.attention import SDPBackend, sdpa_kernel
    with sdpa_kernel(SDPBackend.MATH):                                # a second correct SDPA: the fp16 noise floor
        alt = run(None)
    for h in hooks:
        h.remove()
    mx = lambda a, b: [float((a[l] - b[l]).abs().max() / b[l].abs().max()) for l in range(len(b))]
    fro = lambda a, b: [float((a[l] - b[l]).norm() / b[l].norm()) for l in range(len(b))]
    d = mx(exp, ref)
    res = {"explicit_vs_sdpa_max_rel": max(d), "per_block": d, "tolerance": 1e-3,
           "explicit_vs_sdpa_rel_fro_max": max(fro(exp, ref)),
           "sdpa_math_vs_sdpa_default_max_rel": max(mx(alt, ref)), "sdpa_math_vs_sdpa_default_rel_fro_max": max(fro(alt, ref)),
           "fp16_ulp_rel_at_max": 2 ** -10,
           "note": "hidden states are fp16 throughout PixArt (DiT harness had an fp32 residual stream); 1e-3 relative to "
                   "the max is ~1 fp16 ulp. The math-backend SDPA gives the noise floor between two correct implementations."}
    print(res)
    return res


@torch.no_grad()
def stats(dev, blocks=None, estimate=False):
    emb = prompt_embeds(dev)
    pipe = load_pipe(dev)
    procs = patch_self_attention(pipe)
    nb = len(procs)
    blocks = list(range(nb)) if blocks is None else blocks
    L, H, N = nb, 16, G * G
    offs = offsets(G).to(dev)
    NAMES = ["within_median", "within_p10", "within_sameprompt", "within_diffprompt", "ent_median", "ent_meanmap_median",
             "kl_ratio_median", "branch_cos_median", "kernel_cos_median", "floor_share_min", "side_min",
             "floor_share_median", "side_median"]
    st = {k: torch.full((S_STATS, L, H), float("nan")) for k in NAMES}
    kap_store = {br: torch.zeros(S_STATS, L, H, KN, KN) for br in ("cond", "uncond")}
    logN = math.log(N)
    flat = offs.flatten()
    cnt = torch.zeros(KN * KN, device=dev).index_add_(0, flat, torch.ones(flat.shape[0], device=dev))
    tcb = []

    def kernel_of(M):
        return (torch.zeros(KN * KN, device=dev).index_add_(0, flat, M.flatten()) / cnt).reshape(KN, KN)

    def cb(l, s, h, Pc, Pu):
        t0 = time.time()
        Mc, Mu = Pc.mean(0), Pu.mean(0)
        w, ws, wd = pair_within(Pc[:, None])
        w, ws, wd = w[0], ws[0], wd[0]
        st["within_median"][s, l, h] = w.median()
        st["within_p10"][s, l, h] = torch.quantile(w, 0.1)
        st["within_sameprompt"][s, l, h] = ws.median()
        st["within_diffprompt"][s, l, h] = wd.median()
        logP = Pc.clamp_min(1e-30).log()
        ent = -(Pc * logP).sum(-1)                                                 # [M, N]
        st["ent_median"][s, l, h] = (ent / logN).median()
        st["ent_meanmap_median"][s, l, h] = (-(Mc * Mc.clamp_min(1e-30).log()).sum(-1) / logN).median()
        kl_m = (Pc * (logP - Mc.clamp_min(1e-30).log())).sum(-1)
        del logP
        st["kl_ratio_median"][s, l, h] = ((logN - ent).mean(0) / kl_m.mean(0)).median()
        st["branch_cos_median"][s, l, h] = row_cos(Mc, Mu).median()
        st["kernel_cos_median"][s, l, h] = kernel_cos(Mc[None], offs).median()
        kc, ku = kernel_of(Mc), kernel_of(Mu)
        kap_store["cond"][s, l, h], kap_store["uncond"][s, l, h] = kc.cpu(), ku.cpu()
        for k, v in floor_stats(kc).items():
            st[k][s, l, h] = v
        torch.cuda.synchronize(dev)
        tcb.append(time.time() - t0)

    for l in blocks:
        procs[l].mode, procs[l].steps, procs[l].cb = "stats", tuple(range(S_STATS)), cb

    # cross-attention control: block 13 attn2, heads 0/7/15, cond rows, same `within` statistic over text tokens
    from diffusers.models.attention_processor import AttnProcessor2_0
    cross = {k: torch.full((S_STATS, len(CROSS_HEADS)), float("nan")) for k in ("within_median", "within_sameprompt",
                                                                               "within_diffprompt")}

    class CrossCtl:
        def __init__(self):
            self.base = AttnProcessor2_0()

        def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, **kw):
            s = CTX.step
            if s < S_STATS:
                B = hidden_states.shape[0]
                mask = attn.prepare_attention_mask(attention_mask, encoder_hidden_states.shape[1], B)
                mask = mask.view(B, attn.heads, -1, mask.shape[-1])
                q, k = attn.to_q(hidden_states), attn.to_k(encoder_hidden_states)
                hd = k.shape[-1] // attn.heads
                q = q.view(B, -1, attn.heads, hd).transpose(1, 2)
                k = k.view(B, -1, attn.heads, hd).transpose(1, 2)
                b = B // 2
                for j, h in enumerate(CROSS_HEADS):
                    P = torch.softmax((q[b:, h].float() @ k[b:, h].float().transpose(-1, -2)) * attn.scale
                                      + mask[b:, h].float(), dim=-1)                # [16, N, T]
                    w, ws, wd = pair_within(P[:, None])
                    cross["within_median"][s, j], cross["within_sameprompt"][s, j] = w[0].median(), ws[0].median()
                    cross["within_diffprompt"][s, j] = wd[0].median()
            return self.base(attn, hidden_states, encoder_hidden_states, attention_mask, **kw)

    pipe.transformer.transformer_blocks[CROSS_BLOCK].attn2.set_processor(CrossCtl())
    e = embeds_for(IMAGES, emb, dev)
    CTX.step = -1
    n_steps = 1 if estimate else 20
    t0 = time.time()
    with Run("attn_share", "pixart_stats" + ("_estimate" if estimate else ""), dev,
             {"blocks": blocks, "n_steps": n_steps}) as run:
        out = pipe(prompt=None, num_inference_steps=n_steps if estimate else 20, guidance_scale=4.5, negative_prompt=None,
                   generator=generators(IMAGES, dev), output_type="latent", max_sequence_length=300, **e)
        rec = run.record()
    wall = time.time() - t0
    if estimate:
        per_call = float(np.mean(tcb))
        est = {"stats_s_per_block_step_head": per_call, "wall_1_step_s": wall,
               "projected_full_sweep_gpu_h": (per_call * H * len(blocks) * S_STATS + 20 * 3.0) / 3600}
        print(est)
        atomic_write_json(os.path.join(OUT, "stats_estimate.json"), est)
        return est
    assert CTX.step == 19, f"expected 20 transformer calls, got {CTX.step + 1}"
    os.makedirs(CK, exist_ok=True)
    atomic_torch_save({"kappa": kap_store, "layout": "[step 0-9, block, head, 127, 127]; index [dy + 63, dx + 63]"},
                      os.path.join(CK, "kappa.pt"))
    atomic_torch_save(out.images.cpu(), os.path.join(CK, "fp_latents.pt"))
    imgs = decode_images(pipe, out.images)
    for m, im in enumerate(imgs):
        im.save(os.path.join(CK, f"fp_{m:02d}_p{IMAGES[m][0]}_s{IMAGES[m][1]}.png"))
    from PIL import Image
    grid = Image.new("RGB", (8 * 256, 2 * 256))
    for m, im in enumerate(imgs):
        grid.paste(im.resize((256, 256)), ((m // 2) * 256, (m % 2) * 256))
    grid.save(os.path.join(OUT, "fp_grid.jpg"), quality=90)
    atomic_write_json(os.path.join(OUT, "stats.json"), {
        "blocks": blocks, "steps": list(range(S_STATS)), "images": IMAGES, "prompts": PROMPTS, "shape": "[step][block][head]",
        "stats": {k: v.tolist() for k, v in st.items()}, "cross_attention_control": {
            "block": CROSS_BLOCK, "heads": CROSS_HEADS, "shape": "[step][head]", **{k: v.tolist() for k, v in cross.items()}},
        "stats_s_per_head_call_mean": float(np.mean(tcb)), **rec})
    print("done", rec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["embed", "facts", "parity", "estimate", "stats"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--blocks", default=None)
    a = ap.parse_args()
    blocks = [int(b) for b in a.blocks.split(",")] if a.blocks else None
    if a.what == "embed":
        prompt_embeds(a.device)
    elif a.what == "facts":
        facts(a.device)
    elif a.what == "parity":
        res = parity(a.device)
        atomic_write_json(os.path.join(OUT, "parity.json"), res)
    elif a.what == "estimate":
        stats(a.device, blocks, estimate=True)
    else:
        stats(a.device, blocks)


if __name__ == "__main__":
    main()
