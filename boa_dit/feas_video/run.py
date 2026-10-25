"""Part C: query-aware key selection in CogVideoX-2B (FP). See results/feasibility/rule.md.

Joint attention over [226 text | 13 x 30 x 45 = 17,550 video] tokens (video order: frame, row, column). Batch layout:
rows [0, B) uncond (negative prompt), [B, 2B) cond. One transformer call per step (step counter = forward pre-hook).

  embed  : encode the two prompts + empty negative with T5 (loaded explicitly in fp16), cache, drop T5
  stock  : stock pipeline (fused SDPA), prompt 1 seed 0 -> reference latents/frames, s/video, latents after step 0
  stats  : both prompts, stock attention, plus at steps {0,10,11,25,40} explicit rows for 128 sampled video queries
           (cond branch, all blocks and heads): top-1/5/10% mass, text mass, locality, window overlap, step and
           prompt stability of top-10% sets -> stats.json
  arm X  : full | oracle10 | window10 | random10, chunked explicit attention (query chunks of 1,024, fp32 logits per
           head group, text keys always kept, text queries unmasked) at all blocks and steps; prompt 1, seed 0
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json

MODEL = "/home/models/cogvideox-2b"
OUT = os.path.join(RESULTS, "feasibility", "video")
CK = os.path.join(CKPT_DIR, "feas_video")
PROMPTS = ["A golden retriever running through shallow water on a beach at sunset, camera tracking alongside",
           "A busy city intersection at night in the rain, cars and pedestrians with umbrellas, static wide shot"]
T_TXT, NF, NH, NW = 226, 13, 30, 45
NV = NF * NH * NW
STAT_STEPS = (0, 10, 11, 25, 40)
QCHUNK, HGROUP = 1024, 30           # oracle selection over all heads at once (top-k throughput; see model_facts.md)
K10 = round(0.10 * NV)


class Ctx:
    step = -1
    mode = "stock"          # stock | stats | full | oracle10 | window10 | random10
    cb = None


CTX = Ctx()
_coords = None
_rand = None


def coords(dev):
    global _coords
    if _coords is None:
        f, h, w = torch.meshgrid(torch.arange(NF), torch.arange(NH), torch.arange(NW), indexing="ij")
        _coords = torch.stack([f.flatten(), h.flatten(), w.flatten()], -1).to(dev)
    return _coords


def window_mask(qidx, dev):
    """[len(qidx), NV] bool: keys within +-1 latent frame and spatial Chebyshev radius 12 of each video query."""
    c = coords(dev)
    q = c[qidx]
    df = (q[:, None, 0] - c[None, :, 0]).abs()
    ds = torch.maximum((q[:, None, 1] - c[None, :, 1]).abs(), (q[:, None, 2] - c[None, :, 2]).abs())
    return (df <= 1) & (ds <= 12)


def random_mask(qidx, dev):
    """Seeded random 10% of video keys per video query (fixed across heads, steps and arms)."""
    global _rand
    if _rand is None:
        g = torch.Generator(device=dev).manual_seed(0)
        keep = torch.zeros(NV, NV, dtype=torch.bool, device=dev)
        for i in range(0, NV, 2048):
            r = torch.rand(min(2048, NV - i), NV, generator=g, device=dev)
            keep[i:i + 2048].scatter_(1, r.topk(K10, dim=-1).indices, True)
        _rand = keep
    return _rand[qidx]


class Proc:
    """Replacement for CogVideoXAttnProcessor2_0 (no RoPE in the 2B model: asserted)."""

    def __init__(self, block):
        self.block = block

    def __call__(self, attn, hidden_states, encoder_hidden_states, attention_mask=None, image_rotary_emb=None):
        assert image_rotary_emb is None and attention_mask is None
        T = encoder_hidden_states.size(1)
        assert T == T_TXT
        hs = torch.cat([encoder_hidden_states, hidden_states], dim=1)
        B, N, _ = hs.shape
        q, k, v = attn.to_q(hs), attn.to_k(hs), attn.to_v(hs)
        H = attn.heads
        hd = k.shape[-1] // H
        q, k, v = (t.view(B, -1, H, hd).transpose(1, 2) for t in (q, k, v))
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        if CTX.mode in ("stock", "stats"):
            out = F.scaled_dot_product_attention(q, k, v)
            if CTX.mode == "stats" and CTX.step in STAT_STEPS and CTX.cb is not None:
                CTX.cb(self.block, CTX.step, q[B // 2:], k[B // 2:])
        else:
            out = chunked(q, k, v, hd)
        out = out.transpose(1, 2).reshape(B, -1, H * hd)
        out = attn.to_out[1](attn.to_out[0](out))
        return out[:, T:], out[:, :T]


def chunked(q, k, v, hd):
    """One implementation for every arm: query chunks of QCHUNK through fused SDPA with a boolean keep-mask per chunk
    (fp32 accumulation and softmax inside the kernel; the full arm passes an all-True mask so it takes the same kernel
    path). Text keys always kept; text queries unmasked. oracle10 computes explicit fp32 logits per head group only to
    pick each video query's top 10% video keys. (A fully explicit fp32 softmax measured 26-59 min per video; see
    model_facts.md.)"""
    B, H, N, _ = q.shape
    out = torch.empty_like(q)
    dev = q.device
    for i in range(0, N, QCHUNK):
        j = min(N, i + QCHUNK)
        keep = torch.ones(1, 1, j - i, N, dtype=torch.bool, device=dev)
        if CTX.mode == "full" or j <= T_TXT:
            out[:, :, i:j] = F.scaled_dot_product_attention(q[:, :, i:j], k, v, attn_mask=keep)
            continue
        v0 = max(i, T_TXT)
        qidx = torch.arange(v0 - T_TXT, j - T_TXT, device=dev)
        if CTX.mode in ("window10", "random10"):
            keep[0, 0, v0 - i:, T_TXT:] = window_mask(qidx, dev) if CTX.mode == "window10" else random_mask(qidx, dev)
            out[:, :, i:j] = F.scaled_dot_product_attention(q[:, :, i:j], k, v, attn_mask=keep)
            continue
        for h0 in range(0, H, HGROUP):                                                   # oracle10
            with torch.autocast("cuda", enabled=False):
                lg = (q[:, h0:h0 + HGROUP, v0:j].float() * hd ** -0.5) @ k[:, h0:h0 + HGROUP, T_TXT:].float().transpose(-1, -2)
            km = torch.ones(B, HGROUP, j - i, N, dtype=torch.bool, device=dev)
            km[:, :, v0 - i:, T_TXT:] = torch.zeros_like(lg, dtype=torch.bool).scatter_(-1, lg.topk(K10, dim=-1).indices, True)
            del lg
            out[:, h0:h0 + HGROUP, i:j] = F.scaled_dot_product_attention(q[:, h0:h0 + HGROUP, i:j], k[:, h0:h0 + HGROUP],
                                                                         v[:, h0:h0 + HGROUP], attn_mask=km)
    return out


def load_pipe(dev, text_encoder=False):
    from diffusers import CogVideoXPipeline
    kw = {}
    if text_encoder:
        from transformers import T5EncoderModel
        kw["text_encoder"] = T5EncoderModel.from_pretrained(os.path.join(MODEL, "text_encoder"), dtype=torch.float16)
    else:
        kw["text_encoder"] = None
    pipe = CogVideoXPipeline.from_pretrained(MODEL, torch_dtype=torch.float16, **kw).to(dev)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    def pre(mod, args, kwargs):
        CTX.step += 1
    pipe.transformer.register_forward_pre_hook(pre, with_kwargs=True)
    for l, blk in enumerate(pipe.transformer.transformer_blocks):
        blk.attn1.set_processor(Proc(l))
    return pipe


@torch.no_grad()
def embeds(dev):
    p = os.path.join(CK, "embeds.pt")
    if not os.path.exists(p):
        pipe = load_pipe(dev, text_encoder=True)
        pe, ne = pipe.encode_prompt(PROMPTS, negative_prompt="", do_classifier_free_guidance=True, device=dev, max_sequence_length=226)
        atomic_torch_save({"prompt_embeds": pe.cpu(), "negative_prompt_embeds": ne.cpu()}, p)
        del pipe
        torch.cuda.empty_cache()
    return torch.load(p, map_location="cpu")


@torch.no_grad()
def generate(pipe, e, pi, dev, step0_latents=False):
    CTX.step = -1
    rec = {}

    def cb(pipe_, i, t, kw):
        if i == 0 and step0_latents:
            rec["step0"] = kw["latents"].detach().float().cpu()
        return kw
    out = pipe(prompt_embeds=e["prompt_embeds"][pi:pi + 1].to(dev), negative_prompt_embeds=e["negative_prompt_embeds"][pi:pi + 1].to(dev),
               num_frames=49, height=480, width=720, num_inference_steps=50, guidance_scale=6.0,
               generator=torch.Generator(dev).manual_seed(0), output_type="latent", callback_on_step_end=cb)
    lat = out.frames.float().cpu()
    return lat, rec.get("step0")


@torch.no_grad()
def decode_frames(pipe, lat, dev, idx):
    vid = pipe.decode_latents(lat.to(dev).to(pipe.vae.dtype))                       # [1, C, F, H, W] in [-1, 1]
    fr = ((vid[0].float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).permute(1, 0, 2, 3).cpu()   # [F, 3, H, W]
    return fr[list(idx)], fr


def psnr(a, ref):
    mse = (a - ref).pow(2).mean()
    rng = ref.max() - ref.min()
    return float(10 * torch.log10(rng ** 2 / mse))


GRID = (0, 12, 24, 36, 48)
CLIPF = [int(round(x)) for x in np.linspace(0, 48, 8)]


@torch.no_grad()
def clip_score(frames, text, dev):
    from transformers import CLIPModel, CLIPProcessor
    m = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
    imgs = [f.permute(1, 2, 0).numpy() for f in frames]
    x = proc(text=[text] * len(imgs), images=imgs, return_tensors="pt", padding=True).to(dev)
    o = m(**x)
    ie = o.image_embeds / o.image_embeds.norm(dim=-1, keepdim=True)
    te = o.text_embeds / o.text_embeds.norm(dim=-1, keepdim=True)
    return float((100 * (ie * te).sum(-1).clamp_min(0)).mean())


@torch.no_grad()
def run_arm(name, dev):
    out = os.path.join(OUT, "eval", f"{name}.json")
    if os.path.exists(out):
        return
    e = embeds(dev)
    pipe = load_pipe(dev)
    free = torch.cuda.mem_get_info(dev)[0] / 2 ** 30
    with Run("feas_video", f"arm_{name}", dev, {}) as run:
        CTX.mode = name
        t0 = time.time()
        lat, s0 = generate(pipe, e, 0, dev, step0_latents=True)
        sec = time.time() - t0
        CTX.mode = "stock"
        grid, allf = decode_frames(pipe, lat, dev, GRID)
        atomic_torch_save({"latents": lat, "step0": s0, "grid": grid}, os.path.join(CK, f"{name}.pt"))
        clip = clip_score(allf[CLIPF], PROMPTS[0], dev)
        res = {"arm": name, "seconds_per_video": sec, "gpu_free_gb_before": free, "clip_vitl14_8frames": clip}
        ref = os.path.join(CK, "full.pt") if name != "full" else os.path.join(CK, "stock.pt")
        if os.path.exists(ref):
            r = torch.load(ref, map_location="cpu")
            res["latent_psnr_vs_" + ("full" if name != "full" else "stock")] = psnr(lat, r["latents"])
            if r.get("step0") is not None and s0 is not None:
                res["step0_latent_rel_diff_vs_" + ("full" if name != "full" else "stock")] = float((s0 - r["step0"]).norm() / r["step0"].norm())
        res.update(run.record())
    atomic_write_json(out, res)
    print(res, flush=True)


@torch.no_grad()
def stats(dev):
    e = embeds(dev)
    pipe = load_pipe(dev)
    g = np.random.default_rng(0)
    per = [128 // NF + (1 if f < 128 % NF else 0) for f in range(NF)]               # stratified over frames
    qs = []
    for f in range(NF):
        sp = g.choice(NH * NW, per[f], replace=False)
        qs += [f * NH * NW + int(x) for x in sp]
    qidx = torch.tensor(sorted(qs), device=dev)
    win = window_mask(qidx, dev)                                                       # [128, NV]
    kq = {p: round(p / 100 * NV) for p in (1, 5, 10)}
    acc, tops = {}, {}

    def cb(l, s, q, k):
        qq = q[0, :, T_TXT + qidx].float()                                             # [H, 128, hd] (cond, batch 1)
        with torch.autocast("cuda", enabled=False):
            P = torch.softmax((qq * qq.shape[-1] ** -0.5) @ k[0].float().transpose(-1, -2), -1)   # [H, 128, T + NV]
        txt = P[..., :T_TXT].sum(-1)
        Pv = P[..., T_TXT:]
        srt, idx = Pv.sort(-1, descending=True)
        r = {f"top{p}_mass": srt[..., :n].sum(-1) for p, n in kq.items()}
        top10 = torch.zeros_like(Pv, dtype=torch.bool).scatter_(-1, idx[..., :K10], True)
        r["text_mass"] = txt
        r["locality"] = (Pv * (top10 & win)).sum(-1) / (Pv * top10).sum(-1)
        inter = (top10 & win).sum(-1).float()
        r["window_overlap"] = inter / (top10 | win).sum(-1).float()
        acc[(CTX_P[0], l, s)] = {kk: vv.median(-1).values.cpu() for kk, vv in r.items()}   # median over queries -> [H]
        tops[(CTX_P[0], l, s)] = idx[..., :K10].to(torch.int16).cpu()
    CTX_P = [0]
    times = {}
    with Run("feas_video", "stats", dev, {}) as run:
        for pi in range(2):
            CTX_P[0] = pi
            CTX.mode, CTX.cb = "stats", cb
            t0 = time.time()
            lat, _ = generate(pipe, e, pi, dev)
            times[pi] = time.time() - t0
        CTX.mode, CTX.cb = "stock", None
        rec = run.record()

    def jac(a, b):
        A = torch.zeros(*a.shape[:-1], NV, dtype=torch.bool).scatter_(-1, a.long(), True)
        Bm = torch.zeros(*b.shape[:-1], NV, dtype=torch.bool).scatter_(-1, b.long(), True)
        return ((A & Bm).sum(-1).float() / (A | Bm).sum(-1).float()).median(-1).values      # [H]
    res = {"queries": qidx.tolist(), "steps": STAT_STEPS, "per": {}, "seconds_stats_runs": times}
    for (pi, l, s), r in acc.items():
        d = {kk: vv.tolist() for kk, vv in r.items()}
        if pi == 0 and s == 11:
            d["step_jaccard_10_11"] = jac(tops[(0, l, 10)], tops[(0, l, 11)]).tolist()
        if pi == 0:
            d["prompt_jaccard"] = jac(tops[(0, l, s)], tops[(1, l, s)]).tolist()
        res["per"][f"p{pi}.b{l}.s{s}"] = d
    atomic_write_json(os.path.join(OUT, "stats.json"), {**res, **rec})
    print("stats done", times, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", help="embed | stats | comma list of arms: stock, full, oracle10, window10, random10")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    if a.what == "embed":
        embeds(a.device)
    elif a.what == "stats":
        stats(a.device)
    else:
        for name in a.what.split(","):
            run_arm(name, a.device)


if __name__ == "__main__":
    main()
