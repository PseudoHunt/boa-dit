"""Part V: query-aware key selection in VAR-d30 (FP). See results/feasibility/rule.md.

The attention of every block is patched. For the last three scales (queries of 10x10, 13x13, 16x16 = L of 100, 169,
256, while KV-caching), it computes logits explicitly in fp32, l = (q-hat * scale_mul) . k-hat as in the stock attn_l2_norm path, fp32 softmax, P v in fp16.
Stock uses fused SDPA, so token identity with stock cannot hold (discrete sampler); all arms share this explicit path. Then: optional stats callback, the arm's key mask (masked logits set to
-inf, the first/class token always kept), fp32 softmax, P v. The other scales use the stock path.

  parity : stock vs patched-full logits (rel diff) and tokens / images for 8 images, one seed
  stats  : 64 images; per (block, head, scale): same-scale vs coarser mass, keys for 90/95% mass, positional share of
           the oracle top-25% mass, positional-head share, cross-image Jaccard of top-25% sets -> stats.json
  arm X  : full | oracle25 | positional25 | random25: N images (fixed batches of 16, labels + batch seeds from
           default_rng(0)), saved as uint8; then FID-1k, ResNet-50 label accuracy (Wilson CI), class change vs full
"""
import argparse
import math
import os

import numpy as np
import torch

from boa_dit.attn_share.gate2 import resnet_top1
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_fid import REF_STATS, fid_from_stats, inception_stats
from boa_dit.feas_var.model import PATCH_NUMS, load, sample

OUT = os.path.join(RESULTS, "feasibility", "var")
CK = os.path.join(CKPT_DIR, "feas_var")
LAST = {100: 10, 169: 13, 256: 16}                      # query count -> patch number for the last three scales
BS = 16


class Ctl:
    arm = "stock"          # stock | full | oracle25 | positional25 | random25
    cb = None


CTL = Ctl()


def key_coords(n_keys):
    """Normalized patch-center coordinates of the first n_keys tokens (all scales in order)."""
    pts = []
    for pn in PATCH_NUMS:
        g = (torch.arange(pn) + 0.5) / pn
        yy, xx = torch.meshgrid(g, g, indexing="ij")
        pts.append(torch.stack([yy.flatten(), xx.flatten()], -1))
    return torch.cat(pts)[:n_keys]


_cache = {}


def static_mask(kind, pn, dev):
    """[Lq, K] bool keep-mask for positional25 / random25 (first token always kept)."""
    key = (kind, pn)
    if key not in _cache:
        n_keys = sum(p * p for p in PATCH_NUMS[:PATCH_NUMS.index(pn) + 1])
        kc = key_coords(n_keys)
        qc = kc[-pn * pn:]
        k = round(0.25 * n_keys)
        if kind == "positional25":
            d = torch.cdist(qc.double(), kc.double())
            idx = torch.argsort(d + 1e-9 * torch.arange(n_keys, dtype=torch.float64), dim=-1, stable=True)[:, :k]
        else:
            g = torch.Generator().manual_seed(0 + pn)
            idx = torch.stack([torch.randperm(n_keys, generator=g)[:k] for _ in range(pn * pn)])
        m = torch.zeros(pn * pn, n_keys, dtype=torch.bool)
        m.scatter_(1, idx, True)
        m[:, 0] = True
        _cache[key] = m.to(dev)
    return _cache[key]


def patch(var):
    for bi, blk in enumerate(var.blocks):
        attn = blk.attn
        orig = attn.forward

        def fwd(x, attn_bias, attn=attn, orig=orig, bi=bi):
            B, L, C = x.shape
            if CTL.arm == "stock" or L not in LAST or not attn.caching:
                return orig(x, attn_bias)
            import torch.nn.functional as F
            qkv = F.linear(input=x, weight=attn.mat_qkv.weight,
                           bias=torch.cat((attn.q_bias, attn.zero_k_bias, attn.v_bias))).view(B, L, 3, attn.num_heads, attn.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)                      # BHLc
            scale_mul = attn.scale_mul_1H11.clamp_max(attn.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            k = attn.cached_k = torch.cat((attn.cached_k, k), dim=2)
            v = attn.cached_v = torch.cat((attn.cached_v, v), dim=2)
            pn = LAST[L]
            # fp32 logits and softmax (closest to fused SDPA's fp32 accumulation; see parity.json), P v in fp16
            with torch.autocast("cuda", enabled=False):
                lg = (q.float() * attn.scale) @ k.float().transpose(-1, -2)            # [B, H, L, K]
            if CTL.cb is not None:
                CTL.cb(bi, pn, lg)
            if CTL.arm == "oracle25":
                kk = round(0.25 * lg.shape[-1])
                keep = torch.zeros_like(lg, dtype=torch.bool).scatter_(-1, lg.topk(kk, dim=-1).indices, True)
                keep[..., 0] = True
                lg = lg.masked_fill(~keep, float("-inf"))
            elif CTL.arm in ("positional25", "random25"):
                lg = lg.masked_fill(~static_mask(CTL.arm, pn, lg.device), float("-inf"))
            out = (lg.softmax(dim=-1).to(v.dtype) @ v).transpose(1, 2).reshape(B, L, C)
            return attn.proj_drop(attn.proj(out))
        attn.forward = fwd


def design(n):
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 1000, n).tolist()
    seeds = rng.integers(0, 2 ** 31 - 1, math.ceil(n / BS)).tolist()
    return labels, seeds


@torch.no_grad()
def parity(var, dev):
    """Per-call parity on identical inputs: in every patched call (all blocks, scales 10/13/16), compare the explicit
    logits (fp16, stock arithmetic) with fp32 logits, and the explicit output P v with stock fused SDPA on the same
    q, k, v. Token identity with the stock run is also reported (it cannot hold: fused vs explicit numerics feed a
    discrete sampler)."""
    import torch.nn.functional as F
    labels, seeds = design(8)
    worst = {"logits": 0.0, "out": 0.0}

    def cb(bi, pn, lg):
        pass
    CTL.arm, CTL.cb = "full", cb
    hooks = []
    for bi, blk in enumerate(var.blocks):
        a = blk.attn

        def chk(mod, args, kw, out, a=a):
            x = args[0]
            B, L, C = x.shape
            if L not in LAST:
                return
            qkv = F.linear(input=x, weight=a.mat_qkv.weight,
                           bias=torch.cat((a.q_bias, a.zero_k_bias, a.v_bias))).view(B, L, 3, a.num_heads, a.head_dim)
            q, _, _ = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0)
            sm = a.scale_mul_1H11.clamp_max(a.max_scale_mul).exp()
            q = F.normalize(q, dim=-1).mul(sm)
            k, v = a.cached_k, a.cached_v                                              # include this scale (just appended)
            l16 = q.mul(a.scale) @ k.transpose(-2, -1)
            with torch.autocast("cuda", enabled=False):
                l32 = (q.float() * a.scale) @ k.float().transpose(-1, -2)
            worst["logits"] = max(worst["logits"], float((l16.float() - l32).abs().max() / l32.abs().max()))
            ref = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(B, L, C)
            mine = (l32.softmax(dim=-1).to(v.dtype) @ v).transpose(1, 2).reshape(B, L, C)
            worst["out"] = max(worst["out"], float((mine.float() - ref.float()).abs().max() / ref.float().abs().max()))
            from torch.nn.attention import SDPBackend, sdpa_kernel
            with sdpa_kernel(SDPBackend.MATH):
                alt = F.scaled_dot_product_attention(q, k, v, scale=a.scale).transpose(1, 2).reshape(B, L, C)
            worst["floor"] = max(worst.get("floor", 0.0), float((alt.float() - ref.float()).abs().max() / ref.float().abs().max()))
        hooks.append(a.register_forward_hook(chk, with_kwargs=True))
    xf = sample(var, labels, seeds[0], dev)
    for h in hooks:
        h.remove()
    CTL.arm, CTL.cb = "stock", None
    xs = sample(var, labels, seeds[0], dev)
    return {"per_call_logits_fp16_vs_fp32_max_rel": worst["logits"], "per_call_output_explicit_vs_sdpa_max_rel": worst["out"],
            "per_call_output_sdpa_math_vs_default_max_rel": worst["floor"],
            "tolerance": 1e-3, "images_identical_to_stock": bool(torch.equal(xf, xs)), "n_images": 8,
            "note": "stock uses fused F.scaled_dot_product_attention; token identity with an explicit path cannot hold "
                    "under discrete top-k/top-p sampling. All arms use the explicit path; 'full' is the reference, and a "
                    "'stock' arm checks statistical equivalence."}


@torch.no_grad()
def stats(var, dev, n=64):
    labels, seeds = design(n)
    acc = {}

    def cb(bi, pn, lg):
        P = lg[:lg.shape[0] // 2].softmax(-1)                                      # cond rows [B, H, Lq, K]
        Bq, H, Lq, K = P.shape
        same = P[..., K - Lq:].sum(-1)                                             # mass on same-scale keys
        Ps, idx = P.sort(-1, descending=True)
        cs = Ps.cumsum(-1)
        k90 = ((cs < 0.90).sum(-1) + 1).float() / K
        k95 = ((cs < 0.95).sum(-1) + 1).float() / K
        k25 = round(0.25 * K)
        top = torch.zeros_like(P, dtype=torch.bool).scatter_(-1, idx[..., :k25], True)
        pos = static_mask("positional25", pn, P.device)
        top_mass = (P * top).sum(-1)
        pos_share = (P * (top & pos)).sum(-1) / top_mass
        inter = torch.einsum("bhqk,chqk->hqbc", top.float(), top.float())          # [H, Lq, B, B]
        jac = inter / (2 * k25 - inter)
        off = ~torch.eye(Bq, dtype=torch.bool, device=P.device)
        jac_m = jac[..., off].mean(-1)                                             # [H, Lq]
        a = acc.setdefault((bi, pn), {k: [] for k in ("same", "k90", "k95", "pos", "jac")})
        a["same"].append(same.mean(0)); a["k90"].append(k90.median(0).values if False else k90.float().mean(0))
        a["k95"].append(k95.mean(0)); a["pos"].append(pos_share.mean(0)); a["jac"].append(jac_m)
    CTL.arm, CTL.cb = "full", cb
    for i in range(0, n, BS):
        sample(var, labels[i:i + BS], seeds[i // BS], dev)
    CTL.cb = None
    res = {}
    for (bi, pn), a in acc.items():
        m = {k: torch.stack(v).mean(0) for k, v in a.items()}                     # [H, Lq] (mean over batches)
        med = {k: v.median(-1).values for k, v in m.items()}                        # median over queries -> [H]
        res[f"b{bi}.s{pn}"] = {"block": bi, "scale": pn, "same_scale_mass": med["same"].tolist(),
                               "coarser_mass": (1 - med["same"]).tolist(), "frac_keys_90": med["k90"].tolist(),
                               "frac_keys_95": med["k95"].tolist(), "positional_share": med["pos"].tolist(),
                               "positional_heads_frac": float((med["pos"] >= 0.9).float().mean()),
                               "jaccard_cross_image": med["jac"].tolist()}
    return res


@torch.no_grad()
def arm(var, vae, name, n, dev):
    out = os.path.join(OUT, "eval", f"{name}.json")
    if os.path.exists(out):
        return
    labels, seeds = design(n)
    part = os.path.join(CK, f"{name}.partial")
    done = torch.load(part, map_location="cpu") if os.path.exists(part) else {}
    with Run("feas_var", f"arm_{name}", dev, {"n": n}) as run:
        CTL.arm, CTL.cb = name, None
        for i in range(0, n, BS):
            if i in done:
                continue
            done[i] = (sample(var, labels[i:i + BS], seeds[i // BS], dev).clamp(0, 1) * 255).round().to(torch.uint8).cpu()
            atomic_torch_save(done, part)
        CTL.arm = "stock"
        imgs = torch.cat([done[i] for i in range(0, n, BS)])
        atomic_torch_save(imgs, os.path.join(CK, f"{name}_imgs.pt"))
        mu, sig = inception_stats(imgs.permute(0, 2, 3, 1).numpy(), dev)
        ref = np.load(REF_STATS)
        top1 = resnet_top1(imgs, dev)
        y = torch.tensor(labels)
        acc = float((top1 == y).float().mean())
        z = 1.96
        den = 1 + z * z / n
        c = (acc + z * z / (2 * n)) / den
        h = z * math.sqrt(acc * (1 - acc) / n + z * z / (4 * n * n)) / den
        res = {"arm": name, "n": n, "fid1k": fid_from_stats(mu, sig, ref["mu"], ref["sigma"]), "label_acc": acc,
               "label_acc_ci95_wilson": [c - h, c + h], "top1": top1.tolist(), **run.record()}
    atomic_write_json(out, res)
    os.remove(part)
    print(name, {k: v for k, v in res.items() if k not in ("top1", "config")}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", help="parity | stats | comma list of arms: stock, full, oracle25, positional25, random25")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    os.makedirs(os.path.join(OUT, "eval"), exist_ok=True)
    os.makedirs(CK, exist_ok=True)
    vae, var = load(a.device)
    patch(var)
    if a.what == "parity":
        with Run("feas_var", "parity", a.device, {}):
            r = parity(var, a.device)
        atomic_write_json(os.path.join(OUT, "parity.json"), r)
        print(r)
    elif a.what == "stats":
        with Run("feas_var", "stats", a.device, {"n": 64}) as run:
            r = stats(var, a.device)
            rec = run.record()
        atomic_write_json(os.path.join(OUT, "stats.json"), {"n_images": 64, "per": r, **rec})
        print("stats done")
    else:
        for name in a.what.split(","):
            arm(var, vae, name, a.n, a.device)


if __name__ == "__main__":
    main()
