"""Part B: is the early routing error shared across samples, and does adding the shared part back work? See rule.md.

B1 (calibration seeds 1000-1031, FP trajectory from capture_block0_inputs): block inputs propagated through FP blocks and,
separately, through W4 blocks. At blocks 0-4 and 13, buckets 0-2 and 7-9, per (image, step), per head:
  L_FP = FP qkv on the FP input; L_q = W4 qkv on the FP input; L_q' = W4 block's qkv on the quantized-prefix input
  D = rowcenter(L_FP - L_q·); D-bar = mean over the bucket's (image, step) samples; shared = ||D-bar||^2 / mean ||D||^2
  (also per step, then averaged). Structure of D-bar: SVD energy at rank 4/8/16/32, offset-kernel cosine, branch cosine.
  -> results/prior/b1.json, ckpt/prior/Dbar.pt (fp32)
B2: W4 model, blocks 0-4, steps < 14: logits = L_q' + D-bar[block, head, bucket, branch] (deployed D-bar; variants:
  FP-input D-bar, deployed D-bar at rank 16). Latent MSE seeds 0-127; FID-2k + labels for the main arm.
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.attn_share.gate1 import offsets
from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import load_dit, load_quant_state, make_diffusion
from boa_dit.prior.part_a import OUT, W4
from boa_dit.quantize import CALIB_SEEDS
from boa_dit.sample import CTX

BLOCKS = (0, 1, 2, 3, 4, 13)
BUCKETS = (0, 1, 2, 7, 8, 9)
H, HD, C = 16, 72, 1152
DBAR = os.path.join(CKPT_DIR, "prior", "Dbar.pt")


def logits(x, W, b):
    """x [N, 256, C] fp32 -> row-centered-ready logits [N, H, 256, 256] = q k^T / sqrt(72)."""
    qk = x @ W[:2 * C].T + b[:2 * C]
    q = qk[..., :C].reshape(x.shape[0], 256, H, HD).transpose(1, 2)
    k = qk[..., C:].reshape(x.shape[0], 256, H, HD).transpose(1, 2)
    return (q @ k.transpose(-1, -2)) * HD ** -0.5


def rowc(D):
    return D - D.mean(-1, keepdim=True)


@torch.no_grad()
def b1(dev):
    fp, q = load_dit(dev), load_dit(dev)
    load_quant_state(q, W4)
    diff = make_diffusion(50)
    S, N = 50, len(CALIB_SEEDS)
    acc = {}                                   # (def, block, bucket, branch) -> dict of accumulators
    with Run("prior", "b1", dev, {}) as run:
        X, Cc = capture_block0_inputs(fp, diff, CALIB_SEEDS, 1.5, dev)
        Xq = X.clone()
        for l in range(max(BLOCKS) + 1):
            fb, qb = fp.blocks[l], q.blocks[l]
            cap = {}
            hf = fb.attn.qkv.register_forward_pre_hook(lambda m, i: cap.__setitem__("f", i[0].float()))
            hq = qb.attn.qkv.register_forward_pre_hook(lambda m, i: cap.__setitem__("q", i[0].float()))
            Wf, bf = fb.attn.qkv.weight.float(), fb.attn.qkv.bias.float()
            Wq, bq = qb.attn.qkv.weight.float(), qb.attn.qkv.bias.float()
            try:
                for s in range(S):
                    for br in range(2):
                        r0 = s * 2 * N + br * N
                        with torch.autocast("cuda", dtype=torch.float16):
                            yf = fb(X[r0:r0 + N].float(), Cc[r0:r0 + N])
                            yq = qb(Xq[r0:r0 + N].float(), Cc[r0:r0 + N])
                        X[r0:r0 + N], Xq[r0:r0 + N] = yf.half(), yq.half()
                        bk = s // 5
                        if l not in BLOCKS or bk not in BUCKETS:
                            continue
                        with torch.autocast("cuda", enabled=False):
                            Lfp = logits(cap["f"], Wf, bf)
                            for dname, Lq in (("fpin", logits(cap["f"], Wq, bq)), ("deployed", logits(cap["q"], Wq, bq))):
                                D = rowc(Lfp - Lq)                                      # [N, H, 256, 256]
                                key = (dname, l, bk, br)
                                a = acc.setdefault(key, {"sum": torch.zeros(H, 256, 256, device=dev), "sq": torch.zeros(H, device=dev),
                                                         "n": 0, "step_shared_num": torch.zeros(H, device=dev),
                                                         "step_shared_den": torch.zeros(H, device=dev)})
                                ds = D.sum(0)
                                a["sum"] += ds
                                sq = (D * D).sum((-1, -2))                              # [N, H]
                                a["sq"] += sq.sum(0)
                                a["n"] += N
                                a["step_shared_num"] += ((ds / N) ** 2).sum((-1, -2))
                                a["step_shared_den"] += sq.mean(0)
                                del D
            finally:
                hf.remove(); hq.remove()
            print(f"block {l} done", flush=True)
        rec = run.record()
    offs = offsets(16).to(dev)
    flat = offs.flatten()
    cnt = torch.zeros(961, device=dev).index_add_(0, flat, torch.ones(flat.shape[0], device=dev))
    res, dbar = {"blocks": BLOCKS, "buckets": BUCKETS, "per": {}}, {}
    for (dname, l, bk, br), a in acc.items():
        Db = a["sum"] / a["n"]
        dbar[(dname, l, bk, br)] = Db.cpu()
        shared = (Db * Db).sum((-1, -2)) / (a["sq"] / a["n"])
        shared_step = a["step_shared_num"] / a["step_shared_den"]
        sv = torch.linalg.svdvals(Db.double())                                         # [H, 256]
        e = (sv ** 2).cumsum(-1) / (sv ** 2).sum(-1)
        kap = torch.zeros(H, 961, device=dev).index_add_(1, flat, Db.reshape(H, -1)) / cnt
        R = kap[:, offs]
        kc = (R * Db).sum((-1, -2)) / (R.norm(dim=(-1, -2)) * Db.norm(dim=(-1, -2)))
        res["per"][f"{dname}.b{l}.k{bk}.br{br}"] = {
            "def": dname, "block": l, "bucket": bk, "branch": br, "shared": shared.tolist(), "shared_median": float(shared.median()),
            "shared_per_step_median": float(shared_step.median()),
            "energy_rank": {str(r): float(e[:, r - 1].median()) for r in (4, 8, 16, 32)},
            "kernel_cos_median": float(kc.median()), "rms_D": float((a["sq"] / a["n"] / 256 ** 2).sqrt().median())}
    for dname in ("fpin", "deployed"):
        for l in BLOCKS:
            for bk in BUCKETS:
                c, u = dbar[(dname, l, bk, 0)], dbar[(dname, l, bk, 1)]
                bc = (c * u).sum((-1, -2)) / (c.norm(dim=(-1, -2)) * u.norm(dim=(-1, -2)))
                res["per"][f"{dname}.b{l}.k{bk}.br0"]["branch_cos_median"] = float(bc.median())
    atomic_torch_save({f"{d}.b{l}.k{bk}.br{br}": v for (d, l, bk, br), v in dbar.items()}, DBAR)
    atomic_write_json(os.path.join(OUT, "b1.json"), {**res, **rec})
    for k, v in res["per"].items():
        if v["branch"] == 0 and v["def"] == "deployed":
            print(k, round(v["shared_median"], 3), round(v["shared_per_step_median"], 3), v["energy_rank"]["16"], round(v["kernel_cos_median"], 3), flush=True)


# ---------------------------------------------------------------- B2
def patch_b2(model, dbar, variant, dev):
    """Blocks 0-4, steps < 14: softmax(L_q' + D-bar) V with the model's own (W4) q, k, v."""
    import torch.nn.functional as F
    D = {}
    for l in range(5):
        for bk in (0, 1, 2):
            for br in (0, 1):
                d = dbar[f"{'fpin' if variant == 'fpin' else 'deployed'}.b{l}.k{bk}.br{br}"].to(dev)
                if variant == "r16":
                    U, S_, Vh = torch.linalg.svd(d.double())
                    d = ((U[..., :16] * S_[..., None, :16]) @ Vh[..., :16, :]).float()
                D[(l, bk, br)] = d
    for l in range(5):
        attn = model.blocks[l].attn
        orig = attn.forward

        def fwd(x, attn_mask=None, is_causal=False, l=l, attn=attn, orig=orig):
            s = CTX.step
            if s is None or s >= 14:
                return orig(x)
            B, N, _ = x.shape
            qkv = attn.qkv(x).reshape(B, N, 3, H, HD).permute(2, 0, 3, 1, 4)
            q, k, v = qkv.unbind(0)
            with torch.autocast("cuda", enabled=False):
                L = (q.float() @ k.float().transpose(-1, -2)) * attn.scale
                n = CTX.n
                L[:n] += D[(l, s // 5, 0)]
                L[n:] += D[(l, s // 5, 1)]
                P = torch.softmax(L, -1)
            out = (P.to(v.dtype) @ v).transpose(1, 2).reshape(B, N, C)
            return attn.proj(out)
        attn.forward = fwd
    return model


@torch.no_grad()
def b2(variant, dev, with_fid):
    from boa_dit.eval_latent_mse import per_image
    from boa_dit.sample import sample_seeds
    name = {"deployed": "B2", "fpin": "B2_fpin", "r16": "B2_r16"}[variant]
    out = os.path.join(OUT, "eval", f"latent_{name}.json")
    dbar = torch.load(DBAR, map_location="cpu")
    if not os.path.exists(out):
        xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:128]
        with Run("prior", f"latent_{name}", dev, {"variant": variant}) as run:
            m = load_dit(dev)
            load_quant_state(m, W4)
            patch_b2(m, dbar, variant, dev)
            x = sample_seeds(m, make_diffusion(50), list(range(128)), 1.5, 32, log=False)
            rel = per_image(x, xf)[0]
            res = {"arm": name, "seeds": "0-127", "rel_mse_mean": float(rel.mean()), "rel_mse_per_image": rel.tolist(), **run.record()}
        atomic_write_json(out, res)
        print(name, res["rel_mse_mean"], flush=True)
    if with_fid and not os.path.exists(os.path.join(OUT, "eval", f"fid_{name}.json")):
        from boa_dit.prior import part_a
        orig_build = part_a.build

        def build(arm, dev_):
            m = load_dit(dev_)
            load_quant_state(m, W4)
            return patch_b2(m, dbar, variant, dev_), []
        part_a.build = build
        part_a.fid(name, dev)
        part_a.build = orig_build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["b1", "b2"])
    ap.add_argument("--variants", default="deployed,fpin,r16")
    ap.add_argument("--fid", action="store_true", help="FID-2k + labels for the deployed (main) B2 arm")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    if a.what == "b1":
        b1(a.device)
    else:
        for v in a.variants.split(","):
            b2(v, a.device, a.fid and v == "deployed")


if __name__ == "__main__":
    main()
