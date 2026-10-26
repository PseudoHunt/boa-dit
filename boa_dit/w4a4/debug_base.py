"""W4A4 base debugging (user's checklist), on the H-SVDQuant-r4 base (ckpt/w4a4/q_hsvd_r4_b0.5.pt).
  identity : per layer, on real inputs (screening rows, prompts 0-7, step 8, both branches), fp32:
             y_fold = (x/d)(W~ - L1 L2)^T... i.e. the stored fold with NO quantization anywhere vs x W^T.
  split    : W4A16 (stored R^ + branch, activations unquantized) and W16A4 (smoothed unquantized weights W~, no branch,
             activations quantized on X~): screen proxy + MJHQ-64 PSNR / LPIPS vs bf16.
  qsnr     : activation QSNR = 10 log10(||x~||^2 / ||x~ - Q_A(x~)||^2) per layer type (on X~ = X/d, as the quantizer
             sees it; and on raw X for reference), same rows.
Run: python -m boa_dit.w4a4.debug_base --what identity,qsnr,split --device cuda:1
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.w4a4.common import CK, LAYERS, block_linear, load_embeds, load_pipe
from boa_dit.w4a4.fake import QLinear, q_act, set_module
from boa_dit.w4a4.screen import run_screen, text_for

BASE = "hsvd_r4_b0.5"
OUT = os.path.join(RESULTS, "w4a4", "debug")


def rows_sel(c):
    return [j for j in range(len(c["t"])) if int(c["prompt"][j]) < 8 and int(c["step"][j]) == 8]


@torch.no_grad()
def capture_inputs(tr, c, e, dev):
    """Per layer: list of input tensors from one forward over the selected rows (FP model)."""
    ins = {}
    hooks = []
    for l, blk in enumerate(tr.transformer_blocks):
        for ln in LAYERS:
            def h(mod, inp, key=(l, ln)):
                ins[key] = inp[0].detach()
            hooks.append(block_linear(blk, ln).register_forward_pre_hook(h))
    sel = rows_sel(c)
    pe, pm = text_for(e, c["prompt"][sel], c["branch"][sel], dev)
    tr(c["latent"][sel].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][sel].to(dev),
       added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
    for h in hooks:
        h.remove()
    return ins


def install_variant(tr, qs, variant, dev):
    for key, q in qs.items():
        l, ln = key.split(".", 1)
        blk = tr.transformer_blocks[int(l)]
        lin = block_linear(blk, ln)
        d = q["d"].to(dev).float()
        if variant == "w4a16":
            new = QLinear(lin, q["R"].to(dev), q["L1"].to(dev), q["L2"].to(dev), d, act_quant=False)
        elif variant == "kv16":                                                # single change: cross-attn K/V 16-bit
            if ln in ("attn2.to_k", "attn2.to_v"):
                continue
            new = QLinear(lin, q["R"].to(dev), q["L1"].to(dev), q["L2"].to(dev), d)
        elif variant == "u2":                                                  # single change: shifted unsigned fc2 input
            from boa_dit.w4a4.fake import GELU_MIN
            new = QLinear(lin, q["R"].to(dev), q["L1"].to(dev), q["L2"].to(dev), d,
                          act_shift=(GELU_MIN / d) if ln == "ff.net.2" else None)
        elif variant == "w16a4":
            Wt = lin.weight.data.float() * d[None, :]                          # W~ in nn.Linear layout: W diag(d)
            new = QLinear(lin, Wt, None, None, d, act_quant=True)
        else:
            raise ValueError(variant)
        set_module(blk, ln, new)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="identity,qsnr,split")
    ap.add_argument("--n_img", type=int, default=64)
    ap.add_argument("--variants", default="w4a16,w16a4")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    qs = torch.load(os.path.join(CK, f"q_{BASE}.pt"), map_location="cpu")
    c = torch.load(os.path.join(CK, "screen_cache.pt"), map_location="cpu")
    e = load_embeds("screen")
    with Run("w4a4", f"debug_base_{a.what.replace(',', '+')}", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        tr = pipe.transformer
        whats = a.what.split(",")
        if "identity" in whats or "qsnr" in whats:
            idr, qsr = {}, {}

            def qsnr(Z):
                return float(10 * torch.log10(Z.pow(2).sum() / (Z - q_act(Z)).pow(2).sum()))
            hooks = []
            for l, blk in enumerate(tr.transformer_blocks):
                for ln in LAYERS:
                    def h(mod, inp, key=(l, ln), lin=block_linear(blk, ln)):
                        q = qs[f"{key[0]}.{key[1]}"]
                        X = inp[0].detach().float().reshape(-1, inp[0].shape[-1])
                        W = lin.weight.data.float()
                        d = q["d"].to(dev).float()
                        Xt = X / d
                        if "identity" in whats:
                            L1, L2 = q["L1"].to(dev).float(), q["L2"].to(dev).float()
                            y_ref = X @ W.T
                            y_fold = Xt @ ((W * d[None, :]).T - L1 @ L2) + (Xt @ L1) @ L2
                            idr[f"{key[0]}.{key[1]}"] = float((y_fold - y_ref).norm() / y_ref.norm())
                        if "qsnr" in whats:
                            qsr[f"{key[0]}.{key[1]}"] = {"smoothed": qsnr(Xt), "raw": qsnr(X)}
                    hooks.append(block_linear(blk, ln).register_forward_pre_hook(h))
            sel = rows_sel(c)[:8]
            pe, pm = text_for(e, c["prompt"][sel], c["branch"][sel], dev)
            tr(c["latent"][sel].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][sel].to(dev),
               added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
            for hk in hooks:
                hk.remove()
            res = {}
            if idr:
                v = np.array(list(idr.values()))
                res["identity"] = {"max_rel_err": float(v.max()), "median_rel_err": float(np.median(v)), "per_layer": idr,
                                   "worst": max(idr, key=idr.get), "pass_fp32": bool(v.max() < 1e-4)}
                print(f"identity: max rel err {v.max():.2e} (worst {res['identity']['worst']}), median {np.median(v):.2e}", flush=True)
            if qsr:
                by = {ln: {"smoothed_mean_db": float(np.mean([qsr[f'{l}.{ln}']['smoothed'] for l in range(28)])),
                           "smoothed_min_db": float(np.min([qsr[f'{l}.{ln}']['smoothed'] for l in range(28)])),
                           "raw_mean_db": float(np.mean([qsr[f'{l}.{ln}']['raw'] for l in range(28)]))} for ln in LAYERS}
                res["qsnr"] = {"by_type": by, "per_layer": qsr, "types_under_20db": [ln for ln, v in by.items() if v["smoothed_mean_db"] < 20]}
                for ln, v in by.items():
                    print(f"QSNR {ln:16s} smoothed mean {v['smoothed_mean_db']:5.1f} dB (min {v['smoothed_min_db']:5.1f}), raw {v['raw_mean_db']:5.1f}", flush=True)
            atomic_write_json(os.path.join(OUT, "identity_qsnr.json"), res)
        if "split" in whats:
            from boa_dit.w4a4 import gen
            import copy
            fp_blocks = [copy.deepcopy(b) for b in tr.transformer_blocks]
            for variant in a.variants.split(","):
                for l in range(28):
                    tr.transformer_blocks[l] = copy.deepcopy(fp_blocks[l])
                install_variant(tr, qs, variant, dev)
                name = f"{BASE}_{variant}"
                r = run_screen(tr, None, name, dev)
                atomic_write_json(os.path.join(RESULTS, "w4a4", "screen", f"{name}.json"), r)
                gen.generate(name, a.n_img, dev, pipe=pipe)
                psnr, lp = gen.metrics(name, a.n_img, dev)
                out = {"variant": variant, "proxy": r["rel_mse_mean"], "psnr_mean": float(np.mean(psnr)), "lpips_mean": float(np.mean(lp)),
                       "n": a.n_img, "psnr_per_image": psnr, "lpips_per_image": lp}
                atomic_write_json(os.path.join(OUT, f"split_{variant}.json"), out)
                print(f"{variant}: proxy {r['rel_mse_mean']:.5f}, MJHQ-{a.n_img} PSNR {out['psnr_mean']:.2f} dB, LPIPS {out['lpips_mean']:.3f}", flush=True)
            import json
            full = json.load(open(os.path.join(RESULTS, "w4a4", "eval", f"img256_{BASE}.json")))
            print(f"full W4A4 base on the same {a.n_img}: PSNR {np.mean(full['psnr_per_image'][:a.n_img]):.2f}, "
                  f"LPIPS {np.mean(full['lpips_per_image'][:a.n_img]):.3f}", flush=True)
        run.record()


if __name__ == "__main__":
    main()
