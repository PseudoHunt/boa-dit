"""Screening proxy (rule.md): one transformer forward per cached (latent, t, text, branch) triple, rel. MSE of the noise
prediction (4 eps channels) vs the cached bf16 output; fixed batch composition (32 consecutive triples in cache order).
Also the per-block table: block-l output rel. MSE vs the FP block on FP-reference block inputs, 48 held-out rows
(first 8 screening prompts x steps 0, 8, 16 x 2 branches).
  --arm fp      : the bf16 model through the same harness (the batch-layout floor)
  --arm <name>  : ckpt/w4a4/q_<name>.pt installed as QLinear layers
Run: python -m boa_dit.w4a4.screen --arm hsvd_r4_b0.5 --device cuda:0
"""
import argparse
import copy
import os

import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.w4a4.common import CK, LAYERS, block_linear, load_embeds, load_pipe
from boa_dit.w4a4.fake import QLinear, set_module

BLOCK_ROWS = dict(prompts=range(8), steps=(0, 8, 16))


def install(tr, name):
    qs = torch.load(os.path.join(CK, f"q_{name}.pt"), map_location="cpu")
    dev = next(tr.parameters()).device
    for key, q in qs.items():
        l, ln = key.split(".", 1)
        blk = tr.transformer_blocks[int(l)]
        lin = block_linear(blk, ln)
        g = lambda k: None if q[k] is None else q[k].to(dev)
        set_module(blk, ln, QLinear(lin, g("R"), g("L1"), g("L2"), g("d"), act_shift=g("act_shift") if "act_shift" in q else None))
    return len(qs)


def text_for(e, prompts, branch, dev):
    pe = torch.where(branch.view(-1, 1, 1).bool(), e["prompt_embeds"][prompts], e["negative_prompt_embeds"].expand(len(prompts), -1, -1))
    pm = torch.where(branch.view(-1, 1).bool(), e["prompt_attention_mask"][prompts], e["negative_prompt_attention_mask"].expand(len(prompts), -1))
    return pe.to(dev, torch.bfloat16), pm.to(dev)


@torch.no_grad()
def run_screen(tr, fp_blocks, arm, dev, bs=32, n_q=0):
    """Proxy + per-block table for transformer `tr` (already quantized, or FP). fp_blocks: FP copies of the 28 blocks,
    or None (no per-block table, e.g. the FP floor)."""
    c = torch.load(os.path.join(CK, "screen_cache.pt"), map_location="cpu")
    e = load_embeds("screen")
    R = len(c["t"])
    rel = torch.zeros(R)
    rel_b1 = torch.zeros(R)
    from boa_dit.w4a4.band_gain import band_masks
    m1 = band_masks()[0].to(dev)
    dt = next(tr.parameters()).dtype
    for i in range(0, R, bs):
        sl = slice(i, min(R, i + bs))
        pe, pm = text_for(e, c["prompt"][sl], c["branch"][sl], dev)
        o = tr(c["latent"][sl].to(dev, dt), encoder_hidden_states=pe.to(dt), encoder_attention_mask=pm, timestep=c["t"][sl].to(dev),
               added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0]
        ref = c["eps"][sl].to(dev).float()[:, :4]
        q = o.float()[:, :4]
        rel[sl] = ((q - ref).flatten(1).pow(2).sum(1) / ref.flatten(1).pow(2).sum(1)).cpu()
        P = torch.fft.fft2((q - ref).double()).abs().pow(2)                  # band-1 proxy: error restricted to band 1
        rel_b1[sl] = (P[..., m1].sum(-1).sum(1) / (128 * 128) / ref.double().flatten(1).pow(2).sum(1)).float().cpu()
    res = {"arm": arm, "n_quantized_layers": n_q, "n_triples": R, "rel_mse_mean": float(rel.mean()),
           "rel_mse_per_triple": rel.tolist(), "rel_mse_band1_mean": float(rel_b1.mean()), "rel_mse_band1_per_triple": rel_b1.tolist(),
           "by_step": {int(s): float(rel[c["step"] == s].mean()) for s in sorted(set(c["step"].tolist()))}}
    if fp_blocks is not None:                                                # per-block table on FP-reference inputs
        sel = [j for j in range(R) if int(c["prompt"][j]) in BLOCK_ROWS["prompts"] and int(c["step"][j]) in BLOCK_ROWS["steps"]]
        ins = {}
        hooks = [b.register_forward_pre_hook(lambda m, args, kw, l=l: ins.__setitem__(l, (args[0] if args else kw["hidden_states"], kw)),
                                             with_kwargs=True) for l, b in enumerate(fp_blocks)]
        q_blocks = list(tr.transformer_blocks)
        num, den = torch.zeros(28), torch.zeros(28)
        try:
            for l in range(28):
                tr.transformer_blocks[l] = fp_blocks[l]
            for k in range(0, len(sel), 16):
                idx = sel[k:k + 16]
                pe, pm = text_for(e, c["prompt"][idx], c["branch"][idx], dev)
                ins.clear()
                tr(c["latent"][idx].to(dev, dt), encoder_hidden_states=pe.to(dt), encoder_attention_mask=pm, timestep=c["t"][idx].to(dev),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
                for l in range(28):
                    x, kw = ins[l]
                    kw = {k2: v for k2, v in kw.items() if k2 != "hidden_states"}
                    yf = fp_blocks[l](x, **kw).float()
                    yq = q_blocks[l](x, **kw).float()
                    num[l] += (yq - yf).pow(2).sum().cpu()
                    den[l] += yf.pow(2).sum().cpu()
        finally:
            for l in range(28):
                tr.transformer_blocks[l] = q_blocks[l]
            for h in hooks:
                h.remove()
        res["per_block_rel_mse"] = (num / den).tolist()
        res["per_block_rows"] = len(sel)
    return res


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--tag", default="")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    out = os.path.join(RESULTS, "w4a4", "screen", f"{a.arm}{a.tag}.json")
    with Run("w4a4", f"screen_{a.arm}{a.tag}", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        tr = pipe.transformer
        del pipe.vae
        fp_blocks = [copy.deepcopy(b) for b in tr.transformer_blocks] if a.arm != "fp" else None
        n_q = install(tr, a.arm) if a.arm != "fp" else 0
        res = run_screen(tr, fp_blocks, a.arm, dev, a.bs, n_q)
        res.update(run.record())
    atomic_write_json(out, res)
    print(f"{a.arm}{a.tag}: proxy {res['rel_mse_mean']:.5f}", flush=True)


if __name__ == "__main__":
    main()
