"""Step-coherent vs incoherent error (user follow-up to the K/V-16-bit finding): on the screening cache (64 prompts x 10
steps x 2 branches), per-step errors e_{p,s} = arm - bf16 eps (4 channels). For the guided error
delta_g = 4.5 delta_c - 3.5 delta_u and per branch:
  C = sum_p n_s ||mean_s e_{p,s}||^2   (the part common to all steps of a prompt)
  I = sum_{p,s} ||e_{p,s} - mean_s e_{p,s}||^2   (C + I = total)
Arms: the H-SVDQuant-r4 base, and the base with cross-attn K/V at 16-bit (evaluation-time, as in the single-change delta).
Run: python -m boa_dit.w4a4.coherence --device cuda:0
"""
import argparse
import copy
import os

import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.w4a4.common import CK, GUIDANCE, load_embeds, load_pipe
from boa_dit.w4a4.screen import install, text_for


def ci(E):
    """E: [P, S, ...] errors -> (C, I, total)."""
    m = E.mean(1, keepdim=True)
    C = float(E.shape[1] * m.pow(2).sum())
    I = float((E - m).pow(2).sum())
    return C, I, C + I


@torch.no_grad()
def errors(tr, c, e, dev):
    R = len(c["t"])
    out = torch.zeros(R, 4, 128, 128, dtype=torch.float32)
    for i in range(0, R, 32):
        sl = slice(i, min(R, i + 32))
        pe, pm = text_for(e, c["prompt"][sl], c["branch"][sl], dev)
        o = tr(c["latent"][sl].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][sl].to(dev),
               added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0].float()[:, :4]
        out[sl] = (o - c["eps"][sl].to(dev).float()[:, :4]).cpu()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev, w = a.device, GUIDANCE
    torch.cuda.set_device(dev)
    c = torch.load(os.path.join(CK, "screen_cache.pt"), map_location="cpu")
    e = load_embeds("screen")
    P, S = int(c["prompt"].max()) + 1, len(set(c["step"].tolist()))
    res = {"definition": "C = sum_p n_steps ||mean_s e||^2, I = sum ||e - mean_s e||^2, over 64 prompts x 10 steps", "arms": {}}
    with Run("w4a4", "coherence", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        del pipe.vae
        base_tr = pipe.transformer
        fp_blocks = [copy.deepcopy(b) for b in base_tr.transformer_blocks]
        for arm in ("base", "base_kv16"):
            for l in range(28):
                base_tr.transformer_blocks[l] = copy.deepcopy(fp_blocks[l])
            install(base_tr, "hsvd_r4_b0.5")
            if arm == "base_kv16":
                for l in range(28):
                    blk = base_tr.transformer_blocks[l]
                    blk.attn2.to_k, blk.attn2.to_v = fp_blocks[l].attn2.to_k, fp_blocks[l].attn2.to_v
            err = errors(base_tr, c, e, dev)
            idx = {(int(c["prompt"][j]), int(c["step"][j]), int(c["branch"][j])): j for j in range(len(c["t"]))}
            steps = sorted(set(c["step"].tolist()))
            ec = torch.stack([torch.stack([err[idx[(p, s, 1)]] for s in steps]) for p in range(P)]).double()
            eu = torch.stack([torch.stack([err[idx[(p, s, 0)]] for s in steps]) for p in range(P)]).double()
            eg = w * ec + (1 - w) * eu
            r = {}
            for lab, E in (("guided", eg), ("cond", ec), ("uncond", eu)):
                C, I, T = ci(E)
                r[lab] = {"C": C, "I": I, "total": T, "C_over_I": C / I, "C_share": C / T}
            res["arms"][arm] = r
            print(arm, {k: (f"C {v['C']:.4g}", f"I {v['I']:.4g}", f"C/I {v['C_over_I']:.3f}") for k, v in r.items()}, flush=True)
        for lab in ("guided", "cond", "uncond"):
            b, k = res["arms"]["base"][lab], res["arms"]["base_kv16"][lab]
            res.setdefault("ratios_base_over_kv16", {})[lab] = {"C": b["C"] / k["C"], "I": b["I"] / k["I"], "total": b["total"] / k["total"]}
        res.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "w4a4", "coherence.json"), res)
    print("base / kv16:", res["ratios_base_over_kv16"], flush=True)


if __name__ == "__main__":
    main()
