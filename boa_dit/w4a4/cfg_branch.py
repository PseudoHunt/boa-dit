"""Guidance check on PixArt-Σ (user follow-up to objective D1): H-SVDQuant-r4 base vs bf16 on the design cache (held-out calibration prompts),
prompts 0-31 x the 10 cached steps x both branches. delta_c, delta_u = base - FP eps (4 channels) per branch;
delta_g = w delta_c + (1 - w) delta_u, w = 4.5 (the model's default guidance). Reports ||delta_g||^2 / mean(||delta_c||^2,
||delta_u||^2), the correlation r of delta_c and delta_u, the shared / difference / cross split, by step.
Run: python -m boa_dit.w4a4.cfg_branch --arm hsvd_r4_b0.5 --device cuda:0
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.w4a4.common import CK, GUIDANCE, load_embeds, load_pipe
from boa_dit.w4a4.screen import install, text_for


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hsvd_r4_b0.5")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev, w = a.device, GUIDANCE
    torch.cuda.set_device(dev)
    with Run("w4a4", f"cfg_branch_{a.arm}", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        tr = pipe.transformer
        del pipe.vae
        install(tr, a.arm)
        c = torch.load(os.path.join(CK, "design_cache.pt"), map_location="cpu")
        e = load_embeds("design")
        idx = [j for j in range(len(c["t"])) if int(c["prompt"][j]) < 32]
        D = {}
        for k in range(0, len(idx), 32):
            sl = idx[k:k + 32]
            pe, pm = text_for(e, c["prompt"][sl], c["branch"][sl], dev)
            o = tr(c["latent"][sl].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][sl].to(dev),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)[0].float()[:, :4]
            ref = c["eps"][sl].to(dev).float()[:, :4]
            for j, r in enumerate(sl):
                D[(int(c["prompt"][r]), int(c["step"][r]), int(c["branch"][r]))] = (o[j] - ref[j]).double().cpu()
        rows, steps = [], sorted({s for _, s, _ in D})
        for s in steps:
            dc = torch.stack([D[(p, s, 1)] for p in range(32)])                 # branch 1 = cond (cache layout)
            du = torch.stack([D[(p, s, 0)] for p in range(32)])
            dg = w * dc + (1 - w) * du
            dd = dc - du
            en = lambda t: float(t.pow(2).sum())
            rows.append({"step": s, "g": en(dg), "c": en(dc), "u": en(du), "shared": en(du), "diff": w ** 2 * en(dd),
                         "cross": 2 * w * float((du * dd).sum()),
                         "corr": float(np.corrcoef(dc.flatten().numpy(), du.flatten().numpy())[0, 1])})
        tot = {k: sum(r[k] for r in rows) for k in ("g", "c", "u", "shared", "diff", "cross")}
        res = {"prompt_set": "design (held-out calibration prompts 32-63, rule_addendum_design.md)", "arm": a.arm, "w": w, "per_step": rows, "ratio": tot["g"] / ((tot["c"] + tot["u"]) / 2),
               "corr_mean": float(np.mean([r["corr"] for r in rows])), "raw": tot,
               "split_share": {k: tot[k] / tot["g"] for k in ("shared", "diff", "cross")},
               "build_cfg_arm": tot["g"] / ((tot["c"] + tot["u"]) / 2) >= 2}
        res.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "w4a4", "cfg_branch.json"), res)
    print(f"w {w}: ||dg||^2 {tot['g']:.4g}, ||dc||^2 {tot['c']:.4g}, ||du||^2 {tot['u']:.4g}; ratio {res['ratio']:.2f}; "
          f"r {res['corr_mean']:.2f}; split {res['split_share']}", flush=True)


if __name__ == "__main__":
    main()
