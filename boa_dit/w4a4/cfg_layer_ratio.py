"""Layer-level guidance ratio on the corrected base (user check before the CFG arm), design cache (held-out calibration
prompts). Per quantized layer: E = D^-1 (R^ + L1 L2) - W^T (the base's effective weight residual, c_in x c_out);
ratio = ||E v||^2 / mean(||E x_c||^2, ||E x_u||^2), v = 4.5 x_c - 3.5 x_u, over paired FP-model layer inputs
(same prompt, step, token; 256 random tokens per row, 32 prompts x 10 steps). This is what H_cfg assumes; the network-level
ratio (14.0, cfg_branch.json) is what it should produce. Run: python -m boa_dit.w4a4.cfg_layer_ratio --device cuda:1
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.w4a4.common import CK, GUIDANCE, LAYERS, block_linear, load_embeds, load_pipe
from boa_dit.w4a4.screen import text_for

ARM = "hsvd_r4_b0.5_kv16_u2_rs"
N_TOK = 256


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev, w = a.device, GUIDANCE
    torch.cuda.set_device(dev)
    qs = torch.load(os.path.join(CK, f"q_{a.arm}.pt"), map_location="cpu")
    c = torch.load(os.path.join(CK, "design_cache.pt"), map_location="cpu")
    e = load_embeds("design")
    with Run("w4a4", f"cfg_layer_ratio_{a.arm}", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        tr = pipe.transformer
        del pipe.vae
        E = {}
        for key, q in qs.items():
            l, ln = key.split(".", 1)
            W = block_linear(tr.transformer_blocks[int(l)], ln).weight.data.float()      # [c_out, c_in]
            d = q["d"].float().to(dev)
            Weff = (q["R"].float().to(dev).T + q["L1"].float().to(dev) @ q["L2"].float().to(dev)) / d[:, None]
            E[key] = (Weff - W.T)                                                          # [c_in, c_out]
        acc = {k: torch.zeros(3, dtype=torch.float64) for k in E}                          # sum ||E x_c||^2, ||E x_u||^2, ||E v||^2
        g = torch.Generator().manual_seed(0)
        cur = {}
        hooks = []
        for key in E:
            l, ln = key.split(".", 1)

            def h(mod, inp, key=key):
                x = inp[0]
                n = x.shape[0] // 2                                                       # rows [0,n) uncond, [n,2n) cond
                xt = x[:, cur["tok"]].float() if x.shape[1] == 4096 else x.float()
                xu, xc = xt[:n], xt[n:]
                v = w * xc + (1 - w) * xu
                for i, z in enumerate((xc, xu, v)):
                    acc[key][i] += float((z @ E[key]).pow(2).sum())
            hooks.append(block_linear(tr.transformer_blocks[int(l)], ln).register_forward_pre_hook(h))
        steps = sorted(set(c["step"].tolist()))
        for s in steps:
            for p0 in range(0, 32, 16):
                ps = list(range(p0, p0 + 16))
                idx = {(int(c["prompt"][j]), int(c["branch"][j])): j for j in range(len(c["t"])) if int(c["step"][j]) == s}
                rows = [idx[(p, 0)] for p in ps] + [idx[(p, 1)] for p in ps]
                cur["tok"] = torch.randperm(4096, generator=g)[:N_TOK].to(dev)
                pe, pm = text_for(e, c["prompt"][rows], c["branch"][rows], dev)
                tr(c["latent"][rows].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][rows].to(dev),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
        for hk in hooks:
            hk.remove()
        per = {k: {"ratio": float(v[2] / ((v[0] + v[1]) / 2)), "err_c": float(v[0]), "err_u": float(v[1]), "err_v": float(v[2])}
               for k, v in acc.items()}
        types = sorted({k.split(".", 1)[1] for k in per}, key=LAYERS.index)
        tot_v = sum(v["err_v"] for v in per.values())
        r = np.array([v["ratio"] for v in per.values()])
        by_type = {t: {"median": float(np.median([per[f"{l}.{t}"]["ratio"] for l in range(28)])),
                       "min": float(min(per[f"{l}.{t}"]["ratio"] for l in range(28))),
                       "max": float(max(per[f"{l}.{t}"]["ratio"] for l in range(28))),
                       "share_of_guided_error": sum(per[f"{l}.{t}"]["err_v"] for l in range(28)) / tot_v} for t in types}
        by_block = {l: float(np.median([per[f"{l}.{t}"]["ratio"] for t in types])) for l in range(28)}
        ge4 = [k for k, v in per.items() if v["ratio"] >= 4]
        res = {"prompt_set": "design (held-out calibration prompts 32-63, rule_addendum_design.md)", "arm": a.arm, "w": w,
               "per_layer": per, "by_type": by_type, "by_block_median": by_block,
               "distribution": {"median": float(np.median(r)), "p10": float(np.percentile(r, 10)), "p90": float(np.percentile(r, 90)),
                                "frac_ge_4": float(np.mean(r >= 4)), "frac_le_2": float(np.mean(r <= 2)),
                                "share_of_guided_error_in_ratio_ge_4": sum(per[k]["err_v"] for k in ge4) / tot_v},
               "layers_ratio_ge_4": ge4, **run.record()}
    atomic_write_json(os.path.join(RESULTS, "w4a4", "cfg_layer_ratio.json"), res)
    print("distribution", res["distribution"], flush=True)
    for t, v in by_type.items():
        print(f"{t:16s} median {v['median']:6.2f} [{v['min']:.2f}, {v['max']:.2f}]  share of guided error {v['share_of_guided_error']:.2f}", flush=True)


if __name__ == "__main__":
    main()
