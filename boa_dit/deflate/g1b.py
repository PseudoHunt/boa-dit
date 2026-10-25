"""G1 write-up corrections (reviewer item 4). Same FP calibration pass as g1.py (the Grams and token means were not stored
there, so the pass is rerun; deterministic), with:
  a. fraction of tr(H) captured by the top-8 eigenvectors of H (tr(H_mean) at r = 8 is g1.json's top8_frac)
  b. qkv / fc1 (the modulated inputs): cos(xbar_s, m_s) with m_s = token mean of LN(x) * (1 + scale) = xbar_s - shift;
     all layers: fraction of ||xbar_s||^2 in its top-8 and top-32 channels (per sample, then median)
  c. fc2 in blocks 0-3 and 27: top eigenvector of H_var; its energy in the single largest channel and the top 8
-> results/deflate/g1b.json
"""
import os

import torch
import torch.nn.functional as F

from boa_dit.calib import capture_block0_inputs
from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.deflate.g1 import B, CFG, eigh64, predicted
from boa_dit.model import LINEARS, block_linear, load_dit, make_diffusion
from boa_dit.quantize import CALIB_SEEDS

FC2_BLOCKS = (0, 1, 2, 3, 27)


@torch.no_grad()
def main(dev="cuda:0"):
    torch.cuda.set_device(dev)
    wc, wu = CFG ** 2, (CFG - 1) ** 2
    wc, wu = wc / (wc + wu), wu / (wc + wu)
    res = {"layers": {}}
    with Run("deflate", "g1b", dev, {}) as run:
        model, diffusion = load_dit(dev), make_diffusion(50)
        S, N = diffusion.num_timesteps, len(CALIB_SEEDS)
        X, C = capture_block0_inputs(model, diffusion, CALIB_SEEDS, CFG, dev)
        for l, blk in enumerate(model.blocks):
            st = {ln: {"G": None, "Gm": None, "cos_m": [], "c8": [], "c32": []} for ln in LINEARS}
            cur = {}

            def hook(ln):
                @torch.autocast("cuda", enabled=False)
                def h(mod, inp):
                    x = inp[0].float()
                    xb = x.mean(1)
                    xf = x.reshape(-1, x.shape[-1])
                    a, w = st[ln], (wc if cur["br"] == 0 else wu)
                    g, gm = w * (xf.T @ xf), w * 256.0 * (xb.T @ xb)
                    a["G"] = g if a["G"] is None else a["G"] + g
                    a["Gm"] = gm if a["Gm"] is None else a["Gm"] + gm
                    e = (xb * xb).sort(-1, descending=True).values
                    tot = e.sum(-1)
                    a["c8"].append((e[:, :8].sum(-1) / tot).cpu())
                    a["c32"].append((e[:, :32].sum(-1) / tot).cpu())
                    if ln in ("attn.qkv", "mlp.fc1"):
                        a["cos_m"].append(F.cosine_similarity(xb, xb - cur["p"][ln], dim=-1).cpu())
                return h
            hs = [block_linear(blk, ln).register_forward_pre_hook(hook(ln)) for ln in LINEARS]
            try:
                for s in range(S):
                    for br in range(2):
                        r0 = s * 2 * N + br * N
                        cur.update(br=br, p=predicted(blk, C[r0:r0 + N]))
                        with torch.autocast("cuda", dtype=torch.float16):
                            y = blk(X[r0:r0 + N].float(), C[r0:r0 + N])
                        X[r0:r0 + N] = y.to(torch.float16)
            finally:
                for h in hs:
                    h.remove()
            for ln in LINEARS:
                a = st[ln]
                H, Hm = a["G"], a["Gm"]                                               # common normalization cancels
                eh = eigh64(H)[0].flip(0).clamp_min(0)
                c8, c32 = torch.cat(a["c8"]), torch.cat(a["c32"])
                r = {"top8_frac_H": float(eh[:8].sum() / eh.sum()),
                     "xbar_energy_top8_channels_median": float(c8.median()), "xbar_energy_top32_channels_median": float(c32.median())}
                if a["cos_m"]:
                    cm = torch.cat(a["cos_m"])
                    r["cos_xbar_ln_scale_mean_median"] = float(cm.median())
                    r["cos_xbar_ln_scale_mean_p10"] = float(torch.quantile(cm, 0.1))
                if ln == "mlp.fc2" and l in FC2_BLOCKS:
                    v = eigh64(H - Hm)[1][:, -1]
                    e = (v * v).sort(descending=True).values
                    r["hvar_top_eigvec_top1_channel_frac"] = float(e[0])
                    r["hvar_top_eigvec_top8_channel_frac"] = float(e[:8].sum())
                    r["hvar_top_eigvec_top1_channel"] = int((v * v).argmax())
                res["layers"][f"b{l}.{ln}"] = r
            print(l, {ln: {k: round(v, 3) for k, v in res["layers"][f"b{l}.{ln}"].items() if isinstance(v, float)}
                      for ln in ("attn.qkv", "mlp.fc2")}, flush=True)
            del st
            torch.cuda.empty_cache()
        res.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "deflate", "g1b.json"), res)


if __name__ == "__main__":
    main()
