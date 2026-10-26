"""CFG-alloc ranks, corrected rule (rule_addendum_cfg.md, 2026-10-26 addendum). Per quantized layer, on the design cache
(held-out calibration prompts 32-63, 10 steps, 512 random tokens per row, same tokens in both branches): per-branch Gram
G = (X_c^T X_c + X_u^T X_u)/2 and H_cfg = V^T V (v = w x_c + (1-w) x_u), H_cfg rescaled to tr(G).
E = D^-1 (R^ + L1 L2) - W^T of the corrected base (c_in x c_out). e_k = lambda_k ||v_k^T E||^2 over the eigenpairs of H.
Greedy: one more rank at a time to the layer whose next unused direction (e_k sorted descending) has the largest e_k, cap 32,
until total branch params sum r_l (c_in + c_out) reach r4-everywhere. Done for H_cfg (the arm) and G (the control).
Writes configs/w4a4_cfgalloc_greedy{,_ctl}.json and results/w4a4/cfg_alloc.json.
Run: python -m boa_dit.w4a4.cfg_alloc --device cuda:0
"""
import argparse
import heapq
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, ROOT, Run, atomic_write_json
from boa_dit.w4a4.common import CK, GUIDANCE, LAYERS, block_linear, load_embeds, load_pipe
from boa_dit.w4a4.screen import text_for

ARM = "hsvd_r4_b0.5_kv16_u2_rs"
N_TOK, CAP = 512, 32


def greedy(e, cost, target):
    """e: {layer: e_k sorted descending (top CAP)}; returns {layer: r}."""
    r = {k: 0 for k in e}
    h = [(-float(v[0]), k) for k, v in e.items()]
    heapq.heapify(h)
    params = 0
    while h and params < target:
        _, k = heapq.heappop(h)
        r[k] += 1
        params += cost[k]
        if r[k] < CAP:
            heapq.heappush(h, (-float(e[k][r[k]]), k))
    return r


def summarize(r, e, tot, cost, target):
    params = sum(r[k] * cost[k] for k in r)
    cap = sum(float(e[k][:r[k]].sum()) for k in r)
    cap4 = sum(float(e[k][:4].sum()) for k in r)
    T = sum(tot.values())
    by_type = {}
    for t in LAYERS:
        ks = [k for k in r if k.split(".", 1)[1] == t]
        if not ks:
            continue
        rs = [r[k] for k in ks]
        by_type[t] = {"mean": float(np.mean(rs)), "median": float(np.median(rs)), "min": int(min(rs)), "max": int(max(rs)),
                      "total_rank": int(sum(rs)), "share_of_rank": sum(rs) / sum(r.values()),
                      "share_of_branch_params": sum(r[k] * cost[k] for k in ks) / params,
                      "share_of_total_e": sum(tot[k] for k in ks) / T}
    return {"params": params, "target": target, "ratio": params / target, "within_2pct": abs(params / target - 1) <= 0.02,
            "captured_share": cap / T, "captured_share_r4": cap4 / T, "captured_over_r4": cap / cap4,
            "n_zero": sum(1 for v in r.values() if v == 0), "n_at_cap": sum(1 for v in r.values() if v == CAP),
            "by_type": by_type}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default=ARM)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev, w = a.device, GUIDANCE
    torch.cuda.set_device(dev)
    qs = torch.load(os.path.join(CK, f"q_{a.arm}.pt"), map_location="cpu")
    c = torch.load(os.path.join(CK, "design_cache.pt"), map_location="cpu")
    e = load_embeds("design")
    with Run("w4a4", "cfg_alloc_greedy", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        tr = pipe.transformer
        del pipe.vae
        keys = list(qs)
        S = {k: [None, None] for k in keys}                                                # [G sum, H_cfg sum] fp32 on GPU
        cur, hooks = {}, []
        for key in keys:
            l, ln = key.split(".", 1)

            def h(mod, inp, key=key):
                x = inp[0]
                n = x.shape[0] // 2                                                       # rows [0,n) uncond, [n,2n) cond
                xt = (x[:, cur["tok"]] if x.shape[1] == 4096 else x).float()
                xu, xc = xt[:n].reshape(-1, x.shape[-1]), xt[n:].reshape(-1, x.shape[-1])
                v = w * xc + (1 - w) * xu
                torch.backends.cuda.matmul.allow_tf32 = True                              # Gram GEMMs only, as in quantize
                try:
                    g, hc = (xc.T @ xc + xu.T @ xu) / 2, v.T @ v
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = False
                if S[key][0] is None:
                    S[key] = [g, hc]
                else:
                    S[key][0] += g
                    S[key][1] += hc
            hooks.append(block_linear(tr.transformer_blocks[int(l)], ln).register_forward_pre_hook(h))
        gen = torch.Generator().manual_seed(0)
        for s in sorted(set(c["step"].tolist())):
            idx = {(int(c["prompt"][j]), int(c["branch"][j])): j for j in range(len(c["t"])) if int(c["step"][j]) == s}
            for p0 in range(0, 32, 16):
                ps = list(range(p0, p0 + 16))
                rows = [idx[(p, 0)] for p in ps] + [idx[(p, 1)] for p in ps]
                cur["tok"] = torch.randperm(4096, generator=gen)[:N_TOK].to(dev)
                pe, pm = text_for(e, c["prompt"][rows], c["branch"][rows], dev)
                tr(c["latent"][rows].to(dev), encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=c["t"][rows].to(dev),
                   added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
        for hk in hooks:
            hk.remove()
        eH, eG, totH, totG, cost, spec = {}, {}, {}, {}, {}, {}
        for key in keys:
            l, ln = key.split(".", 1)
            W = block_linear(tr.transformer_blocks[int(l)], ln).weight.data.float()      # [c_out, c_in]
            q = qs[key]
            d = q["d"].float().to(dev)
            E = ((q["R"].float().to(dev).T + q["L1"].float().to(dev) @ q["L2"].float().to(dev)) / d[:, None] - W.T).double()
            G, Hc = (m.double() for m in S.pop(key))
            Hc = Hc * (torch.trace(G) / torch.trace(Hc))
            for M, out, tot in ((Hc, eH, totH), (G, eG, totG)):
                lam, V = torch.linalg.eigh((M + M.T) / 2)
                ek = lam.clamp_min(0) * (V.T @ E).pow(2).sum(1)
                tot[key] = float(ek.sum())
                out[key] = torch.sort(ek, descending=True).values[:CAP].cpu().numpy()
            cost[key] = W.shape[0] + W.shape[1]
            spec[key] = {"top1_share_cfg": float(eH[key][0] / totH[key]), "top4_share_cfg": float(eH[key][:4].sum() / totH[key]),
                         "top32_share_cfg": float(eH[key].sum() / totH[key]), "top4_share_G": float(eG[key][:4].sum() / totG[key])}
            del E, G, Hc
        target = 4 * sum(cost.values())
        rH, rG = greedy(eH, cost, target), greedy(eG, cost, target)
        li = {k: int(k.split(".", 1)[0]) * len(LAYERS) + LAYERS.index(k.split(".", 1)[1]) for k in keys}
        sH, sG = summarize(rH, eH, totH, cost, target), summarize(rG, eG, totG, cost, target)
        sGH = summarize(rG, eH, totH, cost, target)                                       # control's ranks scored under H_cfg
        q_rank = sH["by_type"]["attn2.to_q"]["share_of_rank"]
        res = {"prompt_set": "design (held-out calibration prompts 32-63, rule_addendum_design.md)", "arm": a.arm, "w": w,
               "rule": "greedy on e_k = lambda_k ||v_k^T E||^2, cap 32, params = r4-everywhere (rule_addendum_cfg.md 2026-10-26)",
               "cfg": sH, "control_G": sG, "control_ranks_scored_under_cfg": {k: sGH[k] for k in ("captured_share", "captured_over_r4")},
               "attn2_to_q_share_of_rank_cfg": q_rank, "attn2_to_q_takes_most": q_rank > 0.5,
               "per_layer": {k: {"r_cfg": rH[k], "r_G": rG[k], **spec[k]} for k in keys}, **run.record()}
    atomic_write_json(os.path.join(ROOT, "configs", "w4a4_cfgalloc_greedy.json"), {"r": {str(li[k]): rH[k] for k in keys}})
    atomic_write_json(os.path.join(ROOT, "configs", "w4a4_cfgalloc_greedy_ctl.json"), {"r": {str(li[k]): rG[k] for k in keys}})
    atomic_write_json(os.path.join(RESULTS, "w4a4", "cfg_alloc.json"), res)
    for lab, s in (("H_cfg", sH), ("per-branch G", sG)):
        print(f"{lab}: params x{s['ratio']:.3f}, captured {100 * s['captured_share']:.1f}% of sum e_k vs r4 "
              f"{100 * s['captured_share_r4']:.1f}% ({s['captured_over_r4']:.2f}x); zero {s['n_zero']}, at cap {s['n_at_cap']}", flush=True)
        for t, v in s["by_type"].items():
            print(f"  {t:16s} mean r {v['mean']:5.1f} [{v['min']}, {v['max']}]  rank share {100 * v['share_of_rank']:4.1f}%  "
                  f"params {100 * v['share_of_branch_params']:4.1f}%  e share {100 * v['share_of_total_e']:4.1f}%", flush=True)
    print("attn2.to_q share of rank under H_cfg:", round(q_rank, 3), flush=True)


if __name__ == "__main__":
    main()
