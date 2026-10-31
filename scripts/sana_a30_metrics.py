"""snk1 metrics in the main DiT environment (has lpips + VGG weights; nk_env is left unmodified): per-prompt latent rel. MSE, PSNR and LPIPS-VGG vs
H20 from ckpt/sana_a30/<tag>_outputs.pt, and paired image bootstrap (10,000 draws, numpy seed 0) ratio-of-means / difference CIs.
Run: /root/miniconda3/envs/py3.10/bin/python scripts/sana_a30_metrics.py dev|conf
"""
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CK, OUT = os.path.join(ROOT, "ckpt", "sana_a30"), os.path.join(ROOT, "results", "sana_a30")


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def contrast(Mx, x, y, nb=10000):
    n = len(Mx[x]["rel_mse"])
    g = np.random.default_rng(0)
    idx = [g.integers(0, n, n) for _ in range(nb)]
    out = {}
    for m in ("rel_mse", "psnr", "lpips"):
        a, b = np.asarray(Mx[x][m]), np.asarray(Mx[y][m])
        r = {"diff": float(np.mean(a - b)), "diff_ci": ci([np.mean(a[i] - b[i]) for i in idx]), "median_diff": float(np.median(a - b)),
             "wins_x": int(np.sum(a > b) if m == "psnr" else np.sum(a < b)), "n": n}
        if m == "rel_mse":
            r["ratio"] = float(a.mean() / b.mean())
            r["ratio_ci"] = ci([a[i].mean() / b[i].mean() for i in idx])
        out[m] = r
    return out


@torch.no_grad()
def main(tag):
    import lpips
    D = torch.load(os.path.join(CK, f"{tag}_outputs.pt"), map_location="cpu")
    net = lpips.LPIPS(net="vgg", verbose=False).cuda().eval()
    ref_l, ref_i = D["lat"]["H20"].double(), D["img"]["H20"].float()
    Mx = {}
    for a in D["lat"]:
        if a == "H20":
            continue
        L, I = D["lat"][a].double(), D["img"][a].float()
        rm = ((L - ref_l).flatten(1).pow(2).sum(1) / ref_l.flatten(1).pow(2).sum(1)).tolist()
        ps = [float(10 * torch.log10(255.0 ** 2 / (I[i] - ref_i[i]).pow(2).mean().clamp_min(1e-10))) for i in range(len(I))]
        lp = [float(net(ref_i[i:i + 1].cuda() / 127.5 - 1, I[i:i + 1].cuda() / 127.5 - 1).flatten()[0]) for i in range(len(I))]
        Mx[a] = {"rel_mse": rm, "psnr": ps, "lpips": lp}
    arms = list(Mx)
    pairs = [("R", "A"), ("R", "Q20"), ("A", "Q20"), ("S", "A"), ("R", "S")] + [("R", a) for a in arms if a.startswith("H")]
    C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in pairs if x in Mx and y in Mx}
    S = {a: {m: {"mean": float(np.mean(v[m])), "median": float(np.median(v[m]))} for m in v} for a, v in Mx.items()}
    out = {"tag": tag, "prompts": D["idx"], "calls": D["calls"], "per_prompt": Mx, "summary": S, "contrasts": C,
           "note": "fidelity to H20 only (same seed / prompt); not generation quality. Q is not run-to-run deterministic (determinism.json)."}
    json.dump(out, open(os.path.join(OUT, f"{tag}_metrics.json"), "w"), indent=1)
    for a, v in S.items():
        print(f"  {a}: rel MSE {v['rel_mse']['mean']:.4f} (median {v['rel_mse']['median']:.4f})  PSNR {v['psnr']['mean']:.2f}  LPIPS {v['lpips']['mean']:.4f}")
    for k, c in C.items():
        m = c["rel_mse"]
        print(f"  {k}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] wins {m['wins_x']}/{m['n']} dLPIPS {c['lpips']['diff']:+.4f} "
              f"[{c['lpips']['diff_ci'][0]:+.4f}, {c['lpips']['diff_ci'][1]:+.4f}]")


if __name__ == "__main__":
    main(sys.argv[1])
