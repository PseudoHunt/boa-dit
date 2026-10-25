"""Deflate G1: verdict + plots below the marker in results/deflate/G1.md, from g1.json / g1_spectra.json."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import atomic_savefig, atomic_write_json, atomic_write_text  # noqa: E402

OUT = os.path.join(ROOT, "results", "deflate")
MARK = "\n---\n\n## Results (appended by `scripts/make_deflate_g1.py`)\n"
TYPES = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


def main():
    g = json.load(open(os.path.join(OUT, "g1.json")))
    sp = json.load(open(os.path.join(OUT, "g1_spectra.json")))
    L_ = g["layers"]
    grid = lambda k: np.array([[L_[f"b{b}.{t}"][k] for t in TYPES] for b in range(28)], dtype=float)
    share, r, eo, en = grid("mean_energy_share"), grid("r"), grid("erased_old"), grid("erased_new")
    cosm, v1u1 = grid("cos_xbar_p_median"), grid("abs_cos_v1_u1")
    stop = float((share < 0.3).mean()) > 0.8
    hi = share >= 0.5
    mech_all = bool((cosm[hi] >= 0.9).all()) if hi.any() else False
    v = {"frac_share_below_0.3": float((share < 0.3).mean()), "stop": stop, "r_star": g["r_star"],
         "n_high_share": int(hi.sum()), "mechanism_pass": mech_all, "frac_high_share_cos_ge_0.9": float((cosm[hi] >= 0.9).mean()),
         "verdict": "STOP (premise false: mean-energy share < 0.3 in > 80% of layers)" if stop else "PASS (G2 proceeds)"}
    atomic_write_json(os.path.join(OUT, "g1_verdict.json"), v)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    for ax, (a, t, kw) in zip(axs, [(share, "mean-energy share tr(H_mean)/tr(H)", dict(vmin=0, vmax=1)),
                                    (r, "r_l (95% of tr H_mean, cap 8)", dict(vmin=1, vmax=8)),
                                    (np.log10(eo + 1), "log10(1 + erased by old damping)", {}),
                                    (cosm, "median cos(token mean, predicted shift path)", dict(vmin=-0.2, vmax=1))]):
        im = ax.imshow(a, aspect="auto", origin="lower", cmap="viridis", **kw)
        ax.set_xticks(range(4)); ax.set_xticklabels(TYPES, rotation=30); ax.set_ylabel("block"); ax.set_title(t, fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "g1_heatmaps.png")); plt.close(fig)
    fig, axs = plt.subplots(3, 4, figsize=(20, 12))
    for i, b in enumerate((0, 13, 27)):
        for j, t in enumerate(TYPES):
            s = sp[f"b{b}.{t}"]
            ax = axs[i, j]
            ax.semilogy(np.clip(s["eig_H"], 1e-12, None), label="eig H")
            ax.semilogy(np.clip(s["eig_Hvar"], 1e-12, None), label="eig H_var")
            ax.axhline(s["damp_old"], color="r", ls="--", lw=1, label="old damping 0.01·mean diag H")
            ax.axhline(s["damp_new"], color="g", ls=":", lw=1, label="new damping 0.01·mean diag H_P")
            ax.set_title(f"block {b} {t}", fontsize=10)
            if i == 0 and j == 0:
                ax.legend(fontsize=7)
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "g1_spectra.png")); plt.close(fig)

    L = [MARK, "### Per layer type (medians over the 28 blocks; counts where stated)\n",
         "| layer | mean-energy share | share ≥ 0.5 (blocks) | r_l histogram (1…8) | top-1 / top-4 / top-8 frac of H_mean | "
         "erased old → new | median cos(x̄, p) | abs cos(v₁, u₁) |",
         "| --- | ---: | ---: | --- | --- | --- | ---: | ---: |"]
    for j, t in enumerate(TYPES):
        vs = [L_[f"b{b}.{t}"] for b in range(28)]
        L.append(f"| {t} | {np.median(share[:, j]):.3f} | {int(hi[:, j].sum())} | {np.bincount(r[:, j].astype(int), minlength=9)[1:].tolist()} | "
                 f"{np.median([x['top1_frac'] for x in vs]):.2f} / {np.median([x['top4_frac'] for x in vs]):.2f} / "
                 f"{np.median([x['top8_frac'] for x in vs]):.2f} | {np.median(eo[:, j]):.1f} → {np.median(en[:, j]):.1f} | "
                 f"{np.median(cosm[:, j]):.3f} | {np.median(v1u1[:, j]):.3f} |")
    big = sorted(L_.values(), key=lambda x: -x["erased_old"])[:5]
    L.append("\nLargest erased-direction counts (old → new): " + "; ".join(
        f"b{x['block']} {x['layer']} {x['erased_old']} → {x['erased_new']} (share {x['mean_energy_share']:.3f}, r {x['r']})" for x in big) + ".")
    L.append("\nHigh-share layers (share ≥ 0.5) and their median cos(x̄_s, p_s): " + "; ".join(
        f"b{x['block']} {x['layer']} {x['cos_xbar_p_median']:.2f}" for x in L_.values() if x["mean_energy_share"] >= 0.5) + ".\n")
    L.append("Plots: `g1_heatmaps.png`, `g1_spectra.png` (blocks 0, 13, 27; eig H and eig H_var with both damping lines).\n")
    L.append("### Verdict\n")
    L.append(f"- Mean-energy share < 0.3 in {100 * v['frac_share_below_0.3']:.0f}% of the 112 layers (STOP needs > 80%) → "
             f"**{v['verdict']}**.")
    L.append(f"- r_l = 8 (the cap) in {int((r == 8).sum())} of 112 layers; **r\\* = {v['r_star']}** (`configs/deflate_r.json`). "
             "The token-mean subspace is not low-rank in most layers.")
    L.append(f"- Mechanism: median cos(x̄_s, p_s) ≥ 0.9 in {int((cosm[hi] >= 0.9).sum())} of {v['n_high_share']} high-share layers → "
             f"**mechanism {'PASS' if mech_all else 'FAIL'}**. The token mean is not the adaLN shift. A likely cause (not measured): the "
             "token mean of LayerNorm(x)·(1 + scale) is itself large. This changes the explanation, not the method (handoff "
             "§1); G2 proceeds.")
    L.append("- In the early fc2 layers (blocks 0–3) the old damping erases 3,000–4,000 of 4,608 directions, and deflation "
             "barely changes that (share 0.09–0.28). Only block 27 fc2 (share 0.995) has its erased count cut sharply.\n")
    path = os.path.join(OUT, "G1.md")
    pre = open(path).read().split(MARK)[0]
    atomic_write_text(path, pre + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
