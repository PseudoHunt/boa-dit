"""Part V write-up: results/feasibility/var/{model_facts.md, SUMMARY.md, stats.png, grid.png} from the JSONs."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import CKPT_DIR, atomic_savefig, atomic_write_text  # noqa: E402

OUT = os.path.join(ROOT, "results", "feasibility", "var")
ARMS = ["full", "oracle25", "positional25", "random25", "stock"]


def load(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    par, st = load("parity.json"), load("stats.json")
    R = {a: load(f"eval/{a}.json") for a in ARMS}
    F = ["# VAR-d30 — model facts (verified against the loaded objects)\n",
         "| fact | expected | found |", "| --- | --- | --- |",
         "| code | FoundationVision/VAR | commit 78b9539 (external/VAR) |",
         "| weights | HF FoundationVision/var | vae_ch160v4096z32.pth, var_d30.pth |",
         "| depth / hidden / heads | 30 / 1920 / 30 × 64 | 30 / 1920 / 30 × 64 |",
         "| scales | (1,2,3,4,5,6,8,10,13,16) | same; cumulative tokens 1, 5, 14, 30, 55, 91, 155, 255, 424, 680 |",
         "| attention | block-causal, KV cache | KV cache at inference (attn_bias None); attn_l2_norm (normalized q·k × learned per-head scale) |",
         "| sampling (repo demo) | — | cfg 4, top_k 900, top_p 0.95, more_smooth False, TF32 on (README FID setting differs: cfg 1.5, top_p 0.96) |",
         "| same seed twice | identical tokens | identical images (16-image batch) |",
         "| stock attention kernel | — | torch fused `scaled_dot_product_attention` (no flash-attn / xformers installed) |\n",
         "Deviations, stated before the arms ran:",
         "- **Batch 16, not 32:** batch 32 ran out of memory (the KV cache for 64 CFG rows × 30 blocks, plus fp32 weights). Stock "
         "speed at batch 16 is 0.095 s/image, so 1,000 images per arm fit.",
         "- **Seeds:** VAR's sampler takes one RNG seed per call, so the matched unit is a fixed batch of 16 (labels and batch "
         "seed from default_rng(0)), identical across arms.",
         f"- **Parity:** stock uses fused SDPA, so no explicit path reproduces its tokens (images identical: "
         f"{par['images_identical_to_stock']}). Per call on identical inputs: logits fp16 vs fp32 max rel "
         f"{par['per_call_logits_fp16_vs_fp32_max_rel']:.1e} (tolerance 1e-3); explicit fp32 softmax output vs SDPA "
         f"{par['per_call_output_explicit_vs_sdpa_max_rel']:.1e}, against an SDPA math-vs-default floor of "
         f"{par['per_call_output_sdpa_math_vs_default_max_rel']:.1e}. Every arm (`full` included) uses the same explicit "
         "path, so arms differ only in the mask; a `stock` arm checks that `full` is statistically equivalent to stock."]
    atomic_write_text(os.path.join(OUT, "model_facts.md"), "\n".join(F) + "\n")

    P = st["per"]
    blocks, scales = range(30), (10, 13, 16)
    grid = lambda k: np.array([[np.mean(P[f"b{b}.s{s}"][k]) for s in scales] for b in blocks])
    fig, axs = plt.subplots(1, 2, figsize=(10, 7))
    for ax, (k, t) in zip(axs, [("positional_share", "positional share of oracle top-25% mass"), ("frac_keys_95", "fraction of allowed keys for 95% mass")]):
        im = ax.imshow(grid(k), aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(3)); ax.set_xticklabels([f"{s}×{s}" for s in scales]); ax.set_ylabel("block"); ax.set_title(t, fontsize=10)
        fig.colorbar(im, ax=ax)
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "stats.png")); plt.close(fig)
    rows = [a for a in ARMS if os.path.exists(os.path.join(CKPT_DIR, "feas_var", f"{a}_imgs.pt"))]
    fig, axs = plt.subplots(len(rows), 16, figsize=(24, 1.6 * len(rows)), squeeze=False)
    for r, a in enumerate(rows):
        im = torch.load(os.path.join(CKPT_DIR, "feas_var", f"{a}_imgs.pt"), map_location="cpu")[:16]
        for c in range(16):
            axs[r, c].imshow(im[c].permute(1, 2, 0).numpy()); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[r, 0].set_ylabel(a, fontsize=9)
    fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "grid.png")); plt.close(fig)

    med = lambda k, s: float(np.median([np.median(P[f"b{b}.s{s}"][k]) for b in blocks]))
    S = ["# Part V — VAR-d30 (ImageNet 256): query-aware key selection, feasibility summary\n"]
    full, orc, pos, rnd = R["full"], R["oracle25"], R["positional25"], R["random25"]
    tf = lambda x: "pending" if x is None else f"{100 * x['label_acc']:.1f}% [{100 * x['label_acc_ci95_wilson'][0]:.1f}, {100 * x['label_acc_ci95_wilson'][1]:.1f}]"
    v = None
    if full and orc and pos and rnd:
        ftop = np.array(full["top1"])
        chg = {a: float((np.array(R[a]["top1"]) != ftop).mean()) for a in ARMS if R[a]}
        head = orc["label_acc"] >= full["label_acc"] - 0.015 and orc["fid1k"] <= full["fid1k"] + 1.0
        nonover = pos["label_acc_ci95_wilson"][1] < orc["label_acc_ci95_wilson"][0]
        room = head and orc["label_acc"] - pos["label_acc"] >= 0.03 and nonover and rnd["label_acc"] < pos["label_acc"]
        none_ = abs(pos["label_acc"] - orc["label_acc"]) <= 0.015
        v = "room for a content-aware selector" if room else ("no method to build: the fixed positional rule captures the headroom"
                                                                if (head and none_) else ("no headroom" if not head else
                                                                "neither outcome of the rule: headroom exists, positional-25 is below oracle-25 by more than 1.5 points but not by the 3-point, non-overlapping-CI margin"))
        S.append(f"**Verdict: {v}.**\n")
        S.append(f"- Gap 1, oracle vs full (is sparse attention at 25% harmless?): label accuracy {100 * (orc['label_acc'] - full['label_acc']):+.1f} "
                 f"points, FID-1k {orc['fid1k'] - full['fid1k']:+.2f} → headroom **{head}**.")
        S.append(f"- Gap 2, positional vs oracle (does a fixed rule capture it?): {100 * (pos['label_acc'] - orc['label_acc']):+.1f} points "
                 f"(CIs {'do not overlap' if nonover else 'overlap'}), FID-1k {pos['fid1k'] - orc['fid1k']:+.2f}; random-25 vs positional-25: "
                 f"{100 * (rnd['label_acc'] - pos['label_acc']):+.1f} points.\n")
    S.append("| arm | FID-1k | label acc (95% Wilson CI) | class differs from full |")
    S.append("| --- | ---: | --- | ---: |")
    for a in ARMS:
        x = R[a]
        S.append(f"| {a} | {x['fid1k']:.2f} | {tf(x)} | {100 * chg[a]:.1f}% |" if (x and v) else f"| {a} | " + (f"{x['fid1k']:.2f} | {tf(x)} | — |" if x else "pending | pending | — |"))
    S.append("\nFID-1k against the ADM ImageNet-256 reference is biased upward at 1,000 samples; read differences, not levels. "
             "Demo sampling settings (cfg 4) also favour quality over diversity, which raises FID.\n")
    S.append("## Attention statistics (64 images, cond branch; medians over the 30 blocks of per-head medians over queries)\n")
    S.append("| query scale | mass on same-scale keys | fraction of allowed keys for 90% / 95% mass | positional share of oracle top-25% mass | "
             "positional heads (median ≥ 0.9) | cross-image Jaccard of top-25% sets |")
    S.append("| --- | ---: | --- | ---: | ---: | ---: |")
    for s in scales:
        ph = np.mean([P[f"b{b}.s{s}"]["positional_heads_frac"] for b in blocks])
        S.append(f"| {s}×{s} | {med('same_scale_mass', s):.2f} | {med('frac_keys_90', s):.2f} / {med('frac_keys_95', s):.2f} | "
                 f"{med('positional_share', s):.2f} | {100 * ph:.0f}% | {med('jaccard_cross_image', s):.2f} |")
    S.append("\nPlots: `stats.png` (positional share and 95%-mass fraction over block × scale), `grid.png` (16 matched images per arm). "
             "Facts and deviations: `model_facts.md`. Rule: `../rule.md`.\n")
    S.append("Note: at 256 px VAR's attention and KV cache are small (680 keys at the last scale). Any positive result here would "
             "need reproducing on Infinity-2B at 1024 px before it says anything about speed.\n")
    gh = sum(float(l.split(",")[-1]) for l in open(os.path.join(ROOT, "results", "gpu_hours.csv")).read().splitlines()[1:] if l.startswith("feas_var,"))
    S.append(f"GPU-hours (Part V, logged): {gh:.2f} of 1.5. About 0.05 h of setup timing runs were not logged.")
    atomic_write_text(os.path.join(OUT, "SUMMARY.md"), "\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()
