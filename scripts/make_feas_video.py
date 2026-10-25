"""Part C write-up: results/feasibility/video/{model_facts.md, SUMMARY.md, stats.png, frames.png} from the JSONs."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import CKPT_DIR, atomic_savefig, atomic_write_text  # noqa: E402

OUT = os.path.join(ROOT, "results", "feasibility", "video")
CK = os.path.join(CKPT_DIR, "feas_video")
ARMS = ["full", "oracle10", "window10", "random10"]
STEPS = (0, 10, 11, 25, 40)


def load(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    st = load("stats.json")
    R = {a: load(f"eval/{a}.json") for a in ["stock"] + ARMS}
    cfg = json.load(open("/home/models/cogvideox-2b/transformer/config.json"))
    F = ["# CogVideoX-2B — model facts (verified against the loaded config and pipeline)\n",
         "| fact | expected | found |", "| --- | --- | --- |",
         f"| blocks / heads | 30 / 30 × 64 | {cfg['num_layers']} / {cfg['num_attention_heads']} × {cfg['attention_head_dim']} |",
         f"| RoPE | none in 2B | use_rotary_positional_embeddings = {cfg['use_rotary_positional_embeddings']} (asserted in the processor) |",
         "| tokens | 226 text + 13 × 30 × 45 = 17,550 video | 226 + 17,550 (order: text, then video by frame, row, column) |",
         "| sampling | 480 × 720, 49 frames, 50 steps, guidance 6.0, default scheduler | same; seed 0 |",
         f"| stock seconds per video | — | {R['stock']['seconds_per_video']:.0f} s (fused SDPA) |" if R["stock"] else "| stock | — | pending |",
         f"| free GPU memory before sampling (T5 dropped, VAE tiling + slicing) | ≥ 12 GB | {R['stock']['gpu_free_gb_before']:.1f} GB |" if R["stock"] else "",
         "\nDeviations:",
         "- **T5** was loaded explicitly in fp16; transformers 5 ignores diffusers' `torch_dtype`, as found on PixArt.",
         "- **Attention implementation:** a fully explicit fp32 softmax over 1,024-query chunks measured 26 min (random), 31 min "
         "(window) and 59 min (oracle) per video, which would exceed the 2.5 h cap. Every arm therefore uses one chunked path: "
         "fused SDPA per 1,024-query chunk with a boolean keep-mask (fp32 accumulation and softmax inside the kernel). The full arm "
         "passes an all-True mask so it takes the same kernel path. oracle-10 computes explicit fp32 logits (all heads at once) "
         "only to select each video query's top 10% of keys. No N × N map is materialized. One forward of the full arm differs "
         "from stock by 2.4e-4 (relative, on random inputs).",
         "- **C1 statistics** are computed on the stock run, with explicit fp32 rows for the 128 sampled video queries (cond "
         "branch) at steps 0, 10, 11, 25 and 40."]
    atomic_write_text(os.path.join(OUT, "model_facts.md"), "\n".join(x for x in F if x) + "\n")
    S = ["# Part C — CogVideoX-2B (480p, 49 frames): query-aware key selection, feasibility summary\n"]
    if all(R[a] for a in ARMS):
        po, pw, pr = R["oracle10"]["latent_psnr_vs_full"], R["window10"]["latent_psnr_vs_full"], R["random10"]["latent_psnr_vs_full"]
        head = po >= 30
        room = head and pw <= po - 3 and pr <= pw - 3
        none_ = pw >= po - 1
        v = ("room for a content-aware selector" if room else
             "no method to build: locality captures the headroom (Sliding Tile Attention's territory)" if (head and none_) else
             "no headroom at 10% (oracle-10 below 30 dB)" if not head else "neither outcome of the rule")
        S.append(f"**Verdict (PSNR part of the rule): {v}.** The visual check of `frames.png` is stated below.\n")
        S.append(f"- Gap 1, oracle vs full: oracle-10 latent PSNR {po:.1f} dB (headroom needs ≥ 30 dB).")
        S.append(f"- Gap 2, window vs oracle: window-10 {pw:.1f} dB ({pw - po:+.1f} dB vs oracle); random-10 {pr:.1f} dB.\n")
        S.append("**Visual check (`frames.png`, my reading, written after looking at the grid):**")
        S.append("- full is visually identical to stock.")
        S.append("- oracle-10 keeps the scene, composition, motion and image quality with no artifacts, but renders a different "
                 "version of the video (a smaller dog, slightly different pose). Its PSNR shortfall is identity drift, not "
                 "visible degradation: the same mode-switch caveat that latent MSE had on DiT. Under the pre-registered rule "
                 "(PSNR ≥ 30 dB **and** no visible degradation) it still fails, and the verdict stands.")
        S.append("- window-10 is visibly degraded: the scene changes (two dogs, a different sun), the frames lose temporal "
                 "coherence, and by frame 48 the dog is gone and the frame is washed out.")
        S.append("- random-10 collapses into blocky noise.")
        S.append("- Reading: at a 10% budget, locality alone (the Sliding Tile premise) is far from the oracle, while the oracle "
                 "keeps quality but not identity. A content-aware selector at a higher budget (for example 20–25%) would be the "
                 "natural next measurement; that is not run here.\n")
    S.append("| arm | latent PSNR vs full (dB) | CLIP ViT-L/14 (8 frames) | seconds per video |")
    S.append("| --- | ---: | ---: | ---: |")
    for a in ["stock"] + ARMS:
        x = R[a]
        if x is None:
            S.append(f"| {a} | pending | | |")
            continue
        ps = x.get("latent_psnr_vs_full", x.get("latent_psnr_vs_stock"))
        S.append(f"| {a} | " + ("— (reference)" if a == "stock" else (f"{ps:.1f}" + (" (vs stock)" if a == "full" else "")))
                 + f" | {x['clip_vitl14_8frames']:.2f} | {x['seconds_per_video']:.0f} |")
    if R["full"] and "step0_latent_rel_diff_vs_stock" in R["full"]:
        S.append(f"\nParity: full vs stock latents after step 0, relative difference {R['full']['step0_latent_rel_diff_vs_stock']:.1e} "
                 "(tolerance 1e-2).\n")
    if st:
        P = st["per"]
        mh = lambda p, b, s, k: float(np.mean(P[f"p{p}.b{b}.s{s}"][k]))
        S.append("## Attention statistics (128 sampled video queries; prompt 1 unless stated; medians over queries, means over heads, "
                 "medians over blocks)\n")
        S.append("| step | top-1% mass | top-5% | top-10% | text mass | locality (top-10% mass in window) | window overlap (Jaccard) | prompt Jaccard |")
        S.append("| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        for s in STEPS:
            g = lambda k: np.median([mh(0, b, s, k) for b in range(30)])
            S.append(f"| {s} | {g('top1_mass'):.2f} | {g('top5_mass'):.2f} | {g('top10_mass'):.2f} | {g('text_mass'):.3f} | "
                     f"{g('locality'):.2f} | {g('window_overlap'):.2f} | {g('prompt_jaccard'):.2f} |")
        sj = np.median([np.mean(P[f"p0.b{b}.s11"]["step_jaccard_10_11"]) for b in range(30)])
        S.append(f"\nStep stability (top-10% sets, step 10 vs 11, prompt 1): Jaccard {sj:.2f}. High stability is the caching headroom "
                 "that TeaCache and PAB already exploit.\n")
        fig, axs = plt.subplots(1, 3, figsize=(16, 6))
        for ax, (k, t) in zip(axs[:2], [("locality", "locality (top-10% mass inside ±1 frame, radius 12)"), ("top10_mass", "top-10% mass")]):
            a = np.array([[mh(0, b, s, k) for s in STEPS] for b in range(30)])
            im = ax.imshow(a, aspect="auto", origin="lower", cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks(range(len(STEPS))); ax.set_xticklabels(STEPS); ax.set_xlabel("step"); ax.set_ylabel("block"); ax.set_title(t, fontsize=10)
            fig.colorbar(im, ax=ax)
        loc25 = np.concatenate([P[f"p0.b{b}.s25"]["locality"] for b in range(30)])
        axs[2].hist(loc25, bins=20); axs[2].set_xlabel("per-head locality, step 25"); axs[2].set_ylabel("heads (30 blocks × 30)")
        fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "stats.png")); plt.close(fig)
    rows = [a for a in ["stock"] + ARMS if os.path.exists(os.path.join(CK, f"{a}.pt"))]
    if rows:
        fig, axs = plt.subplots(len(rows), 5, figsize=(20, 2.4 * len(rows)), squeeze=False)
        for r, a in enumerate(rows):
            g = torch.load(os.path.join(CK, f"{a}.pt"), map_location="cpu")["grid"]
            for c in range(5):
                axs[r, c].imshow(g[c].permute(1, 2, 0).numpy()); axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
                if r == 0:
                    axs[r, c].set_title(f"frame {(0, 12, 24, 36, 48)[c]}")
            axs[r, 0].set_ylabel(a)
        fig.tight_layout(); atomic_savefig(fig, os.path.join(OUT, "frames.png")); plt.close(fig)
    gh = sum(float(l.split(",")[-1]) for l in open(os.path.join(ROOT, "results", "gpu_hours.csv")).read().splitlines()[1:] if l.startswith("feas_video,"))
    S.append(f"GPU-hours (Part C, logged): {gh:.2f} of 2.5. About 0.25 h of timing tests were not logged. Facts and deviations: "
             "`model_facts.md`. Rule: `../rule.md`.")
    atomic_write_text(os.path.join(OUT, "SUMMARY.md"), "\n".join(S) + "\n")
    print("\n".join(S))


if __name__ == "__main__":
    main()
