# Attention sharing — Gate 1: is cross-image agreement of early attention real and non-trivial?

Written 2026-10-25T10:07Z (commit 829a03b), while the Gate 1 run was sampling and before any of its statistics existed. Results are appended below the marker by
`scripts/make_attn_gate1.py`; nothing above the marker changes after this time.

## Data
FP DiT-XL/2 256, cfg 1.5, DDIM-50. 32 images: noise seeds 1000–1031; image m has class `CLASSES[m // 2]`, the 16
classes drawn by `np.random.default_rng(0).choice(1000, 16, replace=False)` (sorted; recorded in
`gate1_stats.json`). Steps 0–24, all 28 blocks × 16 heads, both CFG branches. P is rebuilt in fp32 from the
block's own q, k; the sampling forward stays on the fused path. Explicit-softmax vs fused block outputs: max
relative difference recorded as `parity` in the JSON (tolerance 1e-3).

## Statistics (`boa_dit/attn_share/gate1.py`)
As in the handoff, cond branch unless named. Details fixed here:
- `ent_median`: median over (image, position) of row entropy / log 256.
- KL ratio: per position, mean over images of KL(P‖U) divided by mean over images of KL(P‖M), then median over
  positions. M includes the image itself, as specified; a leave-one-out version (M without image m) is also
  reported, as a check that the in-sample mean is not flattering the ratio.
- Class vs noise: same-class pairs are (2c, 2c+1), 16 pairs; different-class pairs are all 480 pairs across classes.

## Reading of the verdict wording (fixed now)
- **Agreement:** the condition is checked **at every step s = 0…9**. At each step, at least 70% of the 448 (block, head)
  pairs must have median `within` ≥ 0.8. The pooled fraction over all (block, head, s ≤ 9) is also reported. If
  the pooled and per-step readings disagree, that is stated, and the per-step one decides.
- **Structure:** among the (block, head, s ≤ 9) triples that clear 0.8, at least half must have KL ratio ≥ 3
  (in-sample M, as specified).
- `s_max` is the largest s ≤ 24 with ≥ 70% of pairs clearing 0.8. The largest s such that every step up to it
  clears is also reported.
- If agreement holds but the KL ratio is near 1: STOP with the note "early attention is near-uniform".

---

## Results (appended by `scripts/make_attn_gate1.py`)

Parity: explicit-softmax vs fused block outputs, max relative difference 7.64e-04 (tolerance 1e-3). The share path fed one image's own maps reproduces the explicit path to 1.15e-03. Classes: [16, 40, 74, 174, 266, 304, 502, 504, 605, 628, 646, 729, 808, 837, 909, 969].

### Per step (cond branch; medians over the 448 (block, head) pairs unless noted)

| s | frac pairs within ≥ 0.8 | within | within p10 | within uncond | ent (images) | ent (mean map) | KL ratio | KL ratio LOO | same-class | diff-class | branch cos | kernel cos | M[s]·M[s+1] | M[s]·M[s+5] |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.99 | 0.973 | 0.967 | 0.986 | 0.988 | 0.991 | 4.697 | 4.406 | 0.982 | 0.973 | 0.999 | 0.989 | 1.000 | 0.988 |
| 1 | 0.99 | 0.971 | 0.964 | 0.986 | 0.987 | 0.990 | 4.904 | 4.603 | 0.981 | 0.971 | 0.999 | 0.987 | 1.000 | 0.987 |
| 2 | 0.99 | 0.968 | 0.960 | 0.985 | 0.984 | 0.988 | 4.984 | 4.673 | 0.979 | 0.967 | 0.999 | 0.986 | 0.999 | 0.985 |
| 3 | 0.99 | 0.964 | 0.955 | 0.984 | 0.982 | 0.986 | 4.982 | 4.669 | 0.978 | 0.964 | 0.999 | 0.984 | 0.999 | 0.985 |
| 4 | 0.99 | 0.960 | 0.949 | 0.982 | 0.979 | 0.983 | 5.184 | 4.839 | 0.976 | 0.959 | 0.998 | 0.982 | 0.999 | 0.985 |
| 5 | 0.98 | 0.955 | 0.942 | 0.981 | 0.975 | 0.981 | 5.203 | 4.870 | 0.974 | 0.954 | 0.998 | 0.981 | 0.999 | 0.985 |
| 6 | 0.99 | 0.950 | 0.934 | 0.979 | 0.972 | 0.978 | 5.125 | 4.805 | 0.971 | 0.949 | 0.998 | 0.980 | 0.999 | 0.986 |
| 7 | 0.98 | 0.941 | 0.925 | 0.976 | 0.966 | 0.974 | 4.849 | 4.548 | 0.966 | 0.940 | 0.998 | 0.980 | 0.999 | 0.987 |
| 8 | 0.98 | 0.933 | 0.914 | 0.973 | 0.961 | 0.971 | 4.646 | 4.355 | 0.962 | 0.932 | 0.998 | 0.979 | 0.999 | 0.987 |
| 9 | 0.97 | 0.925 | 0.903 | 0.969 | 0.954 | 0.968 | 4.410 | 4.124 | 0.955 | 0.924 | 0.998 | 0.979 | 0.999 | 0.989 |
| 10 | 0.92 | 0.913 | 0.889 | 0.964 | 0.948 | 0.964 | 4.104 | 3.845 | 0.950 | 0.912 | 0.997 | 0.977 | 0.999 | 0.990 |
| 11 | 0.87 | 0.903 | 0.876 | 0.959 | 0.943 | 0.960 | 3.885 | 3.636 | 0.939 | 0.901 | 0.997 | 0.976 | 1.000 | 0.991 |
| 12 | 0.81 | 0.888 | 0.858 | 0.951 | 0.936 | 0.957 | 3.663 | 3.412 | 0.927 | 0.887 | 0.997 | 0.975 | 1.000 | 0.992 |
| 13 | 0.76 | 0.872 | 0.839 | 0.942 | 0.929 | 0.954 | 3.476 | 3.239 | 0.911 | 0.871 | 0.996 | 0.974 | 1.000 | 0.992 |
| 14 | 0.70 | 0.853 | 0.821 | 0.930 | 0.922 | 0.951 | 3.309 | 3.074 | 0.893 | 0.852 | 0.996 | 0.974 | 1.000 | 0.991 |
| 15 | 0.62 | 0.837 | 0.798 | 0.917 | 0.914 | 0.947 | 3.079 | 2.865 | 0.873 | 0.835 | 0.996 | 0.973 | 1.000 | 0.990 |
| 16 | 0.55 | 0.812 | 0.776 | 0.901 | 0.905 | 0.942 | 2.892 | 2.676 | 0.851 | 0.811 | 0.995 | 0.972 | 1.000 | 0.990 |
| 17 | 0.46 | 0.787 | 0.747 | 0.886 | 0.899 | 0.938 | 2.723 | 2.514 | 0.826 | 0.785 | 0.995 | 0.972 | 1.000 | 0.989 |
| 18 | 0.42 | 0.759 | 0.717 | 0.865 | 0.892 | 0.936 | 2.562 | 2.373 | 0.798 | 0.759 | 0.994 | 0.972 | 0.999 | 0.988 |
| 19 | 0.37 | 0.732 | 0.684 | 0.845 | 0.885 | 0.932 | 2.437 | 2.234 | 0.772 | 0.731 | 0.993 | 0.971 | 0.999 | 0.988 |
| 20 | 0.34 | 0.703 | 0.651 | 0.824 | 0.877 | 0.929 | 2.346 | 2.140 | 0.742 | 0.701 | 0.992 | 0.970 | 0.999 | — |
| 21 | 0.33 | 0.676 | 0.627 | 0.800 | 0.868 | 0.926 | 2.224 | 2.037 | 0.716 | 0.676 | 0.992 | 0.970 | 0.999 | — |
| 22 | 0.31 | 0.661 | 0.604 | 0.771 | 0.858 | 0.922 | 2.141 | 1.955 | 0.694 | 0.660 | 0.991 | 0.969 | 0.999 | — |
| 23 | 0.30 | 0.642 | 0.585 | 0.742 | 0.848 | 0.918 | 2.086 | 1.892 | 0.673 | 0.641 | 0.991 | 0.968 | 0.999 | — |
| 24 | 0.30 | 0.617 | 0.563 | 0.711 | 0.841 | 0.914 | 2.031 | 1.820 | 0.653 | 0.616 | 0.990 | 0.967 | — | — |

Plots: `gate1_heatmaps.png` (within / entropy / KL ratio over block × step), `gate1_heads.png` (16 heads, blocks 0, 13, 27), `gate1_class_vs_noise.png`, `gate1_examples.png` (block 13, head 0).

### Verdict

- Agreement at every step 0–9 (≥ 70% of pairs with within ≥ 0.8): **True** (pooled fraction over s ≤ 9: 0.985).
- Share of clearing triples (s ≤ 9) with KL ratio ≥ 3: **0.616** (needs ≥ 0.5; leave-one-out: 0.599).
- s_max = **14** (contiguous from s = 0: 14).
- Class vs noise (s ≤ 9, median over pairs): same-class 0.974, different-class 0.956.
- Branch agreement (s ≤ 9): 0.998; offset-kernel reconstruction (s ≤ 9): 0.983.

**Gate 1: PASS**

