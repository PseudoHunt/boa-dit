# Attention sharing — Gate 1: is cross-image agreement of early attention real and non-trivial?

Written 2026-10-25T10:10Z, before any Gate 1 statistic was computed. Results are appended below the marker by
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
