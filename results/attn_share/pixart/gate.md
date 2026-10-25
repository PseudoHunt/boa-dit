# PixArt-Σ 1024: does sample-independent early attention survive text conditioning at 4096 tokens? (pre-registration)

Written 2026-10-25, before any PixArt statistic, timing or image was produced (the weights were still downloading).
Results are appended below the marker by `scripts/make_pixart_gate.py`; nothing above the marker changes afterwards.

## Setup
`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`, diffusers `PixArtSigmaPipeline`, fp16, FP weights, one GPU. 8 fixed captions
(verbatim from the handoff) × seeds {0, 1} = 16 images. Image m has prompt m // 2 and seed m % 2, so the same-prompt
pairs are (2p, 2p+1). The empty prompt is the unconditional branch. Default scheduler and guidance unless
`model_facts.md` records otherwise. Self-attention (`attn1`) maps are rebuilt in fp32, one head at a time, from the
block's own q and k. Statistics are taken at steps 0–9 of 20 (0 = noisiest), on the cond branch unless named.

## Statistics
As in the handoff, §2. Details fixed here, consistent with the DiT-XL/2 Gate 1:
- `within`: per query position i, the mean pairwise cosine of row i across the 16 images, then median (and p10) over
  i. `within_sameprompt` uses the 8 same-prompt pairs; `within_diffprompt` uses the 112 different-prompt pairs.
- KL ratio: per position, mean over images of KL(P‖U) divided by mean over images of KL(P‖M), then median over
  positions (M = mean over the 16 images).
- Offset kernel κ: 127 × 127, from the cond mean map M per (block, head, step). Floor = min(κ) (primary) and median(κ)
  (robustness check). κ_res = (κ − floor)₊. The side of the smallest centered square holding 95% of κ_res's mass is
  measured in tokens.

## Verdict rule (from the handoff, §5), over steps s = 0–7
- **A. Sample-independence:** at every step s ≤ 7, at least 70% of the measured (block, head) pairs have median
  `within` ≥ 0.8 (the per-step reading, as in the DiT-XL/2 Gate 1; the pooled fraction is also reported). **And** among
  the (block, head, s ≤ 7) triples that clear 0.8, at least half have KL ratio ≥ 3.
- **B. Prompt-independence:** `within_diffprompt / within_sameprompt` ≥ 0.9 in at least 70% of the measured
  (block, head, s ≤ 7) triples.
- **C. Compact residual:** over the (block, head, s ≤ 7) cond kernels, with floor = min(κ): the median 95%-mass side is
  ≤ 13 tokens, and at least 90% of kernels have side ≤ 21. The median-floor version is reported as a robustness check.
  It does not enter the verdict.
- `s_max` = the last step s ≤ 9 at which A's per-step agreement condition still holds.

**PASS** requires A, B and C.
- If A or B fails: STOP, "the phenomenon is specific to class-conditional DiT".
- If C alone fails: STOP, with the median support reported ("the finding holds but the linear-time form does not").

Every number is reported regardless of the verdict.

## Scope fallbacks (decided now)
If the full 28-block sweep cannot fit the 1.5 GPU-hour cap (estimated from a 1-step timing before the full run), the
sweep is cut to blocks {0, 4, 9, 13, 18, 22, 27} with all heads, and this is recorded in `model_facts.md`. The verdict
thresholds then apply to the measured pairs.

## Prior from DiT-XL/2 at 256 px (recorded so the reader can judge)
Residual-kernel support there was not compact: the 95% square was the full 31 × 31 grid in about 98% of kernels
(`../kernel_residual_support.json`). C is therefore the condition most likely to fail.

## Addendum, 2026-10-25 (after the results below were computed and published)
Change to condition C, requested by the reviewer: C is now judged on **K_min** from a functional truncation test.
Per (block, head, step ≤ 7) cond kernel κ and side K ∈ {5, 9, 13, 21, 31, 63, 127}: floor f_K = mean of κ over the
cells outside the centered K × K square; κ_K = κ inside the square and f_K outside; rebuild the row-normalized map and
take its cosine to M (median over rows). K_min = the smallest K with cosine ≥ 0.98. **C holds iff median K_min ≤ 13
and the 90th percentile ≤ 21 tokens.** The 95%-mass side is reported alongside as a secondary number.
Reason: the 95%-mass side is biased by far cells. Each is averaged over few token pairs and clipped to non-negative
noise, and they make up most of the kernel's cells.
Honest timing: this addendum was written after the PixArt statistics, including the 95%-mass side and the overall
verdict, had been computed and pushed (commit d587a31). It is therefore not a blind pre-registration of C. It was
written before any K_min was computed for PixArt. The overall verdict cannot change, because A failed.

---

## Results (appended by `scripts/make_pixart_gate.py`)

Blocks measured: all 28; 16 heads; steps 0–9 of 20; 16 images. Processor parity: explicit fp32-softmax vs default SDPA max rel 3.7e-03 (rel Frobenius 5.8e-04). PyTorch's own math-backend SDPA vs the default differs by 4.1e-03 (5.7e-04), so the 1e-3 max-rel tolerance is below the fp16 noise floor of this all-fp16 model. The processor was accepted as equivalent on that basis; the tolerance was not changed.

### Per step (cond; medians over the measured (block, head) pairs)

| s | frac within ≥ 0.8 | within | within p10 | same-prompt | diff-prompt | diff / same | ent (images) | ent (mean map) | KL ratio | branch cos | kernel cos | floor share (min) | 95% residual side (min floor) | side (median floor) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.88 | 0.939 | 0.900 | 0.911 | 0.941 | 1.017 | 0.945 | 0.953 | 7.12 | 0.994 | 0.922 | 0.14 | 127 | 127 |
| 1 | 0.83 | 0.922 | 0.886 | 0.906 | 0.924 | 1.011 | 0.933 | 0.944 | 6.69 | 0.992 | 0.926 | 0.12 | 127 | 127 |
| 2 | 0.75 | 0.897 | 0.853 | 0.891 | 0.897 | 1.002 | 0.914 | 0.929 | 6.31 | 0.989 | 0.931 | 0.09 | 127 | 127 |
| 3 | 0.67 | 0.862 | 0.807 | 0.862 | 0.861 | 1.000 | 0.884 | 0.908 | 5.81 | 0.989 | 0.933 | 0.06 | 127 | 127 |
| 4 | 0.55 | 0.821 | 0.755 | 0.834 | 0.822 | 0.998 | 0.851 | 0.883 | 4.96 | 0.990 | 0.925 | 0.03 | 127 | 127 |
| 5 | 0.44 | 0.763 | 0.693 | 0.779 | 0.761 | 0.986 | 0.822 | 0.867 | 4.26 | 0.991 | 0.918 | 0.02 | 127 | 127 |
| 6 | 0.39 | 0.706 | 0.627 | 0.729 | 0.702 | 0.975 | 0.796 | 0.854 | 3.77 | 0.991 | 0.914 | 0.02 | 127 | 127 |
| 7 | 0.35 | 0.646 | 0.567 | 0.679 | 0.646 | 0.966 | 0.770 | 0.838 | 3.51 | 0.991 | 0.906 | 0.01 | 127 | 127 |
| 8 | 0.33 | 0.607 | 0.531 | 0.637 | 0.603 | 0.958 | 0.749 | 0.827 | 3.35 | 0.992 | 0.901 | 0.01 | 127 | 127 |
| 9 | 0.31 | 0.574 | 0.496 | 0.599 | 0.572 | 0.952 | 0.733 | 0.815 | 3.26 | 0.993 | 0.897 | 0.01 | 127 | 127 |

### Cross-attention control (block 13, heads [0, 7, 15], attn2 image→text rows)

| s | attn1 h0 within / ratio | attn1 h7 within / ratio | attn1 h15 within / ratio | attn2 h0 within / ratio | attn2 h7 within / ratio | attn2 h15 within / ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.817 / 0.998 | 0.784 / 0.924 | 0.807 / 1.005 | 0.139 / 0.077 | 0.160 / 0.100 | 0.141 / 0.080 |
| 1 | 0.784 / 0.976 | 0.714 / 0.897 | 0.775 / 0.975 | 0.139 / 0.077 | 0.165 / 0.106 | 0.143 / 0.082 |
| 2 | 0.737 / 0.962 | 0.636 / 0.877 | 0.739 / 0.952 | 0.139 / 0.078 | 0.168 / 0.110 | 0.144 / 0.083 |
| 3 | 0.678 / 0.944 | 0.545 / 0.846 | 0.691 / 0.933 | 0.139 / 0.078 | 0.167 / 0.109 | 0.145 / 0.084 |
| 4 | 0.610 / 0.917 | 0.464 / 0.794 | 0.647 / 0.906 | 0.139 / 0.078 | 0.163 / 0.105 | 0.145 / 0.084 |
| 5 | 0.548 / 0.899 | 0.403 / 0.777 | 0.605 / 0.885 | 0.139 / 0.078 | 0.160 / 0.102 | 0.145 / 0.084 |
| 6 | 0.494 / 0.893 | 0.355 / 0.776 | 0.572 / 0.870 | 0.139 / 0.078 | 0.159 / 0.100 | 0.145 / 0.084 |
| 7 | 0.449 / 0.896 | 0.325 / 0.772 | 0.550 / 0.870 | 0.139 / 0.078 | 0.156 / 0.097 | 0.145 / 0.084 |
| 8 | 0.414 / 0.892 | 0.313 / 0.771 | 0.539 / 0.862 | 0.139 / 0.078 | 0.154 / 0.094 | 0.144 / 0.084 |
| 9 | 0.387 / 0.898 | 0.303 / 0.778 | 0.535 / 0.873 | 0.140 / 0.078 | 0.152 / 0.091 | 0.143 / 0.083 |

Plots: `heatmaps.png`, `cross_attention_control.png`, `fp_grid.jpg` (16 FP images: prompts left to right, seeds 0 / 1 top / bottom).

### Verdict (rule above, steps 0–7)

- **A** sample-independence: ≥ 70% of pairs with within ≥ 0.8 at every s ≤ 7: **False** (pooled 0.607); KL ratio ≥ 3 among clearing triples: 0.782 (needs ≥ 0.5) → **A False**.
- **B** prompt-independence: diff/same ≥ 0.9 in 0.897 of triples (needs ≥ 0.7; median ratio 1.000) → **B True**.
- **C** compact residual (floor = min κ): median 95% side 127 tokens (needs ≤ 13); ≤ 21 in 0.005 (needs ≥ 0.9); full 127 in 0.624 → **C False**. Robustness, floor = median κ: median side 127, ≤ 21 in 0.005. Floor share of κ's mass: median 0.05 (min floor), 0.44 (median floor).
- s_max = **2** (contiguous from 0: 2).

**STOP: the phenomenon is specific to class-conditional DiT (A or B failed)**

Reading (added after the results; the rule and its verdict are unchanged). The pre-registered STOP label names A *or* B, but only A failed. B passed: the median diff-/same-prompt ratio is 1.000, while the cross-attention control sits near 0.08. So early self-attention is prompt-independent at 4096 tokens, and the failure is one of **duration**. Agreement clears the 70% bar at s = 0–2 (88%, 83%, 75%, 67% at s = 0–3) and then decays. That is 3 of 20 steps (15% of the schedule), against s_max = 14 of 50 (30%) on DiT-XL/2. C fails on its own as well: with the floor removed, the residual's 95% support is the full 127 × 127 at the median, as it was on DiT-XL/2 at 31 × 31. One caveat applies to both models and is untested: κ averages each offset without weighting by pair count, so far offsets (one pair at ±63) get full weight.

