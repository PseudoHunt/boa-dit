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
