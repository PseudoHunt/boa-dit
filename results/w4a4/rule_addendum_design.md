# Addendum (2026-10-26, before any PixArt design measurement on the corrected base): design-side prompt sets

**Rule.** Design-side measurements on PixArt-Σ use calibration prompts or a held-out prompt set only, never an MJHQ or
sDCI prompt used for screening or final eval. Every result JSON states its `prompt_set`.
- **Design-side** means anything that fits or tunes a method: the band-gain table, the CFG ratio, layer attribution,
  rank allocation, filter fitting, ρ.

**Design set.** deepcompressor's 128 calibration prompts (`qdiff.yaml`, the `random.Random(0)` subset, sorted), numbers
32–63.
- None is in our calibration set, which is numbers 0–31.
- None is an MJHQ or sDCI prompt.
- Seed = the prompt's index within the design set.
- The design cache (`ckpt/w4a4/design_cache.pt`) has the same layout as the screening cache: 32 prompts × steps 0, 2, …,
  18 × 2 branches, bf16 model.

**Superseded.** The earlier PixArt CFG ratio (14.9, on the original base) was computed on the screening cache. It is
kept only as an informational number and recomputed on the design set. The band-gain table had not produced any result
when it was stopped.

**Screening and final eval** are unchanged: MJHQ-64 images (`rule_addendum_screen.md`), then MJHQ-1K. Those prompts are
used only for evaluation.
