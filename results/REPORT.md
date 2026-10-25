# Results REPORT (append-only, dated)

## 2026-10-25 — Stage 0 (bring-up) complete
- Ran: deterministic DDIM-50 sampling (seeds 0-7), bit-for-bit repeat check (PASS), timing, deliberate SIGKILL/resume test
  (PASS, bit-identical), FID reference stats (10k images). Details: `results/env.md`, `results/model_facts.md`.
- GPU-hours: 0.05.
- Findings that matter downstream:
  - 0.50 s/image at bs 32 (handoff expected 0.2-0.4) -> FID-2k ~17 min, FID-5k ~42 min per run.
  - Latents depend on batch layout (fp16 kernel choice): rel. MSE 4e-6 between bs 8 and bs 16. All matched-seed
    comparisons use a fixed layout (bs 32, seeds in order).
  - `ddim_sample` defaults to `clip_denoised=True` (clips latent x0 to [-1,1]); we pass False like DiT's sample.py.
  - BoA's GPTQ path is per-row independent, and its solver already supports a per-head stacked H_col, so the K row-group
    Hessians run in ONE solver call by packing groups as zero-padded "heads" (verified bit-identical to K separate
    calls, 3x faster). No solver code changed.
- Next decision needing a human: none.
