# Prior art: temporal decorrelation of quantization error (searched 2026-10-27)

**Per-step quantization in diffusion PTQ changes scales or bit-widths, never the grid phase.**
- **TDQ** (So et al., NeurIPS 2023): an MLP maps the timestep to per-step activation quantization intervals.
- **TFMQ-DM** (Huang et al., CVPR 2024): Finite-Scale Calibration (FSC) calibrates time-information-related layers
  per step.
- **PTQD** (He et al., NeurIPS 2023): step-aware mixed precision (bit-widths by step) plus correlated/uncorrelated
  error correction.
- **TCAQ-DM**: timestep-channel adaptive activation quantization.

**Static quantizers.** DMQ (ICCV 2025) is static per-tensor with learned channel scaling and fixed weights.

**What no diffusion PTQ paper does, as far as we found:**
- alternate between two (or more) weight roundings across sampling steps;
- dither the activation grid over time, i.e. change the offset or phase of a fixed static grid from step to step.

**Classical subtractive dither:**
- L. Schuchman, "Dither signals and their effect on quantization noise", IEEE Trans. Commun. Technol., 1964.
- R. M. Gray and T. G. Stockham, "Dithered quantizers", IEEE Trans. Inf. Theory, 1993.

Here the dither is deterministic and known, and it is subtracted exactly (x → Q(x + o_t) − o_t). The open question is
not the per-step error statistics, which dither does not change in size. It is whether the many-step sampler heals
error that is decorrelated across steps (cf. this repo's gate-1 healing and band-1 coherence measurements).

## Positioning for the feedback / structured-dither protocol (`rule_feedback.md`; added 2026-10-27)
Citations as supplied in the handoff; not independently verified here.
- **DRIFT** (DAC 2026): an empirical self-correction experiment after injected errors. Our distinction: a controlled
  comparison of temporal patterns using real recorded error sequences (stage 1), and its use in quantizer design.
- **Sigma-delta quantized networks** (Zambrano & Bohte, 2016): prior art for error feedback in quantized networks.
- **Spatiotemporal blue-noise dithering** (graphics): prior art for spatial and temporal phase assignment.
- **Q-Drift** and **DNS**: sampler-side corrections, the baselines to compare against (stage 5).
- **Claim:** deliberately constructing cross-step error relationships that the sampler suppresses, with evidence that
  those relationships explain the gain.
