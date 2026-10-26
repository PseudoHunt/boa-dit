# Prior art for the objective diagnostics (recorded 2026-10-26)

- **GAMP** (arXiv 2607.08241, Jul 2026) calibrates activation clipping thresholds and bit allocation on the *guided*
  (CFG-combined) output.
  - Test bed: a 6M-parameter CIFAR-10 UNet.
  - It has no Hessian, no weight solver, and no DiT.
- **TCEC** (arXiv 2508.12094, Aug 2025) models per-step error propagation as δ_{t−1} = A_t δ_t + B_t ε_t.
  - It drops the Jacobian term, which it measures at 0.1–0.5% on SDXL.
  - It adds a runtime channel-scaling correction on top of SVDQuant.
  - It has no frequency analysis and does not change the per-layer objective.
- Neither has a CFG-combined Hessian or a frequency-weighted Hessian. Those are the two losses tested here (D1, D2).
- **Related:** H-SVDQuant (arXiv 2610.11315) for the deflate-and-carry core; see `results/w4a4/prior_art.md`.
