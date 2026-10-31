# Handoff 2026-10-31 — answers to the four questions
Covers sr1 (SANA, branch sana-repeatability), dar1 and w8s1 (branch dit-anchor-refit). Spend 2.132 of 4.0 A30-h: sr1 0.000, dar1 1.856, w8s1 0.276.
Nothing running.

**1. Was SANA variation localized or reduced, and is the reproducible discrepancy reusable?**
- *Localized (earlier, snk1b):* every Nunchaku layer is individually non-bitwise, and divergence compounds through depth. Two float-atomic reductions
  were found; PR #820 covers only one (`lora.cuh`), and the LiteLA epilogue atomic is untouched.
- *Not reduced:* the track was parked by user decision before any sr1 measurement. There was no build, CUDA-13 install or kernel change.
- *Reusability:* on the repeat mean of the current implementation, the K4 held residual is 1.31 > 1, so that mean is poorly reusable.
  - This is a property of the repeat mean, **not** of a hypothetical deterministic implementation, which was not built.
  - It is a local prediction-error energy, **not** endpoint quality.
- Preserved: the reproducer scripts and split-repeat statistics. mjhq1k 464–527 is unopened.

**2. Does BA beat a strong local-refit control on full-trajectory fresh-seed fidelity at the same inference architecture?**
- **Not confirmed.**
- Development (32 images): BA@330/BL@330 = 0.789, and the promotion rule passed.
- Fresh confirmation (124000–124127, 128 seeds): **BA/BL = 0.835 [0.489, 1.423]**, so the latent result is **inconclusive**, BA better on 79/128.
- Secondary (exploratory, unadjusted):
  - LPIPS is significantly better, −0.0063 [−0.0118, −0.0005];
  - BA beats the original branch (B0): 0.665 [0.420, 0.964];
  - vs the time-matched control 0.848 and vs the archived local refit 0.923, both inconclusive.
- Mechanism hint (descriptive, not causal): BA's standalone per-step error is ~5.5× larger than B0's, while its within-interval error is smaller and
  better fitted by the bridge. BA looks anchor-sampler-specific.
- One training seed, so this is a pilot.

**3. Does W8-K2 improve the fidelity / anchor-spacing curve, and what measured kernel cost would an efficiency claim need?**
- **Yes, for accuracy.** On 64 fresh seeds: W8-K2/FP-K4 **0.341 [0.142, 0.802]** (52/64, ΔLPIPS −0.016) and W8-K2/W8-K4 0.273 [0.101, 0.761].
- W8-K2 sits near W8's own floor: vs H8_50 1.054, inconclusive. W8 error is not negligible (H8_50 vs FP50 = 0.00226).
- **Cost:** K2 makes 50 cheap + 25 W8 calls vs 50 + 13 FP. Equal anchor cost needs **C_W8/C_FP ≤ 13/25 ≈ 0.52**, before overheads.
  - The fake-quant simulation gives 1.17. That is not a kernel cost, so **no efficiency claim**.
  - A real W8 kernel at ≤ ~0.5× the FP call cost would be needed, and it is untested.

**4. What is still missing?**
- **Training-seed replication** of BA (and a matched BL).
- BA on the plain (non-anchored) sampler.
- **Generation quality** (FID / ImageReward; everything here is fidelity to FP50 or H20).
- **Real-kernel transfer**: W8 and W4A8 kernel costs on DiT; SANA deterministic kernels.
- **Numerical correctness:** gradients are validated only for the smooth (quantizer-off) case; the static quantizer is an STE.
- The SANA bridge port and the deterministic build, both parked.
