## Answers (written 2026-10-31 after the queue finished)

**Does referencing the anchor help?** Not demonstrated under these conditions.
- On the fresh 128-image confirmation, the primary practical contrast RDB/GB is **0.988 [0.607, 1.556]**: RDB better on 70/128, ΔLPIPS +0.0008
  [−0.0045, +0.0060]. The reading is **inconclusive**.
- The development gate passed on RDB (0.883 [0.588, 1.301] on 32 inspected images), but the confirmation does not reproduce a gain.
- The full-state version RFB/GB is 1.520 [0.799, 2.732]. Its ΔLPIPS CI is wholly above zero (+0.0085 [+0.0007, +0.0184]), so it is perceptually
  worse. The full-state mechanism therefore shows no benefit and tends to hurt.

**Does it survive DCT compression?** There is no gain to survive.
- Diagnostics show why RDB behaves almost like GB: the 4-term token DCT keeps only about 5% of the anchor rounding energy. E[Pm − m] = 3.95e5 of
  E[m_raw] = 4.15e5 (8 dev images, all sites and intervals), so the stored reference is mostly discarded.
- RDB/RFB is 0.650 [0.368, 1.167], i.e. compression does not hurt relative to full state.

**Other contrasts** (exploratory, unadjusted):
- RDB/ACB 0.622 [0.399, 1.006], with ΔLPIPS −0.0108 [−0.0173, −0.0045]. Replacing the old feedback stack with R-DCT4 largely removes its harm, which
  replicates (ACB/GB 1.588). This establishes only repair of the harmful stack, not a gain over GB.
- RD/G is 1.052 [0.968, 1.145]. RDB/RD is 0.311: the bridge again provides the large gain, as does GB/G at 0.332.

**Mechanism diagnostics** (8 dev images; descriptive, each arm on its own trajectory):
- R-full more than doubles the per-site local held error. Here q − x means quantizer output minus its input (the activation rounding error), summed
  over all sites. E[q − x] at held calls is 2.95e6 for RFB vs 1.48e6 for RDB. The fed reference + state energy is 1.76e6, against 7.1e4 for R-DCT4.
- R-full also lowers interval predictability: cos(H, v) is 0.498 vs 0.675–0.686 for the other arms. The short-FP-replay cosine of the actual
  deviation with the bridge estimate falls to 0.01 / 0.22 / 0.61 at starts 0 / 20 / 40, against about 0.55–0.74 for GB, ACB and RDB.
- So the local identity q_t − x_t − m_a = s_{t−1} − s_t holds as arithmetic (identity residual about 1e-15 relative), but pinning each site's held
  error to its anchor rounding does not translate into a smaller whole-denoiser discrepancy at the endpoint here.
- This is consistent with section 3.1's caveat that the sampler holds a whole-denoiser discrepancy, not each site's m_a. It does not prove that
  explanation.
- Clipping error (xcl − xs, about 2.4e5) is excluded from the reference by design. Guard discard and residual clamp are negligible (< 10).

**Cost:**
- State + reference per image (both CFG rows, fp16, measured from tensors): GB/ACB 3.61 MB, RDB 7.23 MB, RFB 462 MB.
- RFB needs batch 16 (it is OOM at 32 on the A30).
- Warm trajectory time at batch 16 is essentially equal across arms (GB 45.7 s, RFB 45.9 s, RDB 46.3 s; FP50 9.8 s). This is fake-quant simulation and
  gives no packed-kernel or real-device claim.

**Scope:** DiT-XL/2 256, DDIM-50, CFG 1.5, unrefitted W4 + static A8, K4 FP anchors, λ = 1, a single cohort. Optional ZB was not run. Nothing here is a
universal impossibility claim. Total spend is 1.324 of 3.0 A30-h.
