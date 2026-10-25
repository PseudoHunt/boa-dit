# Part C — CogVideoX-2B (480p, 49 frames): query-aware key selection, feasibility summary

| arm | latent PSNR vs full (dB) | CLIP ViT-L/14 (8 frames) | seconds per video |
| --- | ---: | ---: | ---: |
| stock | — (reference) | 33.97 | 158 |
| full | 35.9 (vs stock) | 34.02 | 345 |
| oracle10 | pending | | |
| window10 | 22.1 | 30.27 | 415 |
| random10 | 23.2 | 23.32 | 357 |

Parity: full vs stock latents after step 0, relative difference 5.4e-05 (tolerance 1e-2).

## Attention statistics (128 sampled video queries; prompt 1 unless stated; medians over queries, means over heads, medians over blocks)

| step | top-1% mass | top-5% | top-10% | text mass | locality (top-10% mass in window) | window overlap (Jaccard) | prompt Jaccard |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.21 | 0.37 | 0.46 | 0.283 | 0.18 | 0.09 | 0.37 |
| 10 | 0.42 | 0.66 | 0.74 | 0.111 | 0.35 | 0.13 | 0.27 |
| 11 | 0.44 | 0.67 | 0.75 | 0.107 | 0.35 | 0.13 | 0.27 |
| 25 | 0.51 | 0.73 | 0.79 | 0.085 | 0.41 | 0.13 | 0.27 |
| 40 | 0.55 | 0.75 | 0.81 | 0.091 | 0.44 | 0.13 | 0.28 |

Step stability (top-10% sets, step 10 vs 11, prompt 1): Jaccard 0.85. High stability is the caching headroom that TeaCache and PAB already exploit.

GPU-hours (Part C, logged): 0.47 of 2.5. About 0.25 h of timing tests were not logged. Facts and deviations: `model_facts.md`. Rule: `../rule.md`.
