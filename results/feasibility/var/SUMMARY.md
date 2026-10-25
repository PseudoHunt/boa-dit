# Part V — VAR-d30 (ImageNet 256): query-aware key selection, feasibility summary

**Verdict: neither outcome of the rule: headroom exists, positional-25 is below oracle-25 by more than 1.5 points but not by the 3-point, non-overlapping-CI margin.**

- Gap 1, oracle vs full (is sparse attention at 25% harmless?): label accuracy +1.2 points, FID-1k -0.28 → headroom **True**.
- Gap 2, positional vs oracle (does a fixed rule capture it?): -1.7 points (CIs overlap), FID-1k +0.77; random-25 vs positional-25: -14.7 points.

| arm | FID-1k | label acc (95% Wilson CI) | class differs from full |
| --- | ---: | --- | ---: |
| full | 47.84 | 93.7% [92.0, 95.0] | 0.0% |
| oracle25 | 47.56 | 94.9% [93.4, 96.1] | 4.3% |
| positional25 | 48.33 | 93.2% [91.5, 94.6] | 6.0% |
| random25 | 60.44 | 78.5% [75.8, 80.9] | 20.4% |
| stock | 47.89 | 94.8% [93.2, 96.0] | 3.0% |

FID-1k against the ADM ImageNet-256 reference is biased upward at 1,000 samples; read differences, not levels. Demo sampling settings (cfg 4) also favour quality over diversity, which raises FID.

## Attention statistics (64 images, cond branch; medians over the 30 blocks of per-head medians over queries)

| query scale | mass on same-scale keys | fraction of allowed keys for 90% / 95% mass | positional share of oracle top-25% mass | positional heads (median ≥ 0.9) | cross-image Jaccard of top-25% sets |
| --- | ---: | --- | ---: | ---: | ---: |
| 10×10 | 0.50 | 0.12 / 0.20 | 0.84 | 41% | 0.49 |
| 13×13 | 0.53 | 0.10 / 0.18 | 0.89 | 43% | 0.49 |
| 16×16 | 0.61 | 0.08 / 0.15 | 0.90 | 47% | 0.49 |

Plots: `stats.png` (positional share and 95%-mass fraction over block × scale), `grid.png` (16 matched images per arm). Facts and deviations: `model_facts.md`. Rule: `../rule.md`.

Note: at 256 px VAR's attention and KV cache are small (680 keys at the last scale). Any positive result here would need reproducing on Infinity-2B at 1024 px before it says anything about speed.

GPU-hours (Part V, logged): 0.22 of 1.5. About 0.05 h of setup timing runs were not logged.
