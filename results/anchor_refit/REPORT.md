# dar1 — anchor-aware refit of the DiT rank-8 branch: report

Protocol: `PROTOCOL.md` (pre-registered 2026-10-31; dated amendments keep every failure). **Complete.** Spend **1.856 of 2.5 A30-h** (ledger
tag dar1). Handoff total (sr1 + dar1 + w8s1): **2.132 of 4.0**. Nothing running. Single training seed: a **pilot**.

## Configuration (abn1 CB / LCB, verified bitwise)
- DiT-XL/2 256, DDIM-50, CFG 1.5.
- Cheap model: W4A8 static (common smoothed rank-8 W4 base + static single8 A8), feedback off.
- FP anchors K4, bridge λ = 1, 50 cheap + 13 FP calls, 12 corrections.
- Only the 112 rank-8 carries (V, delta) are trained. Adam lr 1e-4, batch 4, training seeds 60100–60163.
- BA and BL use the same draw list of (images, window start).

## Correctness gate (checks.json)
The gate **passes** after dated fixes. It took 3 check attempts, 2 OOM fixes, a corrected count expectation and an fp64 harness defect fix; all are
on record.

**Passed:**
- Untrained differentiable inference vs validated B0: endpoint rel 0.046 ≤ 0.05, step-1 rel 1.2e-7.
- Export ↔ install bitwise (untrained and perturbed).
- abn1 CB and LCB reproduce bitwise at batch 32.
- Zero-update invariance; tables and W4 codes unchanged.
- Window counts (full windows 9 cheap / 3 FP / 2 bridges; terminal 6 / 2 / 1 and 2 / 1 / 0), and the a = 0 window equals the full-rollout corrected
  x₈ bitwise.
- Resume bitwise.
- Gradients reach b0 and b27; teacher parameters get none; the teacher-latent derivative is retained (it changes the gradient by 54%).
- **Finite differences**, after fixing a harness defect: `apply_smoothing`'s pre-hook casts to fp32, which had made the "fp64" check noisy.
  - Quantizer-off fp64, clean second-order convergence: gradient direction 6.5e-5 → 6.7e-9; random all-site direction 5.1e-5 → 2.1e-8.
  - Latent direction (fp32) 7.3e-4.

**Not resolved / not claimed:**
- Earlier fp32 and pre-fix fp64 random-direction FDs were unresolved (recorded).
- The static quantizer is an STE; its quantized FD is reported, with no exactness claim.

## Training (one seed)
| run | updates | training time | notes |
|---|---|---|---|
| BA | 330 (checkpoints 82 / 165 / 330) | 1378 s (0.388 A30-h) | 4.2 s/update vs 3.6 profiled: ~9% over the 0.35 h planning allocation |
| BL | 330 update-matched, then BL-tm at 565 | 1261 s (0.356 A30-h) | BL-tm cap = min(BA time, 0.35 h) = 1260 s, so BL-tm had ~9% *less* time than BA (slightly favours BA) |

## Development selection (60200–60231, inspected; 32 images; full 50-step sampler)
| arm | latent rel. MSE (mean) | LPIPS |
|---|---|---|
| B0 | 0.00337 | 0.0251 |
| BL-arch (archived local refit) | 0.00374 | 0.0218 |
| BL@82 / 165 / 330 | 0.00288 / 0.00279 / **0.00264** | 0.0220 / 0.0209 / 0.0206 |
| BL-tm (565) | 0.00298 | 0.0225 |
| BA@82 / 165 / 330 | 0.00325 / 0.00305 / **0.00208** | 0.0220 / 0.0227 / 0.0168 |

- Selected: BA@330 and BL@330 (lowest dev mean); strongest comparator BL@330.
- BA/BL = **0.789** (dev CI [0.578, 1.047], exploratory), ΔLPIPS −0.0038 → **promotion passed**.

## Fresh confirmation (124000–124127, 128 seeds; frozen arms)
| arm | mean | median | PSNR | LPIPS |
|---|---|---|---|---|
| B0 | 0.00545 | 0.00140 | 36.08 | 0.0298 |
| **BA@330** | **0.00362** | 0.00099 | 37.62 | **0.0201** |
| BL@330 | 0.00434 | 0.00153 | 36.31 | 0.0264 |
| BL-tm | 0.00427 | 0.00113 | 36.86 | 0.0247 |
| BL-arch | 0.00393 | 0.00121 | 36.94 | 0.0241 |

| contrast | ratio of means [95% paired CI] | wins | ΔLPIPS [95% CI] |
|---|---|---|---|
| **BA@330 / BL@330 (primary)** | **0.835 [0.489, 1.423]** | 79/128 | **−0.0063 [−0.0118, −0.0005]** |
| BA@330 / B0 | 0.665 [0.420, 0.964] | 91/128 | −0.0097 [−0.0150, −0.0049] |
| BA@330 / BL-tm | 0.848 [0.525, 1.234] | 79/128 | −0.0046 [−0.0101, +0.0001] |
| BA@330 / BL-arch | 0.923 [0.560, 1.394] | 79/128 | −0.0040 [−0.0093, +0.0007] |
| BL@330 / B0 | 0.797 [0.608, 1.052] | 72/128 | −0.0034 [−0.0079, +0.0011] |

- **Primary reading: inconclusive.** The latent ratio CI includes 1, so this is not a confirmed fidelity improvement. LPIPS is significantly better for
  BA, and BA beats the original branch (B0) on latent error with the CI below 1.
- Secondary contrasts are exploratory and unadjusted. On confirmation the archived BL-arch had a lower mean than the dev-selected BL@330. BA vs BL-arch
  is 0.923, inconclusive.

## Diagnostics (60208–60215; descriptive)
| | raw per-step prediction error | held-interval energy H | bridge residual / H | lag-1 / lag-8 change energy | held-out BA window loss | held-out per-step loss |
|---|---|---|---|---|---|---|
| B0 | 2,987 | 18.5 | 0.531 | 1,164 / 3,334 | 0.908 | 0.715 |
| BL-sel | 2,767 | 17.9 | 0.532 | 1,131 / 3,119 | 0.876 | 0.668 |
| **BA-sel** | **16,468** | 16.2 | 0.481 | 999 / 3,896 | **0.729** | **11.9** |
| BL-arch | 2,954 | 18.3 | 0.530 | 1,147 / 3,282 | 0.887 | 0.702 |

Measured on teacher-forced FP50 states, which are matched across arms. Live-trajectory values are similar and in diag.json.
- **Observed:** BA's raw cheap-model prediction error is **~5.5× larger** than B0's, and its per-step local loss is 17× worse. Its within-interval
  error structure is smaller and better captured by the bridge: lower H, lower lag-1/2 change energy, a lower bridge residual and higher cos(H, v)
  (0.72 vs 0.69).
- **Hypothesis (not established):** the anchor-aware objective trades standalone accuracy for an error that anchor reuse and the bridge cancel. The
  BA branch is therefore specific to the anchored sampler and would likely be much worse as a plain every-step cheap model. That was not measured.
- Short-window improvement (held-out window loss 0.73 vs 0.88–0.91) is reported separately from full-trajectory results.

## Untested
Training-seed replication, plain-sampler (non-anchored) use of the BA branch, generation quality (FID / ImageReward), real-kernel cost, other
K / λ / models. **124000–124127 is now evaluated.**
