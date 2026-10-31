# rfb1 — anchor-referenced feedback + anchor reuse + bridge (pilot R): report

Protocol: `PROTOCOL.md` (amendments dated). Spend: **1.324 of 3.0 A30-h** (ledger tag rfb1).

## Status
| stage | status |
|---|---|
| profile | ? |
| checks | pass |
| dev | complete |
| conf | complete |
| diag | complete |

Frozen batch size **16** (R-full peak reserved at 16: 15.3 GiB; 32: OOM). R-full state + reference: 462 MB per image (both CFG rows).

## Correctness gate
Overall: **pass**.
- S1 overnight reproduction at batch 32: {'FP50': True, 'G': True, 'GB': True, 'ACB': True}
- S2 λ=0: {'GB': True, 'ACB': True, 'RFB': True, 'RDB': True}; S3 K1=FP50: {'G': True, 'fb': True, 'R-full': True, 'R-dct': True, 'J': True}; S4 counts/inputs: True
- S5 independence/reset: {'RFB': {'others_unchanged': True, 'reset_between_batches': True}, 'RDB': {'others_unchanged': True, 'reset_between_batches': True}, 'ACB': {'others_unchanged': True, 'reset_between_batches': True}}; unit row independence: {'R-full': True, 'R-dct': True, 'J': True, 'fb': True}
- S6 resume: {'RDB@8': True, 'RDB@9': True, 'RFB@8': True, 'RFB@9': True, 'ACB@8': True, 'ACB@9': True}; S7 instrumentation on/off: {'RFB': True, 'RDB': True}
- S8 no-anchor R-DCT4 = feedback: True; R anchor call = G: {'RF': True, 'RD': True}
- S9 accounting identity rel. residual energy: RFB 1.32e-15, RDB 6.12e-14
- S10 bytes per image (incl. CFG): ACB: state 3.61 MB + ref 0.00 MB; RFB: state 231.21 MB + ref 231.21 MB; RDB: state 3.61 MB + ref 3.61 MB
- DDIM coefficients: max rel 8.4e-07; synthetic: True

## Development (60200–60231, inspected, exploratory)
| arm | latent rel. MSE mean | median | PSNR | LPIPS |
|---|---|---|---|---|
| G | 0.01645 | 0.00699 | 30.97 | 0.0656 |
| GB | 0.00385 | 0.00106 | 36.13 | 0.0246 |
| ACB | 0.00571 | 0.00234 | 34.05 | 0.0345 |
| RFB | 0.00580 | 0.00192 | 35.21 | 0.0325 |
| RDB | 0.00340 | 0.00142 | 36.13 | 0.0218 |
| RD | 0.01784 | 0.00725 | 30.89 | 0.0683 |

| contrast | ratio of means [95% CI] | wins | ΔLPIPS [95% CI] | ΔPSNR |
|---|---|---|---|---|
| RDB / GB | 0.883 [0.588, 1.301] | 16/32 | -0.0028 [-0.0098, +0.0029] | -0.00 |
| RFB / GB | 1.506 [0.732, 2.458] | 10/32 | +0.0079 [-0.0022, +0.0204] | -0.92 |
| RDB / RFB | 0.586 [0.290, 1.343] | 18/32 | -0.0107 [-0.0286, +0.0020] | +0.92 |
| RDB / ACB | 0.595 [0.331, 0.993] | 24/32 | -0.0127 [-0.0253, -0.0018] | +2.07 |
| RFB / ACB | 1.015 [0.473, 2.022] | 24/32 | -0.0020 [-0.0184, +0.0203] | +1.15 |
| RD / G | 1.085 [1.021, 1.158] | 12/32 | +0.0027 [-0.0005, +0.0059] | -0.08 |
| RDB / RD | 0.191 [0.082, 0.428] | 28/32 | -0.0465 [-0.0832, -0.0226] | +5.24 |
| ACB / GB | 1.484 [0.864, 2.584] | 5/32 | +0.0100 [-0.0042, +0.0238] | -2.07 |
| GB / G | 0.234 [0.096, 0.537] | 28/32 | -0.0410 [-0.0767, -0.0185] | +5.16 |

Gate (run confirmation iff RFB or RDB dev mean error <= 0.95 x GB and mean LPIPS <= LPIPS(GB) + 0.005): RFB: ratio 1.506, ΔLPIPS +0.0079 → fail; RDB: ratio 0.883, ΔLPIPS -0.0028 → pass → **confirmation run**.

## Confirmation (122000–122127, fresh, 128 images)
| arm | latent rel. MSE mean | median | PSNR | LPIPS |
|---|---|---|---|---|
| G | 0.01376 | 0.00615 | 31.34 | 0.0587 |
| GB | 0.00456 | 0.00148 | 36.58 | 0.0242 |
| ACB | 0.00725 | 0.00257 | 34.24 | 0.0358 |
| RFB | 0.00694 | 0.00183 | 35.63 | 0.0328 |
| RDB | 0.00451 | 0.00143 | 36.58 | 0.0250 |
| RD | 0.01448 | 0.00540 | 31.23 | 0.0616 |

| contrast | ratio of means [95% CI] | wins | ΔLPIPS [95% CI] | ΔPSNR |
|---|---|---|---|---|
| RDB / GB | 0.988 [0.607, 1.556] | 70/128 | +0.0008 [-0.0045, +0.0060] | +0.00 |
| RFB / GB | 1.520 [0.799, 2.732] | 47/128 | +0.0085 [+0.0007, +0.0184] | -0.95 |
| RDB / RFB | 0.650 [0.368, 1.167] | 80/128 | -0.0078 [-0.0171, -0.0002] | +0.95 |
| RDB / ACB | 0.622 [0.399, 1.006] | 95/128 | -0.0108 [-0.0173, -0.0045] | +2.35 |
| RFB / ACB | 0.957 [0.487, 1.783] | 79/128 | -0.0031 [-0.0125, +0.0079] | +1.40 |
| RD / G | 1.052 [0.968, 1.145] | 66/128 | +0.0029 [-0.0007, +0.0067] | -0.12 |
| RDB / RD | 0.311 [0.207, 0.476] | 116/128 | -0.0366 [-0.0455, -0.0279] | +5.36 |
| ACB / GB | 1.588 [0.949, 2.738] | 33/128 | +0.0116 [+0.0049, +0.0186] | -2.34 |
| GB / G | 0.332 [0.212, 0.514] | 115/128 | -0.0345 [-0.0432, -0.0260] | +5.24 |

Readings: RDB/GB (primary practical): inconclusive; RFB/GB (secondary full-state mechanism): inconclusive; RDB/RFB (compression): inconclusive; RDB/ACB (repair of the harmful stack only): inconclusive; RD/G (interaction (no additive claim)): inconclusive; RDB/RD (interaction): useful (>= 5% lower mean latent MSE, CI upper < 1, no clear LPIPS regression)

Secondary intervals exploratory/unadjusted; nonsignificance is not equivalence.

## Diagnostics (60208–60215)
| arm | endpoint rel. MSE | raw | held | jump | cos(H,v) | λ_LS frozen window | replay cos(dx,v) 0/20/40 |
|---|---|---|---|---|---|---|---|
| G | 0.01252 | 2657 | 1086 | 676 | 0.684 | 0.878 | 0.58 / 0.64 / 0.74 |
| GB | 0.00303 | 2776 | 1151 | 688.8 | 0.686 | 0.909 | 0.58 / 0.67 / 0.72 |
| ACB | 0.00493 | 2762 | 1217 | 702.2 | 0.665 | 0.915 | 0.55 / 0.64 / 0.72 |
| RFB | 0.00292 | 3191 | 941.2 | 691.7 | 0.498 | 0.427 | 0.01 / 0.22 / 0.61 |
| RDB | 0.00352 | 2756 | 1157 | 682.5 | 0.675 | 0.902 | 0.55 / 0.66 / 0.71 |
| RD | 0.01179 | 2676 | 1098 | 666.5 | 0.667 | 0.868 | 0.55 / 0.63 / 0.73 |

R accounting (all intervals, all sites; energy sums):
- RFB: E[(I-P)r] 0, E[Pm-m] 0, E[clamp a=r0-r] 0.3604, E[fed] 1.764e+06, E[guard_discard] 8.822, E[identity_residual] 3.968e-09, E[m_raw] 4.15e+05, E[q-x] 2.951e+06, E[q-x]_anchor 4.819e+05, E[q-xcl-ref] 1.529e+06, E[r0] 9.628e+05, E[ref_rec-m] 0.003685, E[ref_storage] 0.003685, E[s_prev] 5.815e+05, E[state_storage] 0.008388, E[uc-u] 0, E[xcl-xs] 2.447e+05, E[xcl-xs]_anchor 6.695e+04
- RDB: E[(I-P)r] 1.114e+06, E[Pm-m] 3.945e+05, E[clamp a=r0-r] 0.3379, E[fed] 7.137e+04, E[guard_discard] 0.3957, E[identity_residual] 8.272e-08, E[m_raw] 4.15e+05, E[q-x] 1.476e+06, E[q-x]_anchor 4.815e+05, E[q-xcl-ref] 1.166e+06, E[r0] 1.154e+06, E[ref_rec-m] 3.945e+05, E[ref_storage] 0, E[s_prev] 2.427e+04, E[state_storage] 0, E[uc-u] 0, E[xcl-xs] 2.436e+05, E[xcl-xs]_anchor 6.641e+04
- RD: E[(I-P)r] 1.115e+06, E[Pm-m] 3.946e+05, E[clamp a=r0-r] 0.3332, E[fed] 7.13e+04, E[guard_discard] 0.3765, E[identity_residual] 8.187e-08, E[m_raw] 4.151e+05, E[q-x] 1.465e+06, E[q-x]_anchor 4.778e+05, E[q-xcl-ref] 1.166e+06, E[r0] 1.154e+06, E[ref_rec-m] 3.946e+05, E[ref_storage] 0, E[s_prev] 2.428e+04, E[state_storage] 0, E[uc-u] 0, E[xcl-xs] 2.334e+05, E[xcl-xs]_anchor 6.269e+04

Timing / memory (warm synchronized trajectory at the frozen batch; fake-quant simulation, not packed-kernel evidence):
- FP50: 9.8 s per 16-image trajectory; state + reference 0.00 MB/image
- GB: 45.7 s per 16-image trajectory; state + reference 3.61 MB/image
- ACB: 45.7 s per 16-image trajectory; state + reference 3.61 MB/image
- RFB: 45.9 s per 16-image trajectory; state + reference 462.42 MB/image
- RDB: 46.3 s per 16-image trajectory; state + reference 7.23 MB/image
