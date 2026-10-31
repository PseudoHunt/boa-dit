# cfb1 — zero-reference feedback with joint gains and bridge strength (pilot J): report

Protocol: `PROTOCOL.md` (amendments dated). Spend: **1.158 of 3.0 A30-h** (ledger tag cfb1).

## Status
| stage | status |
|---|---|
| profile | complete |
| checks | pass |
| fit | complete |
| val | complete |
| conf | skipped (validation gate failed; confirmation cohort 121000-121255 not opened) |
| diag | complete |

Frozen batch 32; profiled per-trajectory: J 89.9 s, FP50 19.2 s; planned total 2.76 h.

## Correctness gate
Overall **pass**.
- overnight reproduction (batch 32): {'FP50': True, 'J(0,0)_nobridge=G': True, 'J(0,0,1)=GB': True, 'J(1,1,1)=ACB': True, 'J(1,1,1)_state=ACB_state': True}
- J(1,1)=AC incl. state: True; λ0=no bridge: True; K1=FP50: True
- counts: True; independence True; reset True; resume {'8': True, '9': True}; instrumentation True
- gain placement: anchor g=0 = G call True; between g=1 = AC call incl. state True

## Fit (120000–120031): complete 27-point surface (mean latent rel. MSE vs FP50)

| g_anchor, g_between | λ=0.5 | λ=1 | λ=1.5 |
|---|---|---|---|
| 0, 0 | 0.00686 | 0.00446 | 0.00725 |
| 0, 0.5 | 0.00834 | 0.00407 | 0.00722 |
| 0, 1 | 0.01364 | 0.01123 | 0.01166 |
| 0.5, 0 | 0.00775 | 0.00310 | 0.00633 |
| 0.5, 0.5 | 0.01017 | 0.00550 | 0.00754 |
| 0.5, 1 | 0.01165 | 0.01041 | 0.01224 |
| 1, 0 | 0.01126 | 0.00731 | 0.01364 |
| 1, 0.5 | 0.01665 | 0.01270 | 0.01409 |
| 1, 1 | 0.01024 | 0.00526 | 0.00935 |

Selected (frozen before validation): B = J(0,0,1) (fit 0.00446), U = J(0,0,1) (fit 0.00446), J = J(0.5,0,1) (fit 0.00310)

## Validation (120100–120131)
| role | configuration | latent rel. MSE mean | median | PSNR | LPIPS |
|---|---|---|---|---|---|
| B0 | J(0,0,1) | 0.00446 | 0.00169 | 36.28 | 0.0261 |
| B | J(0,0,1) | 0.00446 | 0.00169 | 36.28 | 0.0261 |
| U | J(0,0,1) | 0.00446 | 0.00169 | 36.28 | 0.0261 |
| N | J(1,1,1) | 0.00673 | 0.00412 | 34.26 | 0.0336 |
| H | J(0,1,1) | 0.00862 | 0.00310 | 33.89 | 0.0374 |
| J | J(0.5,0,1) | 0.00563 | 0.00213 | 35.59 | 0.0295 |

| contrast | ratio of means [95% CI] | wins | ΔLPIPS [95% CI] | ΔPSNR |
|---|---|---|---|---|
| J / B | 1.262 [0.830, 2.183] | 12/32 | +0.0034 [-0.0047, +0.0145] | -0.68 |
| J / U | 1.262 [0.830, 2.183] | 12/32 | +0.0034 [-0.0047, +0.0145] | -0.68 |
| J / N | 0.837 [0.549, 1.180] | 23/32 | -0.0041 [-0.0111, +0.0036] | +1.33 |
| N / B0 | 1.508 [1.202, 2.135] | 7/32 | +0.0074 [+0.0001, +0.0145] | -2.01 |
| H / B0 | 1.931 [1.007, 3.880] | 5/32 | +0.0112 [+0.0004, +0.0266] | -2.38 |
| J / H | 0.654 [0.259, 1.536] | 20/32 | -0.0079 [-0.0272, +0.0081] | +1.70 |
| J / B0 | 1.262 [0.830, 2.183] | 12/32 | +0.0034 [-0.0047, +0.0145] | -0.68 |

Gate (J nonzero feedback, mean error <= 0.95 x B, mean LPIPS <= LPIPS(B) + 0.005): J nonzero feedback True, ratio J/B 1.262, ΔLPIPS +0.0034 → **fail**

## Confirmation
skipped (validation gate failed; confirmation cohort 121000-121255 not opened)

## Diagnostics (120000–120007)
| live configuration | endpoint | E‖H‖² | E⟨H,v⟩ | E‖v‖² | cos | λ_LS (frozen window, not used) | anchor raw | between raw |
|---|---|---|---|---|---|---|---|---|
| J(0,0,1) | 0.00428 | 19.76 | 10.55 | 12.2 | 0.680 | 0.865 | 609 | 2278 |
| J(0,0.5,1) | 0.00379 | 21.16 | 10.95 | 12.45 | 0.675 | 0.879 | 626.5 | 2327 |
| J(0,1,1) | 0.00535 | 22.42 | 10.81 | 12.35 | 0.649 | 0.875 | 610.9 | 2315 |
| J(0.5,0,1) | 0.00338 | 20.56 | 10.63 | 12.43 | 0.665 | 0.855 | 616.1 | 2291 |
| J(0.5,0.5,1) | 0.00552 | 20.71 | 10.77 | 12.44 | 0.671 | 0.866 | 629.2 | 2358 |
| J(0.5,1,1) | 0.00454 | 22.12 | 10.85 | 12.41 | 0.655 | 0.875 | 623.4 | 2329 |
| J(1,0,1) | 0.00618 | 22.43 | 11.08 | 13.07 | 0.647 | 0.848 | 623.7 | 2306 |
| J(1,0.5,1) | 0.00661 | 22.68 | 11.27 | 13.02 | 0.656 | 0.866 | 622.3 | 2307 |
| J(1,1,1) | 0.00569 | 23.46 | 11.62 | 13.25 | 0.659 | 0.877 | 630.5 | 2354 |
| B:J(0,0,1) | 0.00428 | 19.76 | 10.55 | 12.2 | 0.680 | 0.865 | 609 | 2278 |
| U:J(0,0,1) | 0.00428 | 19.76 | 10.55 | 12.2 | 0.680 | 0.865 | 609 | 2278 |
| J:J(0.5,0,1) | 0.00338 | 20.56 | 10.63 | 12.43 | 0.665 | 0.855 | 616.1 | 2291 |

| forced trajectory (FP50 latents) | E‖H‖² | E⟨H,v⟩ | E‖v‖² | cos | anchor raw | between raw |
|---|---|---|---|---|---|---|
| g=(0,0) | 19.18 | 10.47 | 12.18 | 0.685 | 632 | 2317 |
| g=(0,0.5) | 20.52 | 10.61 | 12.18 | 0.671 | 632 | 2323 |
| g=(0,1) | 22.34 | 10.63 | 12.18 | 0.644 | 632 | 2336 |
| g=(0.5,0) | 20.15 | 10.42 | 12.34 | 0.661 | 629.5 | 2317 |
| g=(0.5,0.5) | 20.11 | 10.6 | 12.26 | 0.675 | 627.7 | 2321 |
| g=(0.5,1) | 22.04 | 10.75 | 12.3 | 0.653 | 631.5 | 2341 |
| g=(1,0) | 22.24 | 10.64 | 12.72 | 0.632 | 632.3 | 2317 |
| g=(1,0.5) | 22.16 | 10.96 | 12.73 | 0.653 | 631.3 | 2325 |
| g=(1,1) | 22.68 | 11.21 | 12.83 | 0.657 | 636.9 | 2347 |

## Answers (written 2026-10-31 after the queue finished)

**Does joint gain calibration beat optimized bridge-only?** Not under these conditions.
- On the 32-image fit set, the J family selected (g_anchor, g_between, λ) = (0.5, 0, 1). Its fit error was 0.00310, against 0.00446 for the selected
  bridge-only B = (0, 0, 1), a ratio of 0.70.
- On fresh validation (120100–120131), J/B was **1.262 [0.830, 2.183]**: J better on 12/32 images, ΔLPIPS +0.0034. The pre-registered gate
  (≤ 0.95 × B, LPIPS ≤ +0.005) **failed**.
- The fit-set advantage did not transfer; the 27-point endpoint search over 32 images over-fit to that set.
- No runner-up was chosen, and the 256-image confirmation cohort 121000–121255 was **not opened**. This is a scoped failure of this restricted
  family, not a general impossibility claim.

**Does phase specificity matter?** Not established.
- The shared-gain family U also chose feedback off (U = B = (0, 0, 1)), so J/U is the same contrast as J/B.
- The primary test failed, so J/U was not interpreted.

**Bridge strength:** λ = 1 was best for every gain pair on the fit surface, so bridge recalibration found nothing.
- The frozen-window least-squares λ was 0.85–0.88 for all gain pairs. This was diagnostic only and not used for selection; it shows the local-window
  optimum and the endpoint optimum differ.

**Fixed configurations on validation** (exploratory):
- N = (1, 1, 1), the old ACB stack: N/B0 = 1.508 [1.202, 2.135], harmful again.
- H = (0, 1, 1), feedback between anchors only: H/B0 = 1.931 [1.007, 3.880].
- J/N = 0.837 [0.549, 1.180]. Improvement over N would not establish useful feedback in any case.

**Diagnostics (8 fit images):**
- *Live trajectories:* E‖H_g‖² grows with feedback gain, from 19.8 for gains (0, 0) to 23.5 for (1, 1). The cross term ⟨H, v⟩ grows less (10.6 → 11.6).
  Feedback therefore makes the within-interval held error larger, and the bridge can explain only part of the extra.
- *Common forced trajectory* (cheap and FP evaluated on the FP50 latents, each gain pair with its own state history):
  - Anchor-call raw error is essentially unchanged by g_anchor (628–637).
  - Between-anchor raw error rises slightly with g_between (2317 → 2347).
  - cos(H, v) falls from 0.685 at gains (0, 0) to 0.63–0.66 with feedback.
- The live endpoint gain of (0.5, 0, 1) on these 8 fit images (0.00338 vs 0.00428) is not matched by a clear local-error mechanism. That is
  consistent with a sample-specific fit effect. The alternating-error toy was not used to explain live endpoints.

**Scope:** DiT-XL/2 256, DDIM-50, CFG 1.5, unrefitted W4 + static A8, K4 FP anchors, batch 32, a 27-point grid, 32 fit and 32 validation images.
Spend: 1.158 of 3.0 A30-h.

