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
