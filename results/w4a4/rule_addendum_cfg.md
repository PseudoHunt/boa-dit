# Addendum (2026-10-26, before any CFG arm result): CFG arm, CFG-alloc + control, CFG-amp

All arms use the corrected base's settings (K/V 16-bit, unsigned fc2, range search), are screened on MJHQ-64 paired
PSNR/LPIPS vs the corrected base (`rule_addendum_screen.md`), and are designed on the design set only
(`rule_addendum_design.md`).

- **Layer-level check** (`cfg_layer_ratio.json`, design set):
  - Median per-layer ‖E v‖²/mean branch = 5.0; 66% of layers ≥ 4, holding 58% of the guided error; 5% ≤ 2.
  - So the "most near 1–2, with 14 from a few layers" condition does not hold, and **CFG-partial is not run**.
- **CFG** (`--cfg`): H_cfg = E[v vᵀ], v = 4.5·x_c − 3.5·x_u, rescaled to the per-branch Gram's trace.
  - It replaces the input Gram for L₁ and for the GPTQ metric (through H̃_⊥) only.
  - Smoothing D, Σ_A, the L₂ refit and the scoring keep the per-branch H₀.
- **CFG-alloc** (run regardless of CFG's result):
  - Ranks: r_ℓ = min(32, round(k·‖E v‖²_ℓ)), floor 0, from the layer-level check. k is chosen by bisection so that
    total branch parameters = the r4-everywhere total over the 224 quantized layers, ± 2%.
  - Uses H_cfg as in CFG.
  - The rank distribution is reported by layer type. Expected: attn2.to_q takes most of it, since it holds 46% of the
    guided error.
- **CFG-alloc control:** the same ranks with the per-branch Gram (allocation only, no CFG metric), so that the
  allocation effect and the metric effect separate.
- **CFG-amp:** runs iff CFG's paired mean MJHQ-64 PSNR gain over the base is ≥ +0.2 dB (point estimate).
  - v = x_u + w_eff·(x_c − x_u) with **w_eff = 7.5** (4.5·√(14/5) ≈ 7.53, rounded as given), so that the median
    layer-level ratio matches the network-level 14.
  - One value, no sweep. Everything else as in CFG.
