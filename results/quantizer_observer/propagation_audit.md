# Propagation audit (Experiment D; no GPU), 2026-10-30

Question: do existing measurements contain **signed, same-actuator endpoint response operators** G_t, which a scalar feedback schedule would need?
Inventory: all worktrees and branches, plus `ckpt/`. Keys were checked with `torch.load`/`json`.

| measurement | where | actuator / bed | stored data | signed response vectors? |
|---|---|---|---|---|
| Stage-1 sign-flip replay of the ε error | `boa_dit/temporal/stage1.py`, `results/temporal/stage1/stage1.json`, `ckpt/temporal/stage1_delta_heig8.pt` (temporal-oracles) | heig-8 W4A16's recorded guided-ε error added to the FP DDIM-50 sampler (prediction level) | per-seed endpoint error **norms** (`final_err`, `ratios_to_i`). The cache holds the signed *input* error `delta` [32, 50, 4, 32, 32], not responses | **no** |
| Weight-level sign alternation W± | `results/temporal/weights_joint/` (weights-joint) | heig-8 W4 + carry weights | scalar endpoint ratios (alternating ≈ 0.545 of static) | no |
| Isotropic ε gain probe g_t² | `boa_dit/probe_gain.py`, `results/stage1/gain*.json` | FP sampler, prediction-level isotropic noise | ratios ‖Δx0‖²/‖perturbation‖² per seed and step | no (norm ratios only) |
| DNS response matrix G | `ckpt/dns_response/G_*.pt` (dns-response-pilot) | band-wise latent impulse at 5 updates, heig-8 W4 + carry A16 | signed **48 × 5** matrix on moment statistics | signed, but projected onto moments, a different actuator, not full tensors |
| Analytic-carry responses | `ckpt/analytic_carry/e{1,2}_local_resp.npz` | carry coefficient perturbations, local one-step | local responses; the endpoint check FAILED (not locally linear at h = 0.05–0.2) | no endpoint operator |
| Observability traces | `ckpt/observability/traces_*.pt` | none (no perturbation) | signed per-step errors and states, heig-8 vs FP | not responses |

Descriptive quantities, as reported:
- **Sign-flip replay:** the as-recorded replay reproduces heig-8's endpoint error (153.4 vs 153.3). Alternating the sign leaves 0.0023 [0.0008, 0.0047] of it at amplitude 1, and the factor depends on amplitude (306× / 443× / 549× at 0.5 / 1 / 2).
- **Fixed-energy test:** constant 34.19, alternating 0.050, independent 2.49.
- **Gain probe:** max/median over steps is 17 (fp32, median-of-ratios estimator). The ratio-of-means estimator is unstable (15–500) near trajectory bifurcations. fp16 has a rounding-chaos floor of about 27 at ρ = 1e-4.

**Conclusion:**
- No stored signed endpoint response operators exist for the layer-input (activation-feedback) actuator, or for any actuator at full tensor resolution.
- With G_t the response from a site error to the endpoint and e_t = r_{t−1} − r_t, the interior contribution involves (G_{t+1} − G_t) r_t. Whether it cancels depends on the aligned operators, not on norm ratios.
- The prediction-level sign-flip result shows near-perfect cancellation of an alternating ε-level error in the FP sampler. It does not show that the layer-input operators are scalar or locally constant.
- **A scalar feedback schedule is not identified from current data.** No GPU is spent on it in this allocation, and no derived schedule or gain sweep is proposed.
