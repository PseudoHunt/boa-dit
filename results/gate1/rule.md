# Gate 1 — deflate-and-carry ablations, engine arms, two-sided variants (pre-registration)

Written 2026-10-26, before any arm of this handoff ran. The only jobs that had started were the rebuild of lost
artifacts (§0); none of those is an arm. Results go to `results/gate1/REPORT.md` as they land, and are generated from
the JSONs. Nothing above the marker in this file changes afterwards.

## §0 Rebuild (fresh instance; `ckpt/` was lost)
The DiT-XL/2 weights, the uniform W4 / W3 checkpoints, the FP eval latents, the FID reference stats and Gate A's FP
Grams are not in git. They are rebuilt with the unchanged code: `quantize.py --weighting uniform --K 1 --lam 1
--branch cfg` at W4 and W3, `fp_grams.py`, `eval_fid.py --ref`, and `lowrank.build(..., ["static"])` at W4 and W3.
- **Integrity check:** the rebuilt uniform W4 model is evaluated on seeds 0–511. Its per-image rel. latent MSE must
  equal the stored values (`results/stage2/eval/latent_uniform_w4_K1_lam1_brcfg.json`) exactly. This also checks the
  rebuilt FP latents.
- **If it is exact:** the stored per-image JSONs are reused for pairing. These are uniform W4 / W3, static-8 and
  static-32 at W4 / W3, and the premise-check arms (V rows 25.0%, Q,K rows 6.5%).
- **If it is not exact:** every reused baseline is re-evaluated on this host, and that is recorded.
- **Storage:** checkpoints live on the instance's root overlay (`ckpt -> /root/boa_ckpt`). The 7.7 GB left on
  `/home/jl_fs` cannot hold them.

## Conventions
- **Pipeline:** as the handoff states. adaLN W8 RTN; sequential quantized-prefix calibration on seeds 1000–1031 × 2
  branches × 50 steps; B = 10 buckets of 5 steps, p_b = 1/B; branch weights cfg (s² : (s−1)², s = 1.5); per-channel
  asymmetric; Hessian grid search; replace 1/256; no dead-row rule; one GPU per arm, end to end; all checkpoint
  loads through CPU.
- **Paired bootstrap:** 2,000 resamples, seed 0, over the evaluated images (`scripts/make_gateA.paired`). Reduction
  = 1 − relMSE(arm)/relMSE(ref), positive = better.
- **"Improves by ≥ x% (CI excluding 0)":** the point-estimate reduction is ≥ x **and** the 95% CI's lower bound is
  > 0. Where the handoff gives a threshold with no CI clause (read 5), only the point estimate is used. The CI is
  still reported.
- **Gate averaging** ("as the existing gate code does"): averages are taken within a timestep bucket over images,
  steps and branches, with the same cfg branch weights as H_col. Where the formula uses the gate **squared**
  (ḡ₂,b² in M_b, Σ_b p_b ḡ₁,b² in the Variant 1 V factor), the bucket-mean **energy** E_b[g²] is used, which is
  what `gates.pt` / `weights.py` average. Where the gate enters **linearly** (A_t, diag(ḡ₁,b) R_b diag(ḡ₁,b)), the
  bucket mean E_b[g] is used. The ratio E_b[g²] / E_b[g]² is logged per block so the choice can be audited.
- **Per-sample quantities:** d_t = GELU′_tanh(h_t) from the fc1 pre-activation of the same forward. L_t uses the
  sample's own γ₂ (scale_mlp) and the token's own z_t. LayerNorm eps is 1e-6; σ_t = √(var(z_t) + eps).
- **Factor pass:** factors are accumulated in the same per-block pass as the Grams, before any layer of the block is
  quantized, with the block's FP linears and W8 adaLN. Hooks are read-only. C_b, the attention-weighted Grams and
  R_b are computed in fp32 with TF32 off. They are combined over branches with cfg weights, as H_col is.
- **Variant 2 tokens:** 520 tokens per block (handoff: 512). Per bucket, 26 cond and 26 uncond tokens: distinct
  images, a uniform step within the bucket and a uniform token position, from a fixed generator (seed = block
  index). R_b combines the two branches with the cfg weights.
- **V input side:** the per-head attention-weighted Gram E[(A_h X)ᵀ(A_h X)]. A_h = softmax(q kᵀ/√72) is rebuilt
  explicitly in fp32 from the block's FP qkv output (the `attn_share` construction). In DC arms it is deflated with
  the qkv layer's V_dc: P·G_h·P.

## Solver
- **Two-sided solves** use a row-batched solver in `boa_dit/twosided/solver.py`. It subclasses the fork's `BoA` and
  replaces only `boa()` with TurboBoA's `n_quant_rows` scheme.
  - *Scheme:* rows are quantized in groups. Inside a group, each row runs the fork's GPTQ column loop with H_col and
    there is no row-to-row compensation. After each group, later rows are compensated with U_rowᵀ[r2:, g] U_gg⁻ᵀ
    (W_g − Q_g).
  - *Why not import TurboBoA:* its module (`nested-babai/turboboa`) also defines a top-level `quantizers` package,
    which collides with the fork's, and it carries unrelated nested-lattice changes. The scheme is the same.
- **Group sizes:** n_quant_rows = 64 for fc1 (4,608 rows) and W_o (1,152 rows), as the handoff says. For V it is 8,
  because head_dim is 72: a 64-row group would leave one 8-row remainder per head.
- **Unit tests** (`boa_dit/twosided/test_solver.py`), which must pass before Section 4:
  - with H_row = I, the solver equals the one-sided path bit-exactly;
  - with n_quant_rows = 1, it equals the fork's `BoA.boa()` to fp32 tolerance;
  - on one real layer per type, the two-sided objective tr(ΔW H_col ΔWᵀ H_row) is reported for n_quant_rows
    1 / 8 / 64 and for one-sided GPTQ.
- **qkv split helper:** q, k and v row blocks are solved separately and written back into the fused weight. It must
  reproduce dc-8 bit-exactly on a full arm when all three slices use the one-sided path. If it is not bit-exact, the
  size of the difference is recorded, and the engine arms are compared against a dc-8 rerun through the split path.
- **act_order:** the fork's `act_order_col=True` on every layer. `act_order_row=True` on layers solved two-sided.
- **fc1-rowgroups fallback:** used only if the row-batched solver does not pass its tests within 2 hours of
  engineering.

## D1 — ceilings by layer type (W4 uniform, seeds 0–127)
- **Arms:** all 28 blocks' fc1 weights restored to FP; the same for all fc2; the same for all attn.proj. Biases are
  never quantized.
- **Base:** the stored uniform W4 per-image values for seeds 0–127, valid if the §0 integrity check is exact.
- **Reused:** V rows 25.0%, Q,K rows 6.5% (premise check, same model and seeds).
- **Share of base error removed** = 1 − relMSE(arm)/relMSE(base), with a paired bootstrap CI.

## D2 — does the MLP path reshape the attention objective?
- **The pass:** one factor pass along the uniform W4 prefix. Each block's inputs come from the uniform W4 model's
  earlier blocks, with W8 adaLN. Factors are computed with the block's FP linears; this is the state the solver
  would see.
- **Blocks:** 0, 2, 10 and 27. For each block:
  - λ_max/λ_min and λ_max/λ_median of R = Σ_b p_b R_b. Per-bucket values are also reported.
  - The fraction of attention-output error energy in R's top-16 and bottom-16 eigendirections, at the same 520
    tokens. e_t = attn(x; W4 qkv, W4 proj) − attn(x; FP). It is reported both pre-gate and gated, ḡ₁ ⊙ e_t; gated is
    the residual-stream space that R acts on, and is primary.
  - The same two spreads for D⁻¹ᐟ²·H_row(fc1)·D⁻¹ᐟ² with D = diag(H_row(fc1)).
- **factors.json:** eigenvalue summaries (min, median, max) of every factor for all 28 blocks go to
  `results/gate1/factors.json`, keyed by pass (`d2_uniform_w4`, then one key per engine arm).

## Section 3 — DC subspace ablations (W4 unless stated; latent MSE on seeds 0–511)
- **heig-8, wsvd-8, dc-32 (W4 and W3), dc-8 (W4):** `quantize_dc.py` unchanged, `--mode heig|wsvd|dc --r 8|32`.
- **static-32:** the stored per-image values at W4 and W3, if §0 is exact.
- **Comparisons:** paired CIs vs uniform and vs dc-8 (heig-8, wsvd-8) or dc-32 (static-32).

## Section 4 — engine arms (DC-8 + extension, W4)
- **Arms:** E1–E6 as in the handoff. E6 = whichever of E4/E5 has the lower latent MSE, plus act_order (column and
  row).
- **Script:** `boa_dit/twosided/quantize_ts.py`. Each arm records the solver wall-clock per layer type, the extra
  bits (carry only), and the drift ratio vs uniform.
- **D2 skip rule:** if D2's λ_max/λ_median is < 2 in all four blocks, E5 is skipped and E6 = E4 + act_order.
- **Read-1 fallback:** if read 1 fails before Section 4 starts, E2–E4 run on the uniform-W4 base instead of dc-8,
  as the handoff directs, and are compared against uniform. E5 and E6 follow onto the same base, so that read 4
  (E5 vs E4) stays like-for-like. E1 stays on dc-8.

## Section 5 — FID
- **FID-5k:** seeds 0–4999, against the ADM VIRTUAL ImageNet-256 batch. Arms: uniform W4, dc-8 W4, static-8 W4, and
  the best of E1–E6 by latent MSE.
- **Label accuracy:** ResNet-50 top-1 vs the conditioning label, for the same four, on all 5,000 images. Seeds 0–511
  are also reported.
- **FID-2k:** seeds 0–1999, for dc-32 and static-32 at W3.

## Reads (as in the handoff §6)
1. **Subspace is a finding** iff dc-8 beats both heig-8 and wsvd-8 by ≥ 5% latent MSE, with CIs excluding 0,
   **and** dc-32 ≥ static-32 at both W4 and W3.
   - "dc-32 ≥ static-32" means relMSE(dc-32) ≤ relMSE(static-32) on the point estimate, seeds 0–511. FID-2k at W3 is
     reported alongside but does not decide.
   - Otherwise DC is recorded as a generic deflate-and-carry variant that edges post-hoc correction at low rank.
2. **W4 story stands** iff FID-5k(dc-8) ≤ FID-5k(uniform) − 1.0.
3. **Variant 1 is live** iff D1's fc1 ceiling is ≥ 10% (point estimate) **and** E3 improves dc-8 by ≥ 5% (CI
   excluding 0). The gated V factor belongs in the method iff E2 improves dc-8 by ≥ 5% (CI excluding 0).
4. **Variant 2 is live** iff D2's λ_max/λ_median is ≥ 3 in at least two of blocks {0, 2, 10, 27} **and** E5 improves
   E4 by ≥ 3% (CI excluding 0). If the spread is < 2 in all four, E5 is skipped. It is then recorded that the
   block-level and sub-block objectives coincide on this model.
5. **act_order** is adopted iff E1 improves dc-8 by ≥ 3% (point estimate).
6. **The engine section exists** iff at least one of reads 3–4 holds. **The paper exists** iff reads 1 and 2 hold and
   at least one of reads 3–4 holds.

## Budget
- **Cap:** 8 A30-hours across both GPUs, counted from `results/gpu_hours.csv`, including the §0 rebuild.
- **Estimate:**

  | item | A30-h |
  | --- | ---: |
  | rebuild | 0.45 |
  | D1 + D2 | 0.25 |
  | Section 3 | 0.75 |
  | Section 4 | 1.9 |
  | Section 5 | 3.6 |
  | **total** | **≈ 6.95** |

- **If the cap binds:** Section 5's FID-2k pair at W3 is cut first, then the E6 arm. Each cut is recorded.

---

## Results
See `results/gate1/REPORT.md` (generated by `scripts/make_gate1_engine.py`).
