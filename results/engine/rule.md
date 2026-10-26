# Engine round 2 — where the remaining error lives, the carry's rank curve, ideas A′/B/C on heig-8 (pre-registration)

Written 2026-10-26, before any arm of this handoff ran. Section 0 was done first: the H4dsf FID-5k was stopped, Gate 1
was committed and pushed (`d7246ac`), and the heartbeat was restarted. Results go to `results/engine/REPORT.md`,
generated from the JSONs; deviations are recorded there as they happen. Nothing above the marker changes afterwards.

## Base and conventions
- **Base:** heig-8 = `deflate_heig_r8_w4`.
  - Rel. latent MSE 0.0680 (seeds 0–511); FID-5k 16.21; extra bits 0.148/weight (4.148 bits).
  - Recipe: adaLN W8 RTN; sequential quantized-prefix calibration on seeds 1000–1031 × 2 branches × 50 steps; uniform
    buckets; cfg branch weights; per-channel asymmetric; Hessian grid search; replace 1/256.
  - The solver gets P·H·P, with V = the top-8 eigenvectors of H. The carry δ = (W − W_q)V is attached right after each
    layer is quantized.
- **Every arm reports:**
  - latent MSE on seeds 0–511, with a paired bootstrap CI vs heig-8 (2,000 resamples, seed 0);
  - the per-block held-out error table (block-l output error on the arm's own prefix, seeds 1032–1047; Gate 1
    `downstream.py --all_blocks` protocol, all four linears as in the arm);
  - extra bits per weight;
  - solver seconds per layer type.
- **"Beats by ≥ x% (CI excluding 0)":** point-estimate reduction ≥ x **and** the 95% CI's lower bound > 0.
- **Discipline:** one arm per GPU, end to end. Checkpoint loads go through CPU.

## §1 Attribution on heig-8
- **Block arms:** for each block l, restore block l's four linears to FP (and remove its carries) in the heig-8 model.
  Measure rel. latent MSE on seeds 0–63.
  - share_l = 1 − relMSE(arm_l)/relMSE(heig-8, seeds 0–63), with a paired CI.
  - The base is the stored heig-8 per-image values for seeds 0–63, checked by re-running heig-8 on those seeds (it
    must match exactly).
- **Layer arms:** in the 3 blocks with the largest share, each of the four linears is restored alone (12 arms, same
  protocol).
- **Concentration:** top3 = the sum of the three largest share_l, as a fraction of heig-8's error. Also reported: the
  normalized top-3 fraction top3 / Σ_l max(share_l, 0), and the Gini coefficient of max(share_l, 0).

## §2 Network-sensitivity diagnostic (gates §5)
- **Where:** blocks l ∈ {2, 10, 20, 27}; buckets {0, 4, 8}, each at the bucket's middle step (steps 2, 22, 42).
- **Rows:** held-out seeds 1032–1047 × 2 branches = 32 rows (the handoff's "32 images"). Inputs are captured along the
  FP trajectory, exactly as calibration does.
- **Linearization point:** the output h_l of the heig-8 model's blocks 0..l, with carries.
- **Suffix:** FP blocks l+1..27 + final layer + unpatchify, giving the raw 8×32×32 output of each row. fp32, TF32 off.
- **Probing:** 64 probes r ~ N(0, I) over each row's output, drawn from a fixed generator (seed = 1000·l + step).
  - Each probe is one backward pass, giving g_t = J_tᵀ r per token.
  - F_l = Σ_probes Σ_t g_t g_tᵀ / (64 · n_tokens), with tokens pooled plainly over rows and both branches.
- **Error:** e_t = heig-8 block-l output (with carries) − FP block-l output (W8 adaLN), on the same inputs and tokens.
- **Reported per (block, bucket) and for the bucket mean F̄_l:**
  - λ_max/λ_median of F;
  - the fraction of Σ_t ‖e_t‖² in F's top-32 and bottom-32 eigendirections;
  - ρ = Σ_t e_tᵀ F e_t / (Σ_t ‖e_t‖² · tr(F)/1152).
- **Gate (read 2a):** §5 runs iff ρ ≥ 1 **and** λ_max/λ_median ≥ 10, both for F̄_l, in at least two of the four
  blocks. If ρ < 1 in all four (bucket mean), the report records that GPTQ-plus-carry already leaves its error in
  low-sensitivity directions, and §5 is skipped.

## §3 Rank curve and allocation for the carry
- **Arms:** heig-16 and heig-32 at W4, heig-32 at W3. These use `quantize_dc.py --mode heig --r 16|32`, unchanged.
- **Matched-bit baselines:** static-16 W4 and static-32 at W4 and W3 are reused from the stored Gate A per-image values
  (§0 of Gate 1 showed they reproduce).
- **heig-alloc** at W4:
  - For each layer: E = W − W_q(uniform W4). (λ_k, v_k) are the eigenpairs of the layer's bucket-mean FP Gram (Gate A
    `fp_grams`), sorted by λ_k descending.
  - r90_ℓ = the smallest r such that Σ_{k≤r} λ_k‖E v_k‖² ≥ 0.9 · Σ_k λ_k‖E v_k‖².
  - r_ℓ = min(r90_ℓ, c), floor 0. The cap c is the integer that makes the model-wide extra bits closest to 0.148, and
    it must fall within ± 0.01. If no integer cap does, the closest is used and the report says so.
  - The r_ℓ distribution is reported.
  - Run via `quantize_dc.py --mode heig` with a per-layer rank file (a new `--r_config` option; otherwise unchanged).
- **Output:** the bits-vs-latent-MSE curve for heig and static at 0.15, 0.30 and 0.59 extra bits.

## §4 Idea B — error-ordered asymmetric calibration (one-sided + carry, heig-8 otherwise)
**(i) True-sequential Grams within a block, in the arm's order.**
- A layer's input Gram is collected after every layer before it in the order is quantized, with carries active.
- q, k and v are rows of the fused qkv, solved separately with the split helper. They share one input, so one pass
  serves all three.
- **Forward order:** Q, K, V, O, F1, F2 — 4 Gram passes per block (qkv, proj, fc1, fc2).
- **Error order:** V, O, F1, F2, Q, K — the same 4 passes. Q and K reuse the qkv pass. The downstream layers are
  calibrated with Q and K still at FP.
- **Carries for slices:** a carry for a partly quantized qkv has zero rows for the slices not yet quantized.

**(ii) Paired pass.**
- The FP model runs alongside, on the FP-propagated block inputs for the same seed, step, branch and token. Its layer
  inputs X are captured with all of its layers FP.
- H = E[x̃x̃ᵀ] and C = E[x x̃ᵀ], with the same bucket and cfg weighting as H.

**(iii) Asymmetric target.**
- W̃ = W·C·(H + 0.01·mean diag(H)·I)⁻¹.
- W̃ is quantized against P·H·P, with V = the top-8 eigenvectors of H (true-sequential).
- The carry is δ = (W̃ − W_q)V. The symmetric arms use W̃ = W.

**Arms (W4):** B1 forward + symmetric; B2 error order + symmetric; B3 forward + asymmetric; B4 error order + asymmetric.
B1 is the true-sequential control for heig-8, and its gap to heig-8 is reported. The handoff's prediction to test is
B4 > B3 > B1 ≈ B2.

## §5 Idea A′ (runs only if read 2a passes)
As the handoff specifies: heig-8 with F_l as the output-side factor.
- **Factors:** O and F2 become two-sided with H_row = G·F_l·G. V goes through W_o and then F_l. F1 goes through
  GELU′·W2 and then F_l.
- **Settings:** row-ordered, no act_order, deflated input Grams, carries recomputed; downstream-first for V and F1.
- **Arms:** A′-sym; and A′-asym (B4's order and target) only if read 4 holds.
- **Caveat:** F_l is built on the FP suffix.
- **Cost-driven choices** (all 28 blocks × 10 buckets × 64 probes would far exceed the cap) are pre-registered here:
  - 3 probe steps per block (the middle steps of buckets 0, 4 and 8), with buckets mapped to the nearest probed step;
  - 32 probes;
  - the 16 held-out seeds' rows replaced by 8 calibration seeds (1000–1007) × 2 branches.
  - Any further deviation is recorded before the arm runs.

## §6 Idea C — CD/CASR polish (runs only if the best of B/A′ beats heig-8 by ≥ 5%)
- **Coordinate descent:** as the handoff specifies (ΔJ per coordinate, rank-one updates of G, carries recomputed).
- **Gate and anneal:** taken from "Sanjay's CASR code" if it can be found on this machine or in the user's GitHub. If
  not: τ_s = τ₀·γ^s with τ₀ = 1% of the row's J, γ = 0.7, 30 sweeps, and the report says so.
- **Arms:** C1 CASR; C2 plain CD (control); C3 = C1 with sequential acceptance within each head.

## §7 Oracle fix
- **Changes from Gate 1:**
  - **Regularizer strength:** Gate 1's λ·mean(reg) never forced binarization, so the hard rounding discarded what was
    learned. The regularizer is now applied with weights λ ∈ {1, 10} on the mean over parameters, against the
    nearest-rounding-normalized reconstruction loss.
  - **Logging:** the soft and hard validation losses, and the fraction of h within 0.01 of {0, 1}, are logged.
  - **Init:** the init is checked to reproduce nearest rounding when hard-rounded.
  - **Carry:** the carry is recomputed from the final hard rounding, as before.
- **Validity check:** the chosen configuration's hard validation error must be ≤ the nearest-rounding validation error
  on the same grid. If no configuration passes, the oracle is reported as **invalid**, not as a number.
- **Run:** block 2, on heig-8's prefix inputs; all four linears; true block loss; 500 steps. Reported vs heig-8 and vs
  the best engine arm, on held-out seeds 1032–1047.

## §8 FID-5k
- **Arm:** the single best arm by latent MSE among the equal-bit arms of §3–§6 (heig-alloc, B1–B4, A′, C; extra bits
  0.148 ± 0.01).
  - *Why not heig-16/32:* they are more bits, and read 6's engine claim is an equal-bit claim.
  - *If none beats heig-8:* the FID-5k of that best arm still runs only if its latent MSE is lower than heig-8's.
    Otherwise it is skipped as uninformative and the report says so.
- **Protocol:** seeds 0–4999 with label accuracy, against the stored FP 12.19 / heig-8 16.21 / uniform 18.14.

## Reads (handoff §9)
1. **Remaining error is concentrated** iff top3 ≥ 0.40 (§1, absolute shares of heig-8's error).
2. **A′ has leverage** iff read 2a passes. **A′ earns its place** iff an A′ arm beats heig-8 by ≥ 5% (CI excluding 0).
3. **"Deflate-first" holds at scale** iff heig beats static at matched bits at every rank tested (r = 16 and 32 at W4;
   32 at W3), each with the CI excluding 0. **Allocation is adopted** iff heig-alloc beats heig-8 (CI excluding 0) at
   equal bits.
4. **B earns its place** iff B4 beats B1 by ≥ 5% (CI excluding 0). **The ordering stands** iff B4 beats B3 by ≥ 2%
   (point estimate) **and** |relMSE(B2)/relMSE(B1) − 1| ≤ 1%.
5. **C earns its place** iff C1 beats its base by ≥ 3% (CI excluding 0) **and** relMSE(C1) < relMSE(C2).
6. **The engine section exists** iff at least one of A′, B, C earns its place. The paper's core is the carry (read 3)
   regardless. An engine claim also needs the FID-5k of the best equal-bit engine arm to beat heig-8's 16.21 by ≥ 0.5.

## Order and cap
- **Order:** §1 (GPU 0) ∥ §2 (GPU 1) → §3 (GPU 0) ∥ §4 (GPU 1) → §5 if allowed → §6 if allowed → §7 → §8.
- **Cap:** 8 A30-hours, from `gpu_hours.csv` rows starting after this file is committed.
- **If the cap is reached, cut in this order:** C3, A′-asym, the W3 rank point, the oracle.

---

## Results
See `results/engine/REPORT.md`.
