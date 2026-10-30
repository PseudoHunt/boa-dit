# Retrospective anchor-bridge correction (br1) — pre-registration

Written 2026-10-30, **before any computation of this pilot**. Branch `anchor-bridge-pilot` (worktree /home/boa-dit-br) from anchor-drift-pilot 5177a8f.
Code `boa_dit/anchor_bridge/run.py`, report `scripts/make_anchor_bridge.py`, results `results/anchor_bridge/`, latents `ckpt/anchor_bridge/` (not in Git).
Stage tag **br1**. Specification: the user's two proposals of 2026-10-30 (the sampler-level "bridge" idea and its critique). The critique's corrected design
governs: the anchor jump measures prediction-error drift, not state error; coefficient-only propagation omits the network response; reusing F at a moved
state is first order. Runs on the same GPU **after** ad1's training jobs, never concurrently. Amendments below the marker, dated.

## 1. Question
In aa1's C sampler (static A8 cheap model, FP anchors at i = 0, 4, …, 48, reuse of d_a = Q(x_a) − F(x_a) on guided channels 0–3), can consecutive anchor
measurements estimate the state deviation accumulated during a hold, and does subtracting that estimate at the next anchor improve the endpoint?
Estimator name: **accumulated hold error estimated from consecutive anchor measurements** (linear bridge between d_old and d_new). The bridge is the
conditional mean of a path only under an equal-increment Gaussian random-walk model; that is not assumed to hold here. Activation feedback is off throughout.

## 2. Frozen setup
Cheap model: unrefitted common smoothed rank-8 W4 base + original carry + static single8 tables (`dit_joint.student(dev, None, "static")`, unchanged);
teacher FP `load_dit`; DDIM-50, CFG 1.5 (`weight_anchor.run.guided`), fp16 autocast, fp32 sampler (guided_diffusion `ddim_sample`), as aa1.
DDIM step i (respaced t = 49 − i): x_{i+1} = A_i x_i + b_i ε_i with A_i = √(ᾱ_prev/ᾱ), b_i = √(1 − ᾱ_prev) − √(ᾱ_prev(1 − ᾱ)/ᾱ) from the spaced
diffusion (checked numerically against `ddim_sample`, rel. L2 ≤ 1e-5).

Bridge at anchor a + K (K = 4; a = 0, 4, …, 44; 12 corrections), with d_new = Q(x_{a+K}) − F(x_{a+K}) measured on the uncorrected state:
  Δx̂ = (d_new − d_old) · Σ_{j=1}^{K−1} (j/K) · b_{a+j} · Π_{m=j+1}^{K−1} A_{a+m}
  x_{a+K} ← x_{a+K} − λ Δx̂, then the anchor DDIM step uses the stored F(x_{a+K}) (zero extra calls); d_new is held for the next interval.
The final interval (48 → 49) has no later measurement and is not corrected.

## 3. Arms
| ID | anchors | correction | cheap / FP calls |
|---|---|---|---|
| C | 0, 4, …, 48 | none (aa1 C) | 50 / 13 |
| C_last49 | 0, 4, …, 44, **49** (13) | none | 50 / 13 |
| CB1 | 0, 4, …, 48 | bridge, λ = 1 (unscaled) | 50 / 13 |
| CBλ | 0, 4, …, 48 | bridge, λ = λ* (frozen from the replay, §4) | 50 / 13 |
| CB1_re (diagnostic) | 0, 4, …, 48 | bridge λ = 1, then cheap and FP re-evaluated at the corrected state (anchor prediction = F(x_corr), held d = Q(x_corr) − F(x_corr)) | 62 / 25 (counted) |
| FP50 | — | — | 0 / 50 |

## 4. Screens (development seeds 60208–60223, 16 images, batch 16; inspected, exploratory)
**S1 cached-error calculation** (C's own trajectory; diagnostic same-state FP call at every step, never used live): per interval, the coefficient-weighted
hold residual H = Σ_j b_{a+j} Π_{m>j} A_{a+m} (e_{a+j} − e_a) versus its bridge estimate Ĥ (= Δx̂). Reported pooled over intervals and images: cosine,
‖Ĥ‖/‖H‖, residual ‖H − Ĥ‖²/‖H‖², fraction of Σ‖H‖² removed; per interval. Forward-extrapolation indicator: correlation of (d_{a+K} − d_a) with
(d_a − d_{a−K}) (reported only; no extrapolation arm).

**S2 short FP window replay**: the state immediately after each anchor update, x_{a+1} on C's trajectory, is replayed with FP predictions for steps
a+1 … a+K−1 (diagnostic FP calls), giving x^F_{a+K}; δx_window = x^C_{a+K} − x^F_{a+K}. Compared with Δx̂ and with H (coefficient-only true residual):
cosine, magnitude ratio ‖Δx̂‖/‖δx‖, residual after subtraction ‖δx − Δx̂‖²/‖δx‖², least-squares scale λ_ls = ⟨δx, Δx̂⟩/‖Δx̂‖² (pooled, 2,000-draw
image bootstrap CI). **λ* = clip(⌊10 λ_ls⌋/10, 0, 1)** (one conservative global multiplier; rounded down; never above the formula's own value 1).
If λ_ls's CI lower bound ≤ 0 (no usable direction), CBλ is dropped and only CB1 runs (reported as "direction not established").

**S3 small live run** (same 16 seeds): C, C_last49, CB1, CBλ, CB1_re, FP50; metrics and paired bootstrap as below.
**Confirmation gate:** confirmation runs iff E(CB1) or E(CBλ) ≤ 0.95 · min(E(C), E(C_last49)) on development (mean endpoint latent rel. MSE). Otherwise stop
(screen result reported).

## 5. Confirmation (fresh 98100–98163; audited 2026-10-30: absent from the registry and every branch/worktree *.py/*.md/*.csv/*.txt/*.yaml)
Two batches of 32, opened once: C, C_last49, CB1, CBλ (if kept), CB1_re, FP50. Endpoint latent rel. MSE vs FP50, PSNR, LPIPS-VGG; paired image bootstrap
10,000 draws, numpy seed 0, ratio of mean errors.
- **Primary:** best pre-specified bridge arm vs C_last49 — CBλ if kept, else CB1. Useful: ratio ≤ 0.95, CI upper < 1, ΔLPIPS CI not entirely above 0;
  smaller positive; harmful (CI lower > 1); else inconclusive.
- Secondary: CB1 / C, CBλ / C, C_last49 / C, CB1 / CB1_re (zero-call approximation vs re-evaluation), CBλ / CB1.
- Activation feedback is added only in a later, separately registered step if the bridge helps; not in this pilot.

## 6. Budget
Cap **0.5 A30-h**, ledger tag `br1` (this worktree's `results/gpu_hours.csv`). Estimates: checks 0.03, S1+S2 0.03, S3 0.06, confirmation 0.25.
Confirmation reserve 0.30 h held by earlier guards.

## Checks (stage `checks`)
- B1 coefficients: A_i x + b_i ε equals guided_diffusion's ddim_sample step (rel. L2 ≤ 1e-5) at steps 0, 25, 49.
- B2 the bridge sampler with λ = 0 reproduces aa1 C on 60200–60215 (batch 16) **bitwise** (`ckpt/activation_anchor/checks_dev_latents.pt` "C"); counts 50 / 13.
- B3 C_last49: anchor set {0, …, 44, 49}, 13 FP calls; CB1_re counts 62 / 25.
- B4 synthetic scalar: the protocol's linear-drift example (errors 0.10 … 0.14, b = −0.1, A = 1) is corrected exactly; random-walk example leaves +0.002.

## Stages (`python -m boa_dit.anchor_bridge.run <stage>`)
`checks` → `screen` (S1 + S2, writes λ*) → `dev` (S3, gate) → freeze (commit `frozen.json`) → `conf` → report. Commit/push after every stage.

---

## Amendments and results
(appended below, dated)
- 2026-10-30, checks (0.028 h): B1–B4 pass (coefficients ≤ 1.4e-7; aa1 C reproduced bitwise; λ = 0 = C; counts 50/13, 50/13, 62/25).
