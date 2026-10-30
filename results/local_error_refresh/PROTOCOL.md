# Local weight-error refresh with temporal activation feedback (ler1) — pre-registration

Written 2026-10-30, **before any check, screen or generation of this pilot**. Branch `local-error-refresh-pilot` from activation-anchor-pilot
d1eab46 (aa1 complete). Code `boa_dit/local_refresh/run.py`, report `scripts/make_local_refresh.py`, results `results/local_error_refresh/`, latents
`ckpt/local_error_refresh/` (not in Git). Stage tag **ler1**. The user's handoff "Local weight-error refresh with temporal activation feedback"
(2026-10-30) is the specification; this file freezes the choices it leaves open. Amendments go below the marker, dated, before the affected result.

This is a new hypothesis, not a fix for the inconclusive aa1 result (AC/C = 1.017 [0.789, 1.294]). Claims tested: **H1** local held weight corrections
are useful beyond ordinary quantization; **H2** activation feedback adds value to them. No speedup, novelty or superiority-to-anchoring claim.

## 1. Frozen setup (grounding audit, 2026-10-30)
- Cheap model: unrefitted common smoothed rank-8 W4 base `ckpt/ckpt_deflate_heig_r8_w4_s/model.pt` (sha256 f017af87…7ced), original carry
  `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` = `dit_joint.run.BASE_CARRY` (b4a537b0…edd5, matches `results/dit_joint/base.json`; local copy,
  not committed; regeneration recipe in base.json: `deflate.quantize_dc --mode heig --r 8 --bits 4` on W~ = W diag(s), `python -m boa_dit.dit_joint.run base`),
  smoothing `ckpt/temporal/dit_smooth_s.pt` (7ee15c0c…db26), static single8 A8 tables `ckpt/dit_joint/static_single8.pt` (b73030de…dd67), all 112
  block-linear activation sites, DCT-4 configuration unchanged. Hashes asserted at every stage. No calibration, training, rebuild or range change.
- Hook order on every cheap block linear (from `dit_joint.student`): smoothing pre-hook (fc1/fc2 only, x~ = x/s) → JointQuant pre-hook → dense GEMM
  with the quantized weight Q → carry forward hook (+ q V δᵀ, computed on the consumed q) → **local-correction forward hook (new, registered last)**.
- h_l,t := the JointQuant input (after smoothing, before clipping/quantization/feedback); q_l,t := the JointQuant output (consumed activation, before the
  autocast cast inside the GEMM). Captured by a JointQuant subclass (`LocalQuant`) that calls the parent arithmetic unchanged.
- Reference weight W_l in the same coordinates: FP weight × s (column scale) at fc1/fc2, FP weight at qkv/proj. Bias: the cheap linear's bias, asserted
  equal to the FP bias. M_l h := F.linear(h, Q) + (h V) δᵀ (no bias). Selected W_l are cloned plain tensors (no module, no hook).
- Bed: DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5 (`weight_anchor.run.guided`: ε channels 0–2 guided, channel 3 from the cond half; scored channels 0–3),
  fp16 autocast forward, fp32 sampler, sampler loop as aa1 `sample`. Teacher/scorer: FP model (`load_dit`), separate instance, no hooks.

## 2. Method (frozen)
- Refresh steps: sampler index i with i mod K = 0 (K4: 13 refreshes per site per trajectory; K8: 7; K1 = fresh-correction oracle: 50).
- At a refresh, inside the ordinary cheap forward, sequentially per site: b = F.linear(h, W) − M h computed in **fp32 with autocast disabled**,
  stored **fp16** per row (cond and uncond rows separately). Every step at an active site: out ← (out.float() + b.float()).to(out.dtype), i.e. added after the
  complete cheap linear incl. carry, before any downstream op. Later sites see inputs already affected by earlier corrections.
- No full-reference call in local arms (the FP model object is not passed to the local sampler; FP call counter must be 0).
- Activation feedback state is untouched by the correction. **G** = the exact guarded feedback quantizer forward with the previous state forced to
  zero at every call (state_in dropped before each call; the parent creates zeros, exactly as at t = 0; the computed next state is discarded).
  **A** = normal DCT-4 state. Historical static **Q** is a development-only sanity control. All primary contrasts and the output-anchor comparator use G.
- **MP**: separate cheap instance in which the selected linears' weights are replaced by W and their carry hooks removed (all other sites keep the
  base carry); same activation rule (G for MP, feedback for MPA).
- **C_K**: aa1 output anchoring with reuse on G (cheap every step, FP at i mod K = 0, d = f_G − f_FP on channels 0–3 reused).
- State: b, last-refresh index, refresh counters and feedback state reset per batch; serialized with latent, step, RNG for resume.

## 3. Splits (audited on all 15 remote branches + worktrees, 2026-10-30)
- Selection (E0): 60200–60207 (already inspected; inside the joint-pilot dev block 60200–60231).
- Live development: 60208–60223 (exploratory; same block). Diagnostic subset 60208–60215 (batch 8).
- Offline mean-bias bank (only if the shuffle trigger fires): calibration seeds 60000–60031.
- Bitwise parent-reproduction test only: 61000–61031 (joint-pilot cached latents; no metric).
- **Fresh confirmation: 97100–97163** — absent from SEED_REGISTRY and from every branch's *.py/*.md/*.csv/*.txt/*.yaml (word-boundary search;
  JSON hits are float digits). Reserved now in `handoff/SEED_REGISTRY.md`. 95100–95163 and 96100–96163 are evaluated elsewhere and not used.

## 4. Engineering checks (stage `checks`; mechanics use all 12 pool sites; failures block results)
- C1 parent reproduction: LocalQuant static (Q) and fb (A), with the refresh hooks installed but off, on 61000–61031 batch 32 reproduce
  `ckpt/dit_joint/eval_latents.pt` "A"/"B" **bitwise**. G with zero-state: one forward from a fixed state with A after loading explicit zero state equals G
  **bitwise**; G and A agree bitwise at step 0 of a trajectory.
- C2 capture: at fc1/fc2, h × s equals the raw pre-smoothing input (captured by a test-only prepended pre-hook; rel. L2 ≤ 1e-6, fp32 division round
  trip); q equals the GEMM input seen by the carry hook (bitwise).
- C3 synthetic fp64 identities (CPU): scalar and small-matrix cases with ideal full-state feedback, compressed (projected) state, input clipping, nonzero
  carry, nonzero bias, a deliberately nonorthogonal carry, and a rounded stored correction. Identity y − (W h + bias) = a + v + δb residual ≤ 1e-10 rel.;
  all pairwise cross terms logged; ideal-feedback q − h = r_{t−1} − r_t holds (≤ 1e-10) and is reported as violated (not asserted) for clipped/compressed cases.
- C4 M h: hook-free fp32 helper vs the module computation (lin.forward without hooks, minus bias, plus the installed carry hook) with autocast off:
  rel. L2 ≤ 1e-6. The live fp16-autocast defect of the same computation is reported (no threshold). No dense M is formed except in a labelled debug test.
- C5 K = 1: live corrected output vs fp32 W h + M(q − h) + bias, rel. L2 ≤ **5e-3** per site at steps {0, 25, 49} (fp16 GEMM inputs/outputs and fp16
  storage of b). M = W (reference representation := Q with the same carry) gives b exactly 0 and a trajectory bitwise equal to G.
- C6 scale-0 correction (b multiplied by 0 before the add) reproduces G and A bitwise (K4). MP: selected weights equal W, carry hook absent at selected
  sites and present at all others, and the selected output equals F.linear(q, W, bias) under autocast bitwise.
- C7 quantizer updates: per-site call counter = 50 per trajectory in G/A/L/LA; the measurement helper does not change any counter; refresh counts 13 (K4) /
  7 (K8) per site; no module holds the reference tensors.
- C8 save/resume: an LA4 trajectory stopped after step 30, state saved with torch.save and resumed in a fresh call, ends bitwise equal.
- C9 FP call counter 0 for every local arm; cheap counter 50.
- C10 profile (diagnostics off, batch 16, dev seeds, no metrics): G, A, L4 (12 sites), K1 oracle (12 sites), FP50 wall times → budget reforecast.

## 5. Experiment sequence
**E0 site screen.** No existing ranking qualifies (audit: `engine/sensitivity` is block-level on the unsmoothed `deflate_heig_r8_w4` base; quantizer-observer
site accounting is activation error). Fixed pool: blocks 3, 13, 24 × {attn.qkv, attn.proj, mlp.fc1, mlp.fc2}. On 60200–60207 (batch 8), G's own trajectory,
states before steps {5, 25, 45}: FP prediction, G prediction, and G with a fresh (K = 1) correction at one candidate site, all from the same latent;
score_s = Σ‖f_G − f_FP‖² − Σ‖f_G+s − f_FP‖² (channels 0–3, pooled over images and steps). Select the three highest with at most one per block; ties by pool
order (block, then qkv, proj, fc1, fc2). All 12 scores reported, negatives included. "Top three within this pool."

**E1 persistence and headroom** (frozen sites). Traces (8 images 60208–60215, batch 8) of eW_t = (M − W) h_t at the selected sites without applying
corrections, along G and along A: for lags 1/4/8, Σ‖eW_t − eW_{t−L}‖²/Σ‖eW_t‖² (image-bootstrap CI), E_t/E_{t−L}, cross inner products, uncentered
correlation; periodic-refresh residual Σ_t‖eW_t − eW_{a(t)}‖²/Σ_t‖eW_t‖² for K4 and K8 using actual ages (a(t) = t − t mod K). Live (16 images 60208–60223,
batch 16): G, A, Q (sanity), FP50, K1 oracles at each selected site alone (no feedback), all three (no feedback: O3; feedback: OA3).
**Headroom rule:** continue iff E(O3)/E(G) ≤ 0.95 or E(OA3)/E(A) ≤ 0.95 (mean endpoint latent rel. MSE). Otherwise stop the joint pilot (single-site
signals retained and reported). Not a statistical test.

**E2 held corrections** (same 16 seeds, batch 16; G, A, Q, FP50 reused from E1 — same seeds/batch/code/precision): L4, L8, LA4, LA8, MP, MPA, C4, C8.
**K rule:** K* = 8 if E(LA8) ≤ 1.05 E(LA4) and LPIPS(LA8) ≤ LPIPS(LA4) + 0.005, else 4. **Stop rule:** if neither LA4 nor LA8 has E ≤ 0.95 E(G), stop
before confirmation. No damping search.
**Shuffle control** (LA_K*_shuf): each sample's b refreshed normally from its own current input, then applied to another image through the fixed cyclic
derangement j ← (j + 1) mod 16 (cond rows use cond corrections, uncond rows uncond), label repeats recorded. If (E(A) − E(shuf)) ≥ 0.8 (E(A) − E(LA_K*))
with E(A) − E(LA_K*) > 0, run the offline mean-bias control (means of b per site/step/CFG role on 60000–60031, frozen, applied on dev); otherwise not run.
Freeze K*, sites, precision/cast policy and full fp16 correction format in `frozen.json` (committed) before confirmation.

**Diagnostic accounting** (8 images 60208–60215, L_K* and LA_K*): per selected site, a = (M − W)(h_t − h_a), v = M(q_t − h_t), δb = b_stored − b_exact,
energies, pairwise cross terms, algebraic identity residual (fp32) and live numerical defect; guarded-feedback components with the quantizer-observer
accounting (arithmetic copied from quantizer-observer-pilot `InstrQuant`, energies only): p, −p_next, input clip c0, state guard g, rounding ρ_u, second
clip c1, discarded modes −(I−P)r, residual clamp −a, storage η, identity residual, and the input-clip / guard / second-clip / spare-code rates.
The old second-clip-only counter is never reported as "clipping". Layer identities are not endpoint predictors.

**E3 confirmation** (97100–97163, two batches of 32, opened once): G, A, L, LA, MP, MPA, C (all at K*), FP50.
Metrics: per-image endpoint latent rel. MSE vs FP50, PSNR, LPIPS-VGG (`weight_anchor.controls.metrics`). Paired image bootstrap 10,000 draws, numpy seed
0 (`weight_anchor.shared.contrast`), ratio of mean errors, mean/median paired differences, wins.
- **Primary integration contrast LA/L.** Useful integration signal: ratio ≤ 0.95, CI upper < 1, ΔLPIPS (LA − L) CI not entirely above 0. Smaller
  positive: CI upper < 1 otherwise. Harmful: CI lower > 1. Otherwise inconclusive (never equivalence).
- Local correction useful (H1): the same rule on L/G (and reported for LA/A).
- Secondary: L/G, LA/A, LA/G, L/MP, LA/MPA, LA/C; interaction I = [E(L) − E(LA)] − [E(G) − E(A)] with CI. Separate statements, no omnibus pass.
- 64 images may be inconclusive; no sample additions. No FID/ImageReward.

**Costs** (stage `cost`, after confirmation, if budget remains): artifact/unique-weight bytes (analytic for packed formats, measured for this fake-quant
simulation), correction and feedback-state bytes per image incl. CFG, peak memory per arm with only that arm's resident tensors (teacher released except
for C, where the FP model is part of the method), warm CUDA-synchronized full-trajectory timing, 1 warm-up + 5 timed repeats, batch 8 on 60208–60215,
diagnostics off, no concurrent jobs; loading, denoising and decode separated. If the forecast exceeds the remaining cap, batch 4; if still not, costs
are reported as unmeasured. Simulation timings are not packed-kernel evidence.

## 6. Budget
Cap **2.0 A30-h**, ledger rows tagged `ler1` in `results/gpu_hours.csv` (checks, profiling, screen, traces, dev, diag, conf, cost, failures).
Planning ceilings: setup/checks/profile 0.30, screen + traces 0.40, development 0.55, confirmation 0.55, reserve 0.20. Every pre-confirmation stage
guard keeps 0.55 h for confirmation; runtime stop before each confirmation batch. Cut order if needed: mean-bias control, cost repeats, diagnostics.
Stop with partial results at the cap.

## Stages (`python -m boa_dit.local_refresh.run <stage>`)
`checks` → `screen` → `e1` → `e2` → `diag` → freeze (commit `frozen.json`) → `conf` → `cost` → report. Commit/push after every stage.

---

## Amendments and results
(appended below, dated)
- 2026-10-30, before any check (implementation note): all check tolerances are relative L2 norms (square root of the energy ratio). C2 is dtype-aware:
  1e-6 where h is fp32, 1e-3 where h is fp16 under autocast (fc2 inputs are GELU outputs in fp16, so the smoothing pre-hook (x.float()/s).to(fp16) rounds;
  fp16 unit roundoff 4.9e-4). E0 uses the model inputs actually passed by the sampler (respaced timestep), recorded during G's trajectory.
  In the scalar C3 case "compressed state" discards the state entirely (the only nontrivial orthogonal projection of a 1-token state).
- 2026-10-30, checks (0.184 h + 0.006 h diagnosis): all checks pass except C4 as written (fp32 helper vs module output minus bias: max rel. L2 1.24e-6 at fc2,
  tol 1e-6). Diagnosis (`checks_c4_diagnosis.json`, `scripts/ler1_c4_diagnosis.py`): the bias-free module GEMM + carry hook equals the helper **bitwise** at all
  12 sites; the residual is cuBLAS's fused-bias addmm kernel vs mm, unrelated to b (both terms of b use the bias-free path). C4 criterion amended to the
  bias-free comparison (bitwise); original result kept as `C4_pass_original: false`. Other results: C1 Q/A bitwise vs joint-pilot latents, C5 max 5.2e-4
  (tol 5e-3), C6 zero correction bitwise, MP structure/numerics bitwise, C7/C9 counts, C8 resume bitwise. Live fp16 M h defect 3.3–4.8e-4 (reported).
  Profile (16 images, 12 sites): FP50 9.9 s, G 43.3, A 43.3, L4 46.2, O3 52.7, C4 45.8, MP 43.1 → reforecast: screen 0.02 h, E1 ≈ 0.12 h, E2 ≈ 0.13 h,
  diag ≈ 0.05 h, conf ≈ 0.30 h, cost ≈ 0.15 h.
- 2026-10-30, E0 screen: all 12 scores positive, max +1.16% of G's prediction SSE; selected b13.attn.qkv, b24.mlp.fc1, b3.attn.qkv (frozen).
