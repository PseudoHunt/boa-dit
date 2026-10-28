# Fixed-code scale modulation of heig-8 W4 (protocol)

Written 2026-10-28, before any sampling for this experiment. Branch `weights-joint`, results in
`results/temporal/weights_scale/`. Nothing above the marker changes afterwards; deviations are appended below it, dated.

## Budget
- **1.0 A30-hour hard stop** (set by the assistant; the user gave no number and asked not to be asked), stage
  `weights_scale` in `results/gpu_hours.csv`, with a guard counting only that stage. The estimated need is ≈ 0.3 h.
- Jobs are pilot-timed. Unfinished arms are reported as unrun, and confirmation sample counts are not reduced.

## Construction
- **One packed W4 table:** heig-8's integer codes q and zero points z, fixed. Only the reconstruction scale is modulated,
  W_{q,t} = c_{ℓ,t}·s(q − z).
- **Per linear layer ℓ** (the 112 block linears), one multiplier, shared across the layer's rows. It is applied as
  y_t = c_{ℓ,t}·(x W_qᵀ) + b + δ(Vᵀx): the output is scaled before the bias. **heig-8's rank-8 carry δ(Vᵀx) is the
  high-precision low-rank branch and is not scaled.**
  - There is no extra matmul and no weight re-rounding. For c = 1 exactly the layer is untouched (bit-identical to
    heig-8). For c ≠ 1 the scaling is computed in fp32 on the fp16 output and cast back.
  - The W8 adaLN layers are not modulated.
- **Activations** stay at the high-precision reference (fp16 autocast, no activation quantization). The same
  multiplier serves both CFG branches within a denoiser call.

## Step 1: static scale refitting (no code changes)
- **Formula:** per layer, c*_ℓ = ⟨Y_q, Y − C⟩ / ‖Y_q‖²_F, with Y = W x (FP weight, bias-free), Y_q = W_q x (bias-free),
  and C = δ(Vᵀx) (the unscaled carry).
- **Data:** accumulated over all tokens of the calibration inputs: seeds 1000–1003, all 50 steps, both CFG halves,
  along heig-8's own trajectory.
- **Range:** c* is clipped to the pre-registered range [0.90, 1.10], with boundary hits reported.
- **Baselines:**
  - **original heig-8** (c ≡ 1);
  - **static-refit heig-8** (c ≡ c*).
  The temporal method must beat the second.

## Step 2: temporal modulation around the refit
- **Schedule:** c_{ℓ,t} = c*_ℓ·[1 + a_{f(ℓ)}·φ(t)], with four family amplitudes a_f (attn.qkv, attn.proj, mlp.fc1,
  mlp.fc2) and t the denoiser-call index 0..49.
- **Patterns:**
  - constant: φ = 1;
  - slow: φ = −1 for t < 25, +1 for t ≥ 25;
  - alternating: φ = (−1)^t.
- **Endpoint fit,** separately per pattern, using the same procedure each time:
  - calibration seeds **5000–5007**;
  - baseline = static refit; e_i = x0_i(baseline) − x0_i(FP);
  - for each family f, runs at a_f = ±ε with ε = 0.005 and the others at 0, giving R_{i,f} = (x0(+ε) − x0(−ε))/(2ε);
  - solve min_a Σ_i ‖e_i + Σ_f R_{i,f} a_f‖² + λ‖a‖² subject to |a_f| ≤ 0.02 (bounded least squares).
- **FD resolution check:** R at ε and at 2ε (one extra pair per family, alternating pattern) must agree. The reported
  measure is the relative difference ‖R(2ε) − R(ε)‖/‖R(ε)‖; the symmetric curvature ratio
  ‖x0(+ε) + x0(−ε) − 2x0(base)‖ / ‖x0(+ε) − x0(−ε)‖ is also reported.
- **λ selection:** per pattern, from {0, 0.01, 0.1, 1} × mean diag(RᵀR). The fitted amplitudes are evaluated with
  actual nonlinear sampling on validation seeds **5008–5023** (16), and the λ with the lowest validation latent MSE is
  kept.
- **Shuffled control:** the alternating arm's fitted amplitudes, with a fixed balanced sign schedule (25 + and 25 −,
  random permutation, seed 0). Not refit.
- **Freeze:** everything is frozen and written to `frozen.json` before the evaluation seeds are touched.

## Step 3: evaluation (reserved seeds 4100–4227, 128, untouched until now)
- **Arms:**
  1. original heig-8;
  2. static refit;
  3. endpoint-fitted constant;
  4. endpoint-fitted slow;
  5. endpoint-fitted alternating;
  6. shuffled alternating.
- **Reported:**
  - endpoint latent rel. MSE vs FP (fp16 autocast), per seed, with paired bootstrap CIs vs static refit and vs
    endpoint-fitted constant (2,000 resamples, seed 0);
  - per-step prediction-error energy, absolute coherent energy C, I and C/I on the FP trajectory of seeds 4100–4131;
  - coefficient storage (112 static multipliers + 4 amplitudes + the pattern; the shuffled schedule is 50 bits);
  - sampling time per arm (simulation).
- **Reading:**
  - The **slow schedule winning** supports timestep-dependent scale correction (prior art exists: TDQ, NeurIPS 2023;
    no novelty claim for time-dependent scaling itself).
  - **Alternation winning** over both constant and slow, **and losing that advantage when shuffled,** would be evidence
    for temporal error shaping.
  - Symmetric modulation leaves the fixed-input average output equal to the baseline's, so any gain must come from the
    perturbation interacting with changing inputs, nonlinear layers and the sampler.

## Validation (before the fit)
- c ≡ 1 reproduces heig-8 latents bit for bit.
- Output scaling equals weight scaling (c·W_q) on a layer in fp32, within rounding.
- At a fixed input, the average of the (1 + a) and (1 − a) outputs equals the c = 1 output.

---

## Deviations and results
(appended below as they happen)
- 2026-10-28, refit: c* ≈ 1 everywhere (family means 1.0001–1.0003; no boundary hits). heig-8's scales are already
  locally optimal, so the static-refit baseline is essentially the original.
- 2026-10-28, **fit: the FD resolution check fails** (recorded before any evaluation result):
  - alternating pattern: R(2ε) vs R(ε) relative difference 0.53–0.73; symmetric curvature ratio 0.45–3.1. Constant and
    slow patterns: curvature ≈ 0.33–0.48.
  - At ε = 0.005 the fp16 endpoint response is far from linear (possibly rounding-driven divergence; not tested), so
    the linearized amplitude fit is unreliable.
  - Validation picked λ = 0 (constant) and the strongest λ (slow, alternating). Validation latent: static refit
    0.0669; constant 0.0658; slow 0.0739; alternating 0.0656.
  - The evaluation proceeds as frozen: it tests these particular coefficients by actual sampling. It does **not**
    validate the fitting procedure. A resolved fit would need fp32 finite differences and/or larger ε (not run).
- 2026-10-28, review follow-up (recorded before any of its runs):
  - **From existing data:**
    - Absolute coherence: alternating C = 1551 vs original 1545, i.e. unchanged. I rises 5787 → 6728, and total
      prediction-error energy rises 229 → 259. The lower C/I is added fluctuating error, not reduced coherent error.
    - FD: at |a| = 0.02 the alternating response moves the endpoint by only 2.0–4.3% of ‖e‖ (constant: 31–71%).
      The R(2ε)–R(ε) discrepancy is 1.4–2.3% of ‖e‖ in absolute terms. The relative failure mainly reflects a very
      small alternating response.
  - **Implementation controls:**
    - (a) every pattern with amplitude 0 reproduces the static-refit latents bit for bit (2 seeds);
    - (b) the hook's fp32 path forced with c = 1 exactly, on the reserved seeds 4100–4227 (an implementation diagnostic
      for the original-vs-refit comparison on those seeds, not a selection). It tests whether the refit's 1.6% gain
      is arithmetic.
  - **Direct coordinate search** (no linear-response model), constant vs alternating only:
    - Both start at a = 0 around the static refit; bounds |a_f| ≤ 0.02 for both.
    - Steps δ ∈ {0.01, 0.005, 0.0025}. At each δ, one pass over the families (qkv, proj, fc1, fc2), evaluating
      a_f + δ and a_f − δ (clipped to the bounds) and keeping the best actual calibration loss if it improves. That is
      24 evaluations + 1 start per construction, identical for both.
    - Calibration loss: mean endpoint latent rel. MSE vs FP on seeds 5000–5007.
    - Candidates: the coefficients after each δ level (3 per construction) plus a = 0. Selection: lowest mean latent
      rel. MSE on validation seeds 5008–5023. Then freeze (`frozen_cs.json`).
    - Evaluation on **new untouched seeds 4300–4427** (128): static refit, CS-constant, CS-alternating. Paired CIs
      vs static refit and between the two CS arms; C, I, total energy.
  - **Report wording** as proposed in review: "deployed latency unmeasured".
