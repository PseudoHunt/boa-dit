# rfb1 — anchor-referenced activation feedback with anchor reuse and bridge (pilot R) — pre-registration

Written 2026-10-31, **before any check or quality result of this experiment**. Specification: the user's handoff "two coupled activation-feedback /
anchor / bridge pilots" (2026-10-31), section 3 (pilot R) with the shared sections 1, 2, 5, 6. Pilot J (cfb1) is a separate experiment on branch
`coupled-feedback-bridge-pilot` with its own protocol; its parameters are not merged with these.

Branch `reference-feedback-bridge-pilot` (tag **rfb1**), from `activation-bridge-overnight` 5023a0a (validated abn1 code). New code:
`boa_dit/feedback_bridge/core.py` (quantizer rules, sampler) and `boa_dit/feedback_bridge/rfb.py` (stages); the abn1, ler1 and dit_joint arithmetic is
imported unchanged. Source hashes are in `frozen.json`, written when the gate starts.

Pre-existing state (checked 2026-10-31): no running GPU job (0 MiB, only jupyter-lab); no cfb1/rfb1 branch or protocol on any branch or worktree
(J has not started, so R runs first, then J); abn1 complete (3.407 A30-h).

## 1. Fixed bed (unchanged from abn1)
- DiT-XL/2 256, DDIM-50 eta 0, CFG 1.5 (`weight_anchor.run.guided`, discrepancies on output channels 0–3), fp16 autocast, fp32 sampler.
- Cheap model: the original unrefitted common smoothed rank-8 W4 base (f017af87…), its carry (b4a537b0…), smoothing (7ee15c0c…) and static single8 A8
  tables (b73030de…). High model: FP DiT (`load_dit`), with a separate module instance and no shared hooks or state.
- K4 anchors at 0, 4, …, 48. At an anchor, the cheap model is called once on the pre-correction latent; its quantizer state commits there. FP is
  called on the same latent, timestep, labels and CFG layout, and d_a = cheap − FP. Between anchors, the prediction is cheap − d_a.
- Bridge at anchor c after a (k = c − a): x_c ← x_c − λ(d_c − d_a)·Σ_{j=1}^{k−1}(j/k)·b_{a+j}·Π_{m=j+1}^{k−1}A_{a+m}. After the jump, the DDIM step uses the
  already-computed FP prediction, with no re-evaluation and no state recomputation; the cheap state committed by the one pre-correction call is kept.
  λ = 1. The final interval after anchor 48 stays uncorrected.
- Every anchored arm makes 50 cheap and 13 FP calls; bridge arms make 12 corrections. The quantizer advances once per cheap call, i.e. 50 updates per site.
- Latent error is `weight_anchor.run.rel`: per image, ‖x − x_FP50‖² / ‖x_FP50‖² over the whole 4×32×32 endpoint latent, against the paired FP50
  endpoint (same seed, label and batch). PSNR and LPIPS-VGG are computed on the decoded images (`weight_anchor.controls.metrics`).

## 2. Quantizer rules (`core.FBQuant`)
- **G**: existing `JointQuant._fb` arithmetic (DCT-4 feedback path: guarded clip to [lo, hi_rep], spare code, fp16 consumed cast) with the carried state
  dropped before every call (abn1 G, bitwise).
- **ACB / AC**: existing DCT-4 feedback, unchanged.
- **R** (section 3.2, explicit). Let xcl = clip(xs, lo, hi_rep).
  - *Anchor call:* the arithmetic is exactly G on this input. m_raw = consumed − xcl, i.e. rounding relative to the **clipped** input; clipping error
    xcl − xs is not part of the reference and is reported separately. The stored reference is fp16(m_raw) (R-full) or fp16(Uᵀm_raw) (R-DCT4). The
    rolling state is set to zero.
  - *Held call:* fed = clip(rec(ref) + rec(s_prev), lo − xcl, hi_rep − xcl); u = xcl + fed; uc = clip(u, lo, hi_rep); consumed = the existing quantizer's
    fp16 output on uc; r0 = uc − consumed; r = clamp(r0, ±Δ/2); state = fp16(r) (R-full) or fp16(Uᵀr) (R-DCT4). The reference is not added again in the
    residual update.
  - Both reference and state reset at batch start (step 0) and refresh at every anchor. R-full keeps full tensors for both; R-DCT4 projects both
    (U = the existing 4-term token DCT). Both are stored in fp16; there is no INT4 compression.
  - With anchors disabled (test only), the reference stays zero and there are no resets, so R-DCT4 equals ordinary feedback.
- Ideal-arithmetic identity (local, not an endpoint guarantee): q_t − x_t − m_a = s_{t−1} − s_t (full); with projection,
  (Pm_a − m_a) + (s̃_prev − s̃_t) − s⊥_t. The repeated reference mismatch is measured, not omitted.

## 3. Arms
| arm | quantizer | bridge |
|---|---|---|
| G | matched-range stateless | off |
| GB | G | λ = 1 (matched baseline) |
| ACB | old DCT-4 feedback | λ = 1 (fixed old stack) |
| RFB | R-full | λ = 1 |
| RDB | R-DCT4 | λ = 1 |
| RD | R-DCT4 | off |
FP50 is the reference. All arms have matched call counts. No coefficient tuning. Optional ZB is **not run**: the budget is kept for the mandatory work
and it is descriptive only. There is no OCB and no A6.

## 4. Data and batch
- Development: **60200–60231** (32, already inspected, exploratory; freshness not claimed). Checks use 60200–60207 and overnight 110000–110031 (evaluated)
  for bitwise reproduction at batch 32. Diagnostics use 60208–60215.
- Fresh confirmation: **122000–122127** (128). Audited 2026-10-31: word-boundary search of *.py/*.md/*.csv/*.txt/*.yaml on all local and remote branches
  and all 18 worktrees found no 120000–122999 match (the method finds the known 96100 block). No shift is needed. Reserved in `handoff/SEED_REGISTRY.md`
  before generation.
- Batch: stage `profile` measures R-full peak memory on development noise (6 steps). Frozen rule: the largest of 32/16/8 whose R-full peak reserved memory
  is ≤ 0.85 × device memory. One batch size for all R arms and references in dev, confirmation and diagnostics. The chosen value is recorded below as an
  amendment before any quality sampling. The bitwise reproduction of the overnight arms uses batch 32 (the overnight batch).

## 5. Correctness gate (stage `checks`; all must pass before quality sampling)
S1 reproduces overnight G, GB, ACB, FP50 bitwise (110000–110031, batch 32). S2 λ = 0 equals the no-bridge arm (GB→G, ACB→AC, RFB→RF, RDB→RD) in latent,
final rolling state and reference. S3 K1 = FP50 with zero corrections for G, feedback, R-full, R-DCT4 and gain-scaled J (0.5, 0.5). S4 checks counts
(50 + 13, 12 corrections), 50 quantizer calls per site, and that cheap and FP anchor inputs are identical (pre-hooks on both embedders, 13 compared). S5:
perturbing one sample leaves the others bitwise unchanged; there is a reset between batches; a quantizer-level unit test shows row independence of
outputs, states and references for R-full, R-DCT4, J and feedback. S6 resume immediately before (8) and after (9) an anchor is bitwise in latent, state
and reference (RDB, RFB, ACB). S7 instrumentation (diagnostic dict and R accounting) on/off is bitwise. S8: with no anchors, R-DCT4 equals ordinary
feedback (incl. state, empty reference); an R anchor call equals the G call with junk state and reference present, and the stored state is zero. S9: R
accounting includes input clipping, guard discard, second clip, residual clamp, projection, state storage, reference storage and Pm − m; the identity
consumed − xs = (xcl − xs) + fed + (uc − u) − r0 has relative residual energy ≤ 1e-8 (fp32 rounding only). S10 counts state and reference bytes from the
real tensors (reported). The synthetic fp64 constant/linear examples (abn1 C9) must pass. An alternating example is reported descriptively. The DDIM
coefficients reproduce `ddim_sample` (rel ≤ 1e-5, fp32), with signs reported. A deterministic bug may be fixed and documented before evaluation;
tolerances are not loosened.

## 6. Development gate and confirmation (frozen)
- Before dev, the full confirmation (all six arms + FP50 + metrics) and diagnostics are reserved in the budget check.
- Gate: confirmation runs iff RFB or RDB has dev mean latent error ≤ 0.95 × GB **and** mean LPIPS ≤ LPIPS(GB) + 0.005. Otherwise R confirmation stops (the
  cohort stays unopened), results are written, and the queue proceeds to diagnostics and then J.
- If the gate passes, all six arms run on all 128 confirmation images; never only the favorable arm.
- Primary practical contrast **RDB / GB**. "Useful" iff ratio of means ≤ 0.95, 95% paired-bootstrap upper < 1, and the ΔLPIPS CI is not wholly above
  zero. Otherwise "smaller positive" (upper < 1), "harmful" (lower > 1), or "inconclusive" (never equivalence).
- Secondary (exploratory, unadjusted): RFB/GB (full-state mechanism), RDB/RFB (compression), RDB/ACB (repair of the harmful stack only), RD/G and
  RDB/RD (interaction, no additive claim), ACB/GB, GB/G. If only R-full helps, it is reported as a full-state result with its memory cost.

## 7. Statistics
Paired image bootstrap, 10,000 draws, numpy seed 0 (`weight_anchor.shared.contrast`); ratio of means, mean/median, PSNR, LPIPS, wins. No step- or
token-level bootstrap. No head-to-head significance claims between R and J.

## 8. Diagnostics (stage `diag`, 60208–60215, all six arms)
- Per-interval R accounting energies: m_raw, reference storage, Pm − m, rec(ref) − m, clipping, guard discard, second clip, r0, clamp, (I − P)r, state
  storage, q − xcl − ref, identity residual.
- Interval quantities (abn1 s5 definitions): H, v, cos, residual, raw/held/jump energies, anchor vs between energies.
- FP replay at starts 0, 20, 40 (cosine of the actual deviation with H and with v).
- Bridge jump norms, per-step held discrepancy energies, endpoint error.
- State + reference bytes per image from the real tensors, and one warm synchronized trajectory per arm at the frozen batch (fake-quant simulation; no
  packed-kernel claim).
- Diagnostic FP calls never enter inference (S7).

## 9. Budget and queue
Cap **3.0 A30-h** for rfb1 (ledger tag rfb1 in `results/gpu_hours.csv`, incl. profile, checks, failures, decoding, metrics). The combined R + J ceiling is
6.0. Each stage estimates its workload from the profiled per-trajectory times (×1.10 + metrics) and is skipped whole if spent + estimate + reserve > cap.
There is a runtime stop before every batch and resume from per-batch checkpoints. Order: profile → checks → dev → conf (if the gate passes) → diag →
report; queue `bash scripts/rfb1_queue.sh` (flock, one retry after a crash, commit/push per stage). Large latents stay local with sha256 in
`artifacts.json` and are regenerable from code (standing instruction: nothing trained is uploaded).

---

## Amendments and results
(appended below, dated)
- 2026-10-31 04:48 UTC, profile (0.029 A30-h, no quality data): R-full at batch 32 OOM; batch 16 peak reserved 15.29 GiB
  (<= 0.85 x 23.6 GiB) -> **frozen batch 16** for all R arms and references (dev, confirmation, diagnostics). R-full state + reference
  462 MB per image (both CFG rows). Profiled trajectory estimates at 16: FP50 9.8 s, G/ACB 45.4 s, RDB 46.1 s, RFB 61.6 s.
  Source hashes frozen in frozen.json.
- 2026-10-31 05:05 UTC, gate (0.263 A30-h): all checks pass (S1 overnight G/GB/ACB/FP50 bitwise at batch 32; S2–S10, unit row independence, accounting identity rel. residual energy 1.3e-15 / 6.1e-14). Dev started next.
- 2026-10-31 05:15 UTC, dev complete (32, inspected): RDB/GB 0.883 [0.588, 1.301], ΔLPIPS −0.0028 → gate pass (RDB); RFB/GB 1.506 → fail. Confirmation 122000–122127 opened with all six arms.
