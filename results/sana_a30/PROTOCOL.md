# snk1 — real-kernel SANA anchoring feasibility on the A30 — pre-registration

Written 2026-10-31, **before any SANA GPU execution, timing or quality result**. The specification is the user's handoff "Start now on the A30: real-kernel
SANA anchoring feasibility" (2026-10-31). Branch `sana-a30-feasibility` (worktree /home/boa-dit-snk), from `nunchaku-sdxl-feasibility` 44306d5. Script:
`scripts/sana_a30.py`. Tag **snk1**. Amendments are appended below, dated, before any affected result.

Pre-existing state, checked 2026-10-31 07:42 UTC: GPU idle (0 MiB); no jobs; rfb1/cfb1 complete. Setup started 07:45 UTC (cap 90 min); task cap 4 h
(ends 11:45 UTC).

## 1. Environment (reused, not reinstalled)
- `/home/nk_env` (nk1-validated):
  - Python 3.10.20, torch 2.11.0+cu130, diffusers 0.36.0, transformers 4.55.2.
  - Nunchaku **1.2.1+cu13.0torch2.11**: official GitHub release wheel, sha256 82026d55…, tag v1.2.1 = commit af49f249…
  - The unrelated PyPI `nunchaku` is not installed. The main DiT environment is untouched.
- Official v1.2.1 example `examples/sana1.6b.py`:
  - `NunchakuSanaTransformer2DModel.from_pretrained("nunchaku-tech/nunchaku-sana/svdq-int4_r32-sana1.6b.safetensors")`
  - `SanaPipeline.from_pretrained("Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", variant="bf16", torch_dtype=bfloat16)`
  - VAE and text encoder in bf16; 1024², guidance 4.5, 20 steps, `torch.Generator().manual_seed(seed)` on the CPU.
- Pinned checkpoints (downloaded 07:47 UTC):
  - INT4: `nunchaku-ai/nunchaku-sana` (the `nunchaku-tech/` path redirects) at revision **437e41b9…**, `svdq-int4_r32-sana1.6b.safetensors` sha256
    320be49f…
  - Base: `Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers` at revision **e2b3c0cb…**, bf16 variant files:

    | file | sha256 |
    |---|---|
    | transformer | 66545515… |
    | text_encoder 1/2 | e909230a… |
    | text_encoder 2/2 | 48e14f6b… |
    | vae | 15a4b09e… |

  - Both are Sana 1.0, 1600M, 1024 px, non-distilled; no Sana 1.5, no distilled weights, no PAG. The INT4 checkpoint is packed W4A4 (SVDQuant, rank 32)
    on all 20 transformer blocks; embeddings, caption projection, time embedding and final layer stay BF16. No fake quantization and no FP4 (Ampere).
- The stage `env` records GPU model, compute capability, driver, memory, MIG, power limit, clocks and pip freeze in `env.json`.

## 2. Frozen workload
- 1024×1024, batch 1 image (CFG batch 2), **20 steps**, guidance 4.5, default negative prompt "".
- Pipeline defaults otherwise: complex human instruction, max sequence 300, resolution binning.
- No PAG, caching, LoRA, offloading or `torch.compile`; TF32 is not enabled.
- Native scheduler `DPMSolverMultistepScheduler`: dpmsolver++, solver_order 2, midpoint, `prediction_type = flow_prediction`, use_flow_sigmas,
  flow_shift 3.0, final_sigmas_type zero, lower_order_final.
  - **One transformer evaluation per step**, CFG-batched, so actual evaluations = nominal steps = 20.
  - The scheduler state is (latent, `model_outputs` history of 2 converted predictions, `lower_order_nums`, `step_index`).
- Both denoisers are resident simultaneously; one `SanaPipeline` holds the BF16 text encoder, VAE and scheduler.
  - The denoiser is a routing wrapper installed as `pipe.transformer`. It exposes the BF16 model's config and dtype and dispatches each call to the BF16
    (H) and/or Nunchaku (Q) transformer. The native pipeline loop, CFG and scheduler run unchanged.
  - The text encoder stays resident throughout; this is reported.
  - Denoiser-only timing uses precomputed prompt embeddings (identical for both models); loaded-pipeline latency includes text encoding and VAE decode.
- Prompts (mjhq1k, the existing `boa_dit.w4a4.common.mjhq_sets` list; MJHQ meta_data.json sha256 db9cc0a0…), seed = prompt index:
  - timing: **358, 352, 361, 359** (6 / 10 / 32 / 74 words);
  - development: **352–367** (16);
  - confirmation: **464–527** (64).
  - These are free per `handoff/SEED_REGISTRY.md`. A word-boundary audit of all branches found no other use; they are reserved now.
- Tolerances (stage `smoke`): each model is run twice on identical inputs. Bitwise repeatability is the expected tolerance (0). If it is not bitwise,
  the observed max rel L2 ×10 is used for every "reproduces" check.

## 3. Real-kernel and memory verification (stage `kernels`, target ≤ 0.2 A30-h)
- A profiler trace of one Q call and one H call: GEMM_W4A4 launches and CUDA time.
- The list of modules that remain BF16; parameter/buffer bytes per model, text encoder and VAE.
- Peak allocated/reserved for the loaded pipeline with both denoisers, in a hybrid 20-step run plus decode. **Fit iff peak reserved + 1 GiB ≤ device
  total.** No CPU swapping and no 512 px.
- Interface: both models accept the same latent, timestep, embeddings and mask and return the same shape and dtype (flow prediction, 32 channels).
  The Q−H rel L2 at three steps is reported.

## 4. Timing gate (stage `timing`, target ≤ 0.2 A30-h; no profiler)
- Per call at steps 0 / 10 / 19 (actual scheduler timesteps):
  - H and Q, CFG batch 2, CUDA events.
  - 5 warmups, then 20 measured per (model, step), in a randomized interleaved order (numpy seed 0).
  - Median and range.
- Complete native 20-step denoising loops (latent output), arms **H20, Q20, HYB-K4, HYB-K8**:
  - 1 warmup then 5 measured repetitions each, interleaved by repetition, cycling the 4 timing prompts.
  - Median and range.
  - HYB = cheap every step + high at anchors (K4: 0, 4, 8, 12, 16 → 25 calls; K8: 0, 8, 16 → 23 calls), returning high at anchors.
- Loaded-pipeline latency (text encode + loop + VAE decode, PIL output) for H20, Q20 and the passing hybrid(s): 3 repetitions. Model loading and
  download are timed separately.
- Idealized conditions: K4 needs C_Q/C_H < 0.75; K8 needs < 0.85.
- **Gate:** a schedule passes iff its measured hybrid loop median ≤ 0.90 × the H20 loop median and both models fit. K4 is chosen if both pass;
  otherwise K8 if it alone passes.
  - If neither passes: stop quality work, push the report and profiler output, and identify the bottleneck. No SDXL rerun and no architecture switch.
  - Sub-unit ratios above 0.90 are reported as insufficient margin.

## 5. Conditional native-scheduler error-reuse pilot (only if the gate passes; no feedback, refit or bridge)
- Wrapper at the transformer-output interface consumed by the native pipeline, on the full CFG-batched output, before guidance.
  - Guidance is linear, so this equals applying the reuse to the guided prediction.
  - Anchor: Q and H on the identical current state, store d = Q − H, return H. Held step: return Q − d. β = 1.
  - Flow-prediction parameterization and timestep scaling are unchanged.
- Arms, at the selected schedule:
  - **H20**
  - **Q20**
  - **A**: H at anchors, Q between; Q is not called at anchors, which is the deployment control
  - **R**: Q every step, plus H and reuse at anchors
  - **S**: H at anchors, stale H prediction between
  - Every call is counted. R vs A is not matched-cost (A omits Q at anchors).
- Checks:
  - wrapper H20/Q20 equal the unwrapped pipeline (Nunchaku or BF16 transformer installed directly);
  - anchor-every-call equals H20;
  - identical stateless Q = H (BF16 as both) gives d = 0 and the H20 endpoint;
  - reuse disabled equals A with Q at anchors computed but unused (same routing);
  - save/resume at step 9 including the full scheduler state (deepcopy of the scheduler) equals an uninterrupted run;
  - no caching or offload; diagnostic calls never enter inference.
- Development (16 prompts):
  - endpoint latent rel. MSE vs H20, PSNR and LPIPS-VGG;
  - local error persistence at the same states along the H20 trajectory (Q and H at each H20 state): raw energies, lag-1/2/4/8 correlation, and
    energy remaining after held subtraction (e_i − e_anchor);
  - diagnostics do not gate.
- **Development rule:** confirmation runs iff R's mean endpoint rel. MSE is ≤ 0.90 × A's, R is finite, and R's measured warm loop median is
  ≤ 0.90 × H20. The confirmation budget is profiled before it is opened; if it does not fit the cap, stop. Otherwise a scoped screen result is reported.
- Confirmation (64): all arms plus **Hk**.
  - k = the largest integer in 1..20 whose warmed measured BF16 k-step loop latency is ≤ R's.
  - Hk uses the official scheduler with `num_inference_steps = k`, not truncation. k is chosen from timing only, before confirmation metrics.
  - Metrics: same-seed latent rel. MSE, PSNR and LPIPS vs H20; per-prompt values; paired bootstrap 10,000, seed 0; R vs A, R vs Hk, R vs Q20, A vs Q20.
  - Whole-pipeline latency and peak memory are reported.
  - If H20 is faster than R, the efficiency gate has failed.
- These are fidelity-only results, not generation-quality results.

## 6. Bridge-port design note
The native scheduler class, order, coefficients and history are recorded. The DiT bridge is not copied. Any bridge port needs a derivation for the
multistep (latent, history) state and replay validation under its own protocol.

## 7. Budget
- Cap **1.5 incremental A30-h** (ledger tag snk1 in `results/gpu_hours.csv`), including smoke tests, profiling, failures, decoding and diagnostics.
- Setup cap 90 min; task cap 4 h elapsed.
- Each stage checks spent + estimate ≤ cap before starting. Stop and save if blocked.
- No concurrent GPU workloads. Images and latents stay local with sha256, regenerable from the pinned code, models and seeds.
- Nothing is uploaded, and earlier releases are preserved.

---

## Amendments and results
(appended below, dated)
- 2026-10-31 08:10 UTC, smoke and determinism (before any timing or quality result; 0.016 A30-h incl. one FAILED attempt where precomputed embeddings
  needed `negative_prompt=None`):
  - Both models are finite and coherent on the official example; the example images are saved in ckpt/sana_a30/.
  - **BF16 is bitwise repeatable. The official Nunchaku INT4 SANA transformer is not.** Six identical single calls differ by up to rel L2 0.087 (max
    abs 0.375), against Q-vs-H 0.227. This is independent of the padding mask, unpadded tokens and batch size (determinism2.json); two identical
    20-step runs differ by 0.278 in the endpoint latent. The cause is not investigated further (scope: no open-ended debugging).
  - Consequences, fixed now:
    - Any check involving Q cannot be bitwise. "Reproduces" for Q paths will be judged against Q's own measured run-to-run spread, reported, not
      gated by the 10× rule (which would be vacuous at 2.78).
    - Bitwise checks remain for H-only paths.
    - Q's call-to-call noise (~9%) is a non-reusable component of any held discrepancy.
  - Bug found: the router run did not match the direct pipeline even for BF16 (0.394), because `encode_prompt` was called without the pipeline's
    `__call__` defaults (complex human instruction). Fixed: embeddings now use the pipeline's own defaults. Router vs direct is re-checked in `kernels`.
- 2026-10-31 08:13 UTC, kernels and timing (no quality data):
  - **Kernels:** 260 GEMM_W4A4 launches, 41.3 of 59.0 ms CUDA time in the INT4 call. BF16 stays only in 6 linears (time embedding, caption
    projection, proj_out) and the patch-embed conv.
  - **Memory:** both denoisers + text encoder + VAE resident; peak reserved 12.18 GiB of 23.6 GiB (fits with a 1 GiB reserve).
  - **Router:** router vs direct is bitwise for H after the embedding fix.
  - **Per-call Q/H:** 0.515 / 0.519 / 0.515 at steps 0 / 10 / 19.
  - **Loops (medians):** H20 2.345 s, Q20 1.186 s, HYB-K4 1.775 s (**0.757**), HYB-K8
    1.540 s (0.657).
  - **Gate:** both pass; **K4 selected** (timing-only, before any quality result).
  - **Loaded pipeline:** H20 2.68 s, Q20 1.52 s, HYB-K4 2.11 s, HYB-K8 1.88 s.
  - **LPIPS:** computed by a separate metrics script in the main DiT environment from saved uint8 images, because nk_env has no lpips and is not
    modified.
- 2026-10-31 08:22 UTC, checks pass (Q paths judged against Q's own spread; routing/resume bitwise with a deterministic stand-in); development screen: R/A 1.110 [0.951, 1.276] (rule <= 0.90) → **fails**; confirmation 464–527 not opened; Hk not evaluated. Same-state diagnostics: Q run-to-run noise = 57% of Q−H energy; held subtraction leaves 1.56x. Report written; nothing running.
