# boa-dit

Proof of concept: BoA (GPTQ-style, Hessian-aware) weight-only quantization of DiT-XL/2 (256 px, class-conditional), with
calibration statistics reweighted by diffusion-sampler gain and adaLN gate energy. The project ran as a sequence of
pre-registered gates. It ended in **STOP**: reweighting did not beat plain pooled statistics, and the time-indexed
low-rank pivot did not beat a static correction at equal bits.

`boa_dit/attn_share/` and `results/attn_share/` hold a separate, ongoing experiment (sample-independent attention). It
does not touch quantization.

## Where to start reviewing
- `results/REPORT.md`: dated log of every decision, result and bug, in order.
- Gate write-ups, each with its pre-registration above a marker and generated results below it:
  `results/gate1.md` (Stage 1 probes), `results/gate2.md` (W4/W3 pooled / uniform / full),
  `results/gate2b.md` (W4 diagnostic arms), `results/gateA.md` (low-rank residual correction + row attribution and
  guard arms).
- `results/env.md`, `results/model_facts.md`: environment and model facts; `results/gpu_hours.csv`: compute log.

## Code map
| file | role |
| --- | --- |
| `boa_dit/model.py`, `sample.py` | DiT/VAE loading, deterministic DDIM-50 sampler with CFG (official `forward_with_cfg`), step context for hooks |
| `boa_dit/probe_gain.py`, `probe_gates.py`, `probe_attn.py` | Stage 1: sampler gain g_t², adaLN gate energy, Gram dissimilarity, attention stability |
| `boa_dit/calib.py`, `weights.py` | calibration input capture, per-bucket weighted Gram accumulation, bucket/branch/row-group weights |
| `boa_dit/qutils.py`, `quantize.py` | wrappers around the unmodified BoA solver (`external/boa`, `quantizers/boa.py`); block-wise quantization driver |
| `boa_dit/eval_latent_mse.py`, `eval_fid.py` | primary metric (matched-seed relative latent MSE, seeds 0–511) and FID-2k |
| `boa_dit/fp_grams.py`, `lowrank.py`, `item2.py` | Gate A: FP Grams, low-rank residual correction, row attribution, guard arms |
| `scripts/make_*.py` | regenerate every results table from the JSONs (no hand-typed numbers) |

## Reproducing
`git clone --recursive`, then `pip install -r requirements.txt`. The DiT-XL/2-256 weights and the ADM reference batch
are downloaded on first use. Each module is run as `python -m boa_dit.<module> --help` (see its docstring); each
results JSON stores the exact config it was produced with (`config`, `git_sha`). Known host issue: GPU
peer-to-peer copies were silently wrong on the original machine, so every checkpoint load goes through CPU and is
checked tensor by tensor (`model.load_quant_state`).
