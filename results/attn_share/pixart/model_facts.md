# PixArt-Σ 1024 — model facts (verified against the loaded objects; `model_facts.json`)

| fact | expected (handoff) | found |
| --- | --- | --- |
| transformer blocks | 28 | 28 |
| hidden | 1152 | 1152 |
| heads × head dim | 16 × 72 | 16 × 72 |
| patch | 2 | 2 |
| latent | 4 × 128 × 128 | 4 × 128 × 128 |
| image tokens | 4096 | 4096 |
| modulation | adaLN-single | ada_norm_single |
| cross-attention attn2 | yes | True |
| sampler | DPM-Solver++, 20 steps | DPMSolverMultistepScheduler (dpmsolver++), 20 steps |
| guidance | 4.5 | 4.5 |
| text encoder | T5-v1_1-XXL fp16 | T5EncoderModel, d_model 4096, 24 layers; fp32 shards converted to fp16 on download (disk limit); loaded in fp16 |
| VAE | SDXL | AutoencoderKL, scaling 0.13025 |
| GPU free after T5 dropped | ≥ 14 GB | 21.9 GB |

- CFG batch layout: rows [0, B) uncond (empty prompt), rows [B, 2B) cond (pipeline: cat([negative, prompt])). One transformer call per step; the step counter is a forward pre-hook (asserted: 20 calls).
- All 16 images are batched in one sampler call (32 rows with CFG). Maps are computed one head at a time in fp32 ([32, 4096, 4096] = 2 GB per head); the VAE decodes one image at a time.
- Sweep: all 28 blocks, all 16 heads, steps 0–9; no block cut was needed (1-step timing projected ≈ 0.11 GPU-h for the statistics).
- Disk: the T5 shards were downloaded in fp32 and converted to fp16 one at a time; one shard is on /home and symlinked (quota on /home/jl_fs). Tokenizer needed `sentencepiece` (installed). transformers 5 ignored diffusers' `torch_dtype` for T5, so T5 is loaded explicitly in fp16.
