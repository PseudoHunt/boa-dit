# CogVideoX-2B — model facts (verified against the loaded config and pipeline)

| fact | expected | found |
| --- | --- | --- |
| blocks / heads | 30 / 30 × 64 | 30 / 30 × 64 |
| RoPE | none in 2B | use_rotary_positional_embeddings = False (asserted in the processor) |
| tokens | 226 text + 13 × 30 × 45 = 17,550 video | 226 + 17,550 (order: text, then video by frame, row, column) |
| sampling | 480 × 720, 49 frames, 50 steps, guidance 6.0, default scheduler | same; seed 0 |
| stock seconds per video | — | 158 s (fused SDPA) |
| free GPU memory before sampling (T5 dropped, VAE tiling + slicing) | ≥ 12 GB | 19.1 GB |

Deviations:
- **T5** was loaded explicitly in fp16; transformers 5 ignores diffusers' `torch_dtype`, as found on PixArt.
- **Attention implementation:** a fully explicit fp32 softmax over 1,024-query chunks measured 26 min (random), 31 min (window) and 59 min (oracle) per video, which would exceed the 2.5 h cap. Every arm therefore uses one chunked path: fused SDPA per 1,024-query chunk with a boolean keep-mask (fp32 accumulation and softmax inside the kernel). The full arm passes an all-True mask so it takes the same kernel path. oracle-10 computes explicit fp32 logits (all heads at once) only to select each video query's top 10% of keys. No N × N map is materialized. One forward of the full arm differs from stock by 2.4e-4 (relative, on random inputs).
- **C1 statistics** are computed on the stock run, with explicit fp32 rows for the 128 sampled video queries (cond branch) at steps 0, 10, 11, 25 and 40.
