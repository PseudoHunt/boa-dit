# VAR-d30 — model facts (verified against the loaded objects)

| fact | expected | found |
| --- | --- | --- |
| code | FoundationVision/VAR | commit 78b9539 (external/VAR) |
| weights | HF FoundationVision/var | vae_ch160v4096z32.pth, var_d30.pth |
| depth / hidden / heads | 30 / 1920 / 30 × 64 | 30 / 1920 / 30 × 64 |
| scales | (1,2,3,4,5,6,8,10,13,16) | same; cumulative tokens 1, 5, 14, 30, 55, 91, 155, 255, 424, 680 |
| attention | block-causal, KV cache | KV cache at inference (attn_bias None); attn_l2_norm (normalized q·k × learned per-head scale) |
| sampling (repo demo) | — | cfg 4, top_k 900, top_p 0.95, more_smooth False, TF32 on (README FID setting differs: cfg 1.5, top_p 0.96) |
| same seed twice | identical tokens | identical images (16-image batch) |
| stock attention kernel | — | torch fused `scaled_dot_product_attention` (no flash-attn / xformers installed) |

Deviations, stated before the arms ran:
- **Batch 16, not 32:** batch 32 ran out of memory (the KV cache for 64 CFG rows × 30 blocks, plus fp32 weights). Stock speed at batch 16 is 0.095 s/image, so 1,000 images per arm fit.
- **Seeds:** VAR's sampler takes one RNG seed per call, so the matched unit is a fixed batch of 16 (labels and batch seed from default_rng(0)), identical across arms.
- **Parity:** stock uses fused SDPA, so no explicit path reproduces its tokens (images identical: False). Per call on identical inputs: logits fp16 vs fp32 max rel 6.9e-04 (tolerance 1e-3); explicit fp32 softmax output vs SDPA 2.9e-03, against an SDPA math-vs-default floor of 9.5e-04. Every arm (`full` included) uses the same explicit path, so arms differ only in the mask; a `stock` arm checks that `full` is statistically equivalent to stock.
