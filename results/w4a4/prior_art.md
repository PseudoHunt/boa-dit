# Prior art for the W4A4 phase (recorded 2026-10-26)

## H-SVDQuant: arXiv 2610.11315 (posted 2026-10-08)
Wang, Zhao, Yuan, *Deflating the Hessian: Rank-4 W4A4 Quantization for Multimodal Diffusion Transformers*.
No code released.

**What it publishes that overlaps our deflate-and-carry core.** A 16-bit rank-r branch, plus a deflated Hessian
H_⊥ = H − HL₁(L₁ᵀHL₁)⁻¹L₁ᵀH, plus GPTQ on the residual. With L₁ = the Hessian eigenvectors this is exactly our P·H·P
with the carry. Our DiT-XL/2 weight-only W4 results (heig-8: FID-5k 16.21 vs uniform 18.14 vs FP 12.19) are therefore
not a claim; they are a reproduction on another test bed.

**Their recipe:**
- **Subspace:** Hessian-whitened weight SVD, L^(β) = H^(−β)[H^β W]_r with β = ½.
- **Output factor:** L₂ is refit in closed form.
- **Activation noise:** a diagonal surrogate Σ̄_A for A4, with weight λ_A (0.05 on PixArt-Σ).
- **Smoothing:** a closed-form smoothing vector, d_i ∝ h_i^{1/4} / ‖P_i‖^{1/2}.
- **Iterations:** two outer iterations; the best state is kept by W4A4 output MSE.
- **Setting:** W4A4, rank 4, weight and activation groups of 64.
- **Models:** PixArt-Σ, SANA-1.6B, FLUX.1-dev, Qwen-Image.
- **Metrics:** PSNR / LPIPS against the 16-bit model's images, on MJHQ and sDCI.
- **Their App. C:** derives Hessian eigenvectors as optimal for tr(H_⊥), but never runs that choice.

## DiRotQ: arXiv 2605.16732 (code released)
- **Method:** keeps the top ~10% activation-PCA directions in 16-bit and random-rotates the rest.
- **Standing:** beats H-SVDQuant on FLUX; loses on PixArt-Σ and SANA.

## Target numbers: their Table 1, PixArt-Σ
| method | MJHQ PSNR | MJHQ LPIPS | sDCI PSNR | sDCI LPIPS |
| --- | ---: | ---: | ---: | ---: |
| H-SVDQuant r4 | 16.47 | 0.386 | 15.36 | 0.422 |
| SVDQuant r4 | 14.94 | 0.478 | 13.75 | 0.521 |
| SVDQuant r32 | 16.27 | 0.385 | 15.45 | 0.413 |
| DiRotQ | 16.36 | 0.389 | 15.39 | 0.416 |

Not all of their protocol is stated (steps, guidance, LPIPS backbone), so expect offsets. Our SVDQuant-r32 reproduction
calibrates that offset (`rule.md`, §2).

## What is still open (not in either paper)
1. **Subspace:** Hessian-eigen vs β = ½. Derived as optimal in their App. C, never run.
2. **Rank allocation across layers:** they fix r = 4.
3. **A network-level objective:** nobody has it.
4. **An asymmetric (FP-prefix vs quantized-prefix) target:** not in theirs.

## Evidence from our DiT-XL/2 work that bears on 3
Engine round 2, §2 (`results/engine/sensitivity.json`) measured ρ, the leverage of heig-8's block error in the
network-sensitivity factor F_l:

| block | 2 | 10 | 20 | 27 |
| --- | ---: | ---: | ---: | ---: |
| ρ | 2.30 | 0.18 | 0.015 | 0.029 |

Away from block 2, GPTQ-plus-carry already leaves its error where the output is insensitive.
