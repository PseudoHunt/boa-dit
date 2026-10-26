# Addendum (2026-10-26, before any PixArt arm is screened): the screen switches to MJHQ-64 images

**Why.** Three results:
- **K/V at 16-bit:** cross-attn K/V at 16-bit moves MJHQ-64 PSNR by +4.12 dB (12.40 → 16.52) but moves the step-error
  proxy by only 2% (0.00297 → 0.00290).
- **C is also blind:** its step-coherent part C moves 1.08× (guided; cond 1.08×, uncond 1.00×), against 1.02× for the
  total (`coherence.json`).
- **Range search:** it cuts the proxy 5% (0.00281) but moves PSNR −0.08 dB.

So per-step noise-prediction error, total or coherent, misses the conditioning-path error that decides the image. Per
the user's rule, the screen becomes images.

**New screen.**
- MJHQ-64 (the first 64 of the MJHQ-1K set; seed = prompt index), paired per-image PSNR and LPIPS-VGG vs bf16 for each
  arm and for the base, with a paired bootstrap 95% CI (2,000 resamples, seed 0).
- **Noise floor:** the fp16-vs-bf16 control, 21.27 dB / 0.238 LPIPS.
- **An arm advances iff** its paired mean PSNR gain over the base is ≥ +0.3 dB with the CI excluding 0, **and** its
  LPIPS is not worse (CI upper bound ≤ 0).
  - The 0.3 dB bar is the smallest dB bar among the original reads (rule.md reads 2 and 4).
  - The original read thresholds then apply at MJHQ-1K as before.
- **The step-error proxy and the band-1 proxy** are still reported as secondary numbers.
- **CFG+K** must beat the better of its parts by ≥ 0.2 dB (CI excluding 0) to count as a combination; this replaces
  the proxy's 2% rule.
