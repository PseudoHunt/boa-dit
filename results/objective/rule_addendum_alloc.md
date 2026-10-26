# Addendum (2026-10-26, before any run): band-1 attribution and latent-aware rank allocation (DiT, heig-8)

- **(1) Attribution.** For each of the 112 (block, layer) pairs, restore that layer to FP in heig-8 and remove its
  carry.
  - c_ℓ = 1 − E_b1(restored)/E_b1(heig-8). E_b1 is the band-1 guided-error energy (4 eps channels, band 1 of D2) vs FP
    at steps 20 and 25 of the FP trajectory, eval seeds 0–63, summed.
  - Latent share = 1 − relMSE(restored)/relMSE(heig-8) on seeds 0–63, with a paired bootstrap CI. heig-8's seeds 0–63
    rerun matches the stored values exactly (`results/engine/attribution/blocks.json`).
  - **Concentration** = Σ top-10 c_ℓ / Σ_ℓ max(c_ℓ, 0). **Stop after (1)** if it is < 25%.
- **(2) heig-latent-alloc.**
  - r_ℓ = min(32, round(k · max(c_ℓ, 0))), floor 0. k is chosen by bisection so that total carry bits =
    0.148 ± 0.005.
  - The subspace is the top-r_ℓ eigenvectors of H (`quantize_dc.py --mode heig --r star --r_config`).
- **(3) heig-alloc (control).** The redirect's 90% rule is degenerate on DiT (r90 ≥ 26 everywhere, so the cap binds
  everywhere and it equals heig-8). The pre-registered fallback is used instead:
  - greedy by the raw marginal e_k = λ_k‖E v_k‖² per carry parameter;
  - E = the uniform-W4 residual, (λ_k, v_k) from the FP Grams;
  - rank cap 32, same bits.
- **Both arms:** latent MSE on seeds 0–511 (paired CI vs heig-8), and band-1 energy at steps 20–25.
  - FID-5k only for the winner, and only if it beats heig-8 by ≥ 5% with the CI excluding 0.
- **Budget:** this item is about 1.5 A30-h, taking the objective phase to about 3 h against its 2 h cap. That was
  accepted in the request ("about 1 h").

**Correction (2026-10-26, before any allocation arm was evaluated).** The first attribution run used eval seeds 0–63.
That would fit the allocation on the test set, so it was voided (`layer_attrib_VOID_eval_seeds0_63.json`; the two arm
builds were stopped and deleted before evaluation). The attribution reruns on seeds **1100–1163**, which are disjoint
from eval 0–511, calibration 1000–1031 and held-out 1032–1047. The FP trajectory, FP latents and heig-8 baseline for
those seeds are computed in the run. Everything else is as above.
