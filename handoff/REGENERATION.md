# Regeneration (user instruction 2026-10-29: no trained or generated artifact is uploaded; everything is regenerated on the fresh VM)

Only code, protocols, results (JSON/CSV/reports), the static prompt file and the ImageReward wheel are in Git. Public upstream inputs are downloaded at
pinned revisions with checksums (`scripts/fetch_upstream.py`). Everything the project generated is rebuilt with the commands below.
`handoff/REFERENCE_HASHES.json` holds the old VM's SHA-256 of every generated file, used only to **report** whether a regeneration is bitwise identical.
On a different GPU or software stack, numerical differences are expected. Re-measure the controls instead of reusing old numbers.

## A. What the DiT pilot needs (`scripts/bootstrap_vm.sh` runs all of it)
| step | command | output | approx. cost (A30) | check |
|---|---|---|---|---|
| upstream | `python scripts/fetch_upstream.py --what dit,vae,fidref` | DiT-XL/2 weights, sd-vae-ft-ema @ f04b2c4, ADM ref batch | download ≈ 4.5 GB | sha256 of DiT and the ref batch must match |
| FID reference stats | `python -m boa_dit.eval_fid --ref` | results/stage0/ref_stats.npz | minutes | reported vs old sha a11af8b0… |
| heig-8 W4 + rank-8 carry | `python -m boa_dit.temporal.rebuild --device cuda:0` | ckpt/ckpt_deflate_heig_r8_w4/model.pt, ckpt/temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt, ckpt/temporal/heig8_layers.pt | ≈ 0.1 h | rebuilt carry vs the committed `results/deflate/quant/deflate_heig_r8_w4/carry.pt` (bitwise reported) |
| static-A8 smoothing + reference tables | `python -m boa_dit.temporal.dit_act calib` | ckpt/temporal/dit_smooth_s.pt, dit_static_b_{single8,single6,step8}.pt, dit_calib_hist.pt | ≈ 0.2 h | reported vs old hashes |
| pilot's own artifacts | `scripts/run_dit_joint.sh` (smoothed common base, recalibrated A8 tables, carries, evaluations) | ckpt/dit_joint/, results/dit_joint/ | see PROTOCOL.md profile | protocol checks C5 |

## B. Parked threads (only if the user resumes one; costs are A30 estimates from the original runs)
| thread | regenerate | approx. cost |
|---|---|---|
| E3 confirmation (DiT carry refits, H16, FID) | `scripts/run_e3c.sh` (needs the heig-8 base from A, then builds H16 via `quantize_dc --mode heig --r 16`) | ≈ 12 h (15 training runs + 6 FID-2k) |
| DNS / C-DNS | dns-cov branch: `scripts/run_early.sh` then `scripts/run_f6.sh` (calibration, checks, traces, sampling, FID) | ≈ 5 h |
| Follow-up E1/E3 (DiT) | followup-weights: `scripts/run_f4.sh` E1, E3 sections | ≈ 2.5 h |
| PixArt W4A4 base (needed by all PixArt work) | `python scripts/fetch_upstream.py --what pixart,mjhq`; `python -m boa_dit.w4a4.common` embeddings (`WC.embed_all`); `python -m boa_dit.w4a4.caches` (block-0 calibration cache); `python -m boa_dit.w4a4.quantize --arm hsvd_r4_b0.5_kv16_u2_rs` (temporal-oracles / w4a4 code) | ≈ 1–2 h + 12 GB download |
| PixArt branch refit (Part 2, diagnostics, screens) | e3-confirm: `python -m boa_dit.e3c.pixart_refit check/cache/profile/plan/design`, then `px_diag`, `px_screen`, `px_fresh` (PARKED_WORK §1 order) | cache 0.5 h; design 1.3 h; diagnostics ≈ 4 h; each screen ≈ 2 h |
| PixArt activation feedback (DCT-4/16, D1/D2) | temporal-oracles: `scripts/run_c3_*.sh` | ≈ 3.4 h + ImageReward env |
| Follow-up E2 (PixArt tile rounding) | followup-act: `scripts/run_f4.sh` E2 section | ≈ 1.2 h |
| weights-joint / weights-scale | weights-joint branch protocols | ≈ 2 h |

Seeds and prompts are fixed by each protocol (`handoff/SEED_REGISTRY.md`), so regenerated runs use the original seeds.
