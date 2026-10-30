# Release inventory: `park-assets-2026-10-29` (PseudoHunt/boa-dit)

- **Date:** 2026-10-30. **Release tag:** `park-assets-2026-10-29`. **Size:** 12 tar parts, 19,722,496,000 bytes (19.72 GB); payload 536 files, 19,721,544,481 bytes (19.72 GB).
- **Read-only.** Nothing was deleted, modified, uploaded or downloaded. Release metadata was read with `gh release view`; contents were listed with `tar -t` on the local staging copies in `/home/park_stage`.
- **A pushed source branch is not an artifact backup.** Git holds code, results JSON and reference hashes. It does not hold the checkpoints, caches or images listed here, except the two small files noted below (carry, qdiff.yaml).
- **Sources:** manifest `handoff/ASSETS.json` at a8eb2fe; `handoff/REGENERATION.md`, `REFERENCE_HASHES.json`, `START_HERE.md` and `PARKED_WORK.md` on `weight-anchor-controls`; parked branch tags `park/2026-10-29/*`; `results/weight_anchor/{REPORT.md,provenance.json}`.

## Integrity checks performed
- The 12 staged parts in `/home/park_stage` match the release asset digests (GitHub `sha256:` digest = manifest `parts` sha256) for all 12.
- `/home/park_stage/verify/B_dit_analysis.tar.part01` (an earlier re-download) also matches.
- The `tar -t` member lists of the 4 reassembled groups equal the manifest `dest` set exactly (16 + 63 + 7 + 450 = 536 files).
- All 536 payload files exist at their manifest `old_path` on local disk: 533 under `/home/boa-dit/ckpt`, plus `/home/boa-dit/results/stage0/ref_stats.npz`, `/root/ext/deepcompressor/.../qdiff.yaml` and `/root/ext/mjhq/meta_data.json`. **All 536 sha256 values match the manifest.** Every file was hashed, not just the item-5 files.
- Upstream (not in the release): `/home/boa-dit/ckpt/DiT-XL-2-256x256.pt` and `VIRTUAL_imagenet256_labeled.npz` match `REFERENCE_HASHES.json` `upstream`.

## Class definitions used
- **1 upstream public:** re-fetchable from a pinned public source, or committed bitwise in git.
- **2 regenerable, verified:** REGENERATION.md §A gives a file-specific command plus a check against REFERENCE_HASHES.json or a committed reference.
  - **2\*:** the same, but the command and hash are documented in `results/weight_anchor/REPORT.md` / `provenance.json` rather than REGENERATION.md.
  - Caveat (REFERENCE_HASHES.json note): a bitwise match is expected only for deterministic steps or the same GPU/software stack. On other hardware the check *reports* a difference; it does not guarantee equality.
- **3 regenerable, unverified:** only a thread-level command (REGENERATION.md §B), with no stated check. A REFERENCE_HASHES entry exists for every file but is for reporting only, and GPU training/sampling will not match it bitwise on a different stack.
- **4 release-only** (no regeneration path in the handoff docs):
  - **4a:** no producing code found.
  - **4b:** undocumented, but a producing code path exists on a parked branch. Not checked to run; treat as release-only for planning.

## Summary (file counts; bytes in parentheses)

| group | 1 | 2 | 2* | 3 | 4a | 4b | total |
|---|---|---|---|---|---|---|---|
| A_dit_base | 0 | 7 (2.74 GB) | 1 (26.7 MB) | 0 | 0 | 8 (0.16 GB) | 16 (2.93 GB) |
| B_dit_analysis | 0 | 0 | 2 (0.42 GB) | 61 (4.31 GB) | 0 | 0 | 63 (4.73 GB) |
| C_pixart_base | 2 (8.9 MB) | 0 | 0 | 5 (4.88 GB) | 0 | 0 | 7 (4.89 GB) |
| D_pixart_analysis | 0 | 0 | 0 | 284 (2.38 GB) | 0 | 166 (4.79 GB) | 450 (7.17 GB) |
| all | 2 (8.9 MB) | 7 (2.74 GB) | 3 (0.45 GB) | 350 (11.57 GB) | 0 | 174 (4.95 GB) | 536 (19.72 GB) |

## Per-artifact table
Groups A–C are listed per file. Image directories (`*/img/*`, all in group D) are aggregated per directory, since every file in one shares group, role and class. Everything else is listed per file. Local = present under its manifest path / sha256 vs manifest. `ckpt/` = `/home/boa-dit/ckpt/` (manifest dest `data/ckpt/`).

| path | group | size | class | local / hash | regeneration command | verification |
|---|---|---|---|---|---|---|
| `ckpt/ckpt_deflate_heig_r8_w4/model.pt` | A | 2.70 GB | 2 | yes / match | `python -m boa_dit.temporal.rebuild --device cuda:0` (REGENERATION §A) | REFERENCE_HASHES.json; provenance.json |
| `ckpt/temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt` | A | 8.3 MB | 2 | yes / match | `python -m boa_dit.temporal.rebuild --device cuda:0` (REGENERATION §A) | REFERENCE_HASHES.json; also bitwise vs committed `results/deflate/quant/deflate_heig_r8_w4/carry.pt` (same sha256, checked) |
| `ckpt/temporal/dit_smooth_s.pt` | A | 0.7 MB | 2 | yes / match | `python -m boa_dit.temporal.dit_act calib` (REGENERATION §A) | REFERENCE_HASHES.json |
| `ckpt/temporal/dit_static_b_single6.pt` | A | 0.3 MB | 2 | yes / match | `python -m boa_dit.temporal.dit_act calib` (REGENERATION §A) | REFERENCE_HASHES.json |
| `ckpt/temporal/dit_static_b_single8.pt` | A | 0.3 MB | 2 | yes / match | `python -m boa_dit.temporal.dit_act calib` (REGENERATION §A) | REFERENCE_HASHES.json |
| `ckpt/temporal/dit_static_b_step8.pt` | A | 0.3 MB | 2 | yes / match | `python -m boa_dit.temporal.dit_act calib` (REGENERATION §A) | REFERENCE_HASHES.json |
| `ckpt/temporal/fp_traj32_all.pt` | A | 53.0 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/temporal/dit.py` / `dit_act.py` (temporal-oracles) | REFERENCE_HASHES.json only |
| `ckpt/temporal/fp_traj_calib1000_all.pt` | A | 52.5 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/temporal/dit.py` / `dit_act.py` (temporal-oracles) | REFERENCE_HASHES.json only |
| `ckpt/temporal/stage1_delta_heig8.pt` | A | 26.7 MB | 2* | yes / match | `python -m boa_dit.temporal.stage1` (record step) -- documented in results/weight_anchor/REPORT.md, not REGENERATION.md | results/weight_anchor/provenance.json + REFERENCE_HASHES.json |
| `ckpt/fp_eval_latents.pt` | A | 8.4 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/eval_latent_mse.py` (`fp_reference`) | REFERENCE_HASHES.json only |
| `ckpt/fp_eval_latents_s3000.pt` | A | 1.1 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/eval_latent_mse.py` (`fp_reference`) | REFERENCE_HASHES.json only |
| `ckpt/fp_eval_latents_s3100.pt` | A | 8.4 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/eval_latent_mse.py` (`fp_reference`) | REFERENCE_HASHES.json only |
| `ckpt/fp_eval_pixels64.pt` | A | 12.6 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/eval_latent_mse.py` (`fp_reference`) | REFERENCE_HASHES.json only |
| `ckpt/fp_eval_pixels64_s3000.pt` | A | 12.6 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/eval_latent_mse.py` (`fp_reference`) | REFERENCE_HASHES.json only |
| `ckpt/fp_eval_pixels64_s3100.pt` | A | 12.6 MB | 4b | yes / match | none documented; created on demand if missing by `boa_dit/eval_latent_mse.py` (`fp_reference`) | REFERENCE_HASHES.json only |
| `results/stage0/ref_stats.npz` | A | 33.6 MB | 2 | yes / match | `python -m boa_dit.eval_fid --ref` (REGENERATION §A) | REFERENCE_HASHES.json (a11af8b0...) |
| `ckpt/ckpt_deflate_heig_r16_w4/model.pt` | B | 2.70 GB | 3 | yes / match | `python -m boa_dit.deflate.quantize_dc --mode heig --r 16 --bits 4` (via e3-confirm `scripts/run_e3c.sh`, REGENERATION §B) | no check stated; REFERENCE_HASHES entry for reporting only |
| `ckpt/e3c/carry_C1L_s0.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1L_s1.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1L_s2.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1t_s0.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1t_s1.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1t_s2.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1u_s0.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1u_s1.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C1u_s2.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C2L_s0.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C2L_s1.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C2L_s2.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C2_s0.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C2_s1.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/carry_C2_s2.pt` | B | 8.3 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/eval_latents.pt` | B | 39.9 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/m_C2_s0.pt` | B | 0.0 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/feats_C0.npy` | B | 16.4 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/feats_C1t_s0.npy` | B | 16.4 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/feats_C2L_s0.npy` | B | 16.4 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/feats_C2M_s0.npy` | B | 16.4 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/feats_C2_s0.npy` | B | 16.4 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/feats_H16.npy` | B | 16.4 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/latents_C0.pt` | B | 32.8 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/latents_C1t_s0.pt` | B | 32.8 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/latents_C2L_s0.pt` | B | 32.8 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/latents_C2M_s0.pt` | B | 32.8 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/latents_C2_s0.pt` | B | 32.8 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/e3c/fid/latents_H16.pt` | B | 32.8 MB | 3 | yes / match | e3-confirm `scripts/run_e3c.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/cal_x0.pt` | B | 49.2 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/cdns_traces.pt` | B | 0.53 GB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/feats_cdns_full.npy` | B | 16.4 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/feats_cdns_ind.npy` | B | 16.4 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/feats_dns.npy` | B | 16.4 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/feats_qdrift.npy` | B | 16.4 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/latents_cdns_full.pt` | B | 32.8 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/latents_cdns_ind.pt` | B | 32.8 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/latents_dns.pt` | B | 32.8 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/fid/latents_qdrift.pt` | B | 32.8 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/paired/latents_cdns_full.pt` | B | 2.1 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/paired/latents_cdns_ind.pt` | B | 2.1 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/paired/latents_dns.pt` | B | 2.1 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/dns/paired/latents_qdrift.pt` | B | 2.1 MB | 3 | yes / match | dns-cov `scripts/run_early.sh` then `scripts/run_f6.sh` (REGENERATION §B) | no check stated |
| `ckpt/observability/traces_cal.pt` | B | 0.21 GB | 2* | yes / match | `python -m boa_dit.observability.run traces` -- documented in results/weight_anchor/REPORT.md, not REGENERATION.md | results/weight_anchor/provenance.json + REFERENCE_HASHES.json |
| `ckpt/observability/traces_eval.pt` | B | 0.21 GB | 2* | yes / match | `python -m boa_dit.observability.run traces` -- documented in results/weight_anchor/REPORT.md, not REGENERATION.md | results/weight_anchor/provenance.json + REFERENCE_HASHES.json |
| `ckpt/followup/e1/confirm_images.pt` | B | 50.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e1/confirm_latents.pt` | B | 4.2 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e1/design_latents.pt` | B | 4.2 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/cache.pt` | B | 79.2 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_design_C1_lr0.0001.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_design_C1_lr0.0003.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_design_C2_lam0.5.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_design_C2_lam2.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_design_C3_lam0.5.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_design_C3_lam2.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_final_C1.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_final_C2.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_final_C3.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_profile_C1.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/carry_profile_C2.pt` | B | 8.3 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/eval_images.pt` | B | 0.13 GB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/followup/e3/eval_latents.pt` | B | 10.5 MB | 3 | yes / match | followup-weights `scripts/run_f4.sh` E1/E3 (REGENERATION §B) | no check stated |
| `ckpt/w4a4/q_hsvd_r4_b0.5_kv16_u2_rs.pt` | C | 2.09 GB | 3 | yes / match | `python -m boa_dit.w4a4.caches`; `python -m boa_dit.w4a4.quantize --arm hsvd_r4_b0.5_kv16_u2_rs` (REGENERATION §B; needs PixArt upstream) | no check stated |
| `ckpt/w4a4/embeds_calib.pt` | C | 81.2 MB | 3 | yes / match | `python -m boa_dit.w4a4.common` (`WC.embed_all`) (REGENERATION §B) | no check stated |
| `ckpt/w4a4/embeds_design.pt` | C | 81.2 MB | 3 | yes / match | `python -m boa_dit.w4a4.common` (`WC.embed_all`) (REGENERATION §B) | no check stated |
| `ckpt/w4a4/embeds_mjhq1k.pt` | C | 2.46 GB | 3 | yes / match | `python -m boa_dit.w4a4.common` (`WC.embed_all`) (REGENERATION §B) | no check stated |
| `ckpt/w4a4/embeds_screen.pt` | C | 0.16 GB | 3 | yes / match | `python -m boa_dit.w4a4.common` (`WC.embed_all`) (REGENERATION §B) | no check stated |
| `data/ext/deepcompressor/examples/diffusion/prompts/qdiff.yaml` | C | 0.1 MB | 1 | yes / match | committed in git: `assets_static/deepcompressor_qdiff.yaml` (BOA_QDIFF) | bitwise identical (sha256 54da5756..., checked) |
| `data/ext/mjhq/meta_data.json` | C | 8.8 MB | 1 | yes / match | `python scripts/fetch_upstream.py --what mjhq` (playgroundai/MJHQ-30K @ 15b0a659) | pinned HF revision; script does no sha256 check |
| `ckpt/e3c/pixart/branches/branch_design_P1_lr0.0001.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P1_lr0.0003.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P1_lr1e-05.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P1_lr3e-05.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P2_lam0.5.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P2_lam0.5_lr1e-05.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P2_lam2.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/branches/branch_design_P2_lam2_lr1e-05.pt` | D | 5.8 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/cache.pt` | D | 0.38 GB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P0.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P1_lr1e-4.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P1_lr1e-5.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P1_lr3e-5.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P2_l0.5_lr1e-4.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P2_l0.5_lr1e-5.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P2_l2_lr1e-4.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/diag_endpoints_P2_l2_lr1e-5.pt` | D | 4.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/grid_P0.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/grid_P1_lr1e-5.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/grid_P1_lr1e-5_restore_o.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/grid_P1_lr1e-5_restore_qo.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/grid_P1_lr3e-5.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) | no check stated |
| `ckpt/e3c/pixart/grid_P1_lr3e-5_restore_o.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) (from the PARTIAL interv2 run) | no check stated |
| `ckpt/e3c/pixart/grid_P1_lr3e-5_restore_qo.pt` | D | 25.2 MB | 3 | yes / match | e3-confirm `python -m boa_dit.e3c.pixart_refit cache/design`, `px_diag` (REGENERATION §B) (from the PARTIAL interv2 run) | no check stated |
| `ckpt/temporal/pixart/design20_cache.pt` | D | 0.34 GB | 4b | yes / match | none documented; produced by `boa_dit/temporal/pixart.py` (temporal-oracles; subcommand unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/dither_w16.pt` | D | 0.1 MB | 4b | yes / match | none documented; produced by `boa_dit/temporal/pixart.py` (temporal-oracles; subcommand unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/static_b_single6.pt` | D | 0.4 MB | 4b | yes / match | none documented; produced by `boa_dit/temporal/pixart.py` (temporal-oracles; subcommand unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/static_b_single8.pt` | D | 0.4 MB | 4b | yes / match | none documented; produced by `boa_dit/temporal/pixart.py` (temporal-oracles; subcommand unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/static_b_step8.pt` | D | 0.4 MB | 4b | yes / match | none documented; produced by `boa_dit/temporal/pixart.py` (temporal-oracles; subcommand unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/static_w16.pt` | D | 0.4 MB | 4b | yes / match | none documented; produced by `boa_dit/temporal/pixart.py` (temporal-oracles; subcommand unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/bf16/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/bf16_b1/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/bf16_b1_smoke/*` (1 files) | D | 6.8 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e1_w16a4_base_screen/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e1_w16a4_feedback_screen/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e1_w4a4_base_screen/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e1_w4a4_feedback_guard_screen/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e1_w4a4_feedback_screen/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e1_w4a4_unif_screen/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e4_bf16_s10/*` (1 files) | D | 54.5 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e4_bf16_s50/*` (1 files) | D | 54.5 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e4_fb_s10/*` (1 files) | D | 54.5 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e4_fb_s50/*` (1 files) | D | 54.5 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e4_single8_s10/*` (1 files) | D | 54.5 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/e4_single8_s50/*` (1 files) | D | 54.5 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/s3_single8_feedback/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/s3_single8_feedback_dct16/*` (8 files) | D | 0.22 GB | 3 | yes / 8/8 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/s3_single8_feedback_dct4/*` (8 files) | D | 0.22 GB | 3 | yes / 8/8 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/s3_single8_feedback_smoke/*` (1 files) | D | 6.8 MB | 4b | yes / 1/1 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/s3_single8_none/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/s3_single8_unif/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/s3_step8_none/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/u_bf16_b1/*` (4 files) | D | 0.11 GB | 3 | yes / 4/4 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/u_single8_feedback/*` (4 files) | D | 0.11 GB | 3 | yes / 4/4 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/u_single8_feedback_dct16/*` (4 files) | D | 0.11 GB | 3 | yes / 4/4 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/u_single8_feedback_dct4/*` (4 files) | D | 0.11 GB | 3 | yes / 4/4 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/u_single8_none/*` (4 files) | D | 0.11 GB | 3 | yes / 4/4 match | temporal-oracles `scripts/run_c3_*.sh` / `python -m boa_dit.temporal.dist gen*` (REGENERATION §B; arm-to-script mapping unverified) | no check stated |
| `ckpt/temporal/pixart/img/w16_a8s_calibrated/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/w16_a8s_golden/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/w16_a8s_none/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/w16_a8s_period2/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/temporal/pixart/img/w16_a8s_random/*` (8 files) | D | 0.22 GB | 4b | yes / 8/8 match | none documented (earlier temporal-oracles stage/ext runs, e.g. run_temporal_alloc3_b.sh, run_ext_block4.sh; unverified) | REFERENCE_HASHES.json only |
| `ckpt/followup/e2/img/adj_0.25/design/*` (16 files) | D | 54.6 MB | 3 | yes / 16/16 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/adj_1/design/*` (16 files) | D | 54.6 MB | 3 | yes / 16/16 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/adj_4/design/*` (16 files) | D | 54.6 MB | 3 | yes / 16/16 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/adj_4/mjhq1k/*` (32 files) | D | 0.11 GB | 3 | yes / 32/32 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/bf16/design/*` (16 files) | D | 54.6 MB | 3 | yes / 16/16 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/bf16/mjhq1k/*` (32 files) | D | 0.11 GB | 3 | yes / 32/32 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/nearest/design/*` (16 files) | D | 54.6 MB | 3 | yes / 16/16 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/nearest/mjhq1k/*` (32 files) | D | 0.11 GB | 3 | yes / 32/32 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/rnd_4/design/*` (16 files) | D | 54.6 MB | 3 | yes / 16/16 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/followup/e2/img/rnd_4/mjhq1k/*` (32 files) | D | 0.11 GB | 3 | yes / 32/32 match | followup-act `scripts/run_f4.sh` E2 (REGENERATION §B) | no check stated |
| `ckpt/w4a4/img/bf16/*` (8 files) | D | 0.20 GB | 4b | yes / 8/8 match | none documented; `boa_dit/w4a4/dtype_control.py` / w4a4 queue scripts (temporal-oracles; unverified) | REFERENCE_HASHES.json only |
| `ckpt/w4a4/img/fp16_unquantized/*` (8 files) | D | 0.20 GB | 4b | yes / 8/8 match | none documented; `boa_dit/w4a4/dtype_control.py` / w4a4 queue scripts (temporal-oracles; unverified) | REFERENCE_HASHES.json only |

## Inputs needed to reproduce the weight-anchor controls (branch `weight-anchor-controls`)
Hashes are from `results/weight_anchor/provenance.json`. "Local" = sha256 recomputed today on this VM.

| input | in release? | class | local / hash | regeneration command | verification |
|---|---|---|---|---|---|
| `ckpt/DiT-XL-2-256x256.pt` | **no** (upstream) | 1 | yes / match (9ec1876e...) | `python scripts/fetch_upstream.py --what dit` | script asserts sha256 = REFERENCE_HASHES `upstream` |
| `ckpt/ckpt_deflate_heig_r8_w4/model.pt` (facee154...) | yes, A_dit_base | 2 | yes / match | `python -m boa_dit.temporal.rebuild --device cuda:0` (needs the DiT weights; ≈ 0.1 A30-h) | REFERENCE_HASHES.json / provenance.json. REGENERATION.md's check column names only the carry. A bitwise model.pt match on a different GPU/stack is **unverified** |
| `ckpt/temporal/rebuild_quant/deflate_heig_r8_w4/carry.pt` (f40ea01d...) | yes, A_dit_base | 2, also in git | yes / match | same rebuild command | bitwise vs committed `results/deflate/quant/deflate_heig_r8_w4/carry.pt`. That git file has sha256 f40ea01d... (checked), so the carry also survives in git |
| sd-vae-ft-ema VAE | **no** | 1 | present: HF cache `/home/.cache/huggingface/hub/models--stabilityai--sd-vae-ft-ema` and `/home/jl_fs/hf/hub/...`, snapshot f04b2c4b | `python scripts/fetch_upstream.py --what vae` (stabilityai/sd-vae-ft-ema @ f04b2c4b98319346dad8c65879f680b1997b204a) | pinned HF revision only; no sha256 check in the script |
| LPIPS VGG weights | **no** | 1 | present: `lpips/weights/v0.1/vgg.pth` inside the pip package (lpips 0.1.4; sha256 a78928a0...) and torchvision `/home/.cache/torch/hub/checkpoints/vgg16-397923af.pth` (sha256 397923af..., consistent with the filename prefix) | `pip install -r requirements-lock.txt` (lpips==0.1.4 ships vgg.pth); torchvision downloads vgg16 on first use | lpips: pinned wheel version. vgg16: filename hash prefix. No project-side check |
| `ckpt/observability/traces_cal.pt` (89b645a0...) — **re-fit only** | yes, B_dit_analysis | 2* | yes / match | `python -m boa_dit.observability.run traces` (≈ 0.15 A30-h guard) | provenance.json + REFERENCE_HASHES.json. The command is documented in `results/weight_anchor/REPORT.md`, not REGENERATION.md. GPU sampling, so a bitwise match is **unverified** |
| `ckpt/observability/traces_eval.pt` (cf46a1d3...) — **re-fit only** | yes, B_dit_analysis | 2* | yes / match | same | same |
| `ckpt/temporal/stage1_delta_heig8.pt` (52617834...) — **re-fit only** (lag diagnostics) | yes, A_dit_base | 2* | yes / match | `python -m boa_dit.temporal.stage1` (record step) | same caveat |

- **The controls themselves do not need the traces or Stage-1 cache.** Their frozen β^4 table is committed in `results/weight_anchor/coefficients.json`. The traces and Stage-1 cache are needed only to re-fit or re-audit those coefficients (`python -m boa_dit.weight_anchor.run provenance|lag|sim`).
- **Release-only check for the controls:** none of the required inputs is release-only. The only release-hosted inputs are `model.pt` and the carry, and both have a documented, hash-checked regeneration path. The carry is also in git.
- **C1 check dependency:** check C1 compares against the wa1 cached latents `ckpt/weight_anchor/live_dev_latents.pt`.
  - That file is **not in the release** and not in git (REPORT.md: "Live latents are cached in `ckpt/weight_anchor/` (not committed)").
  - It exists only on this VM's disk. Without it, C1 would need the wa1 `live_dev` run to be repeated first: `python -m boa_dit.weight_anchor.run live_dev`.

## Risks if the release were deleted
- **174 files (4.95 GB) would have no documented regeneration path** (class 4b). Producing code exists on parked branches, but its exact invocation is undocumented and was not tested:
  - PixArt static-A8 tables (`static_b_*`, `static_w16`), `dither_w16` and `design20_cache.pt`;
  - the PixArt stage-3/E1/E4/w16 image sets and the W4A4 phase images;
  - the DiT FP trajectory caches `fp_traj32_all` / `fp_traj_calib1000_all` and the `fp_eval_*` latents/pixels.
- **350 files (11.6 GB) could only be regenerated approximately** (class 3). This covers the E3C carries/FID features, the DNS/C-DNS traces, follow-up E1/E2/E3, the PixArt base tables and embeddings, and the e3c PixArt refit cache/branches/diagnostics.
  - Rebuilding costs roughly 25+ A30-h in total (REGENERATION §B: E3C ≈ 12 h, DNS ≈ 5 h, etc.).
  - The results would not be bitwise comparable, so committed numbers could not be re-derived exactly from the same inputs.
  - Two grids (`grid_P1_lr3e-5_restore_{o,qo}`) come from the PARTIAL interv2 run.
- **The only other copies are on this VM:** the files under `/home/boa-dit/ckpt` (all 536 match) and the staged parts in `/home/park_stage`. Both disappear if the VM is reclaimed.
  - The handoff docs (START_HERE: "none are hosted") assume regeneration and do not point at this release. A fresh VM would not find these files by following them.
- **Git (including the pushed `weight-anchor-controls` branch) is not a substitute.** It holds only reference hashes, plus the heig-8 carry and qdiff.yaml.
- **Low risk for the weight-anchor controls themselves:** every required input is upstream or regenerable with a hash check.
  - Re-fitting the coefficients would rely on GPU-regenerated traces and Stage-1 data whose bitwise match is unverified.
