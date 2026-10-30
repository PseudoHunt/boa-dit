# Parked work (2026-10-29). Nothing here runs unless the user explicitly resumes it.

Parked by user instruction: the DiT joint-feedback pilot (results/dit_joint/PROTOCOL.md) is the only active work.
All queues were stopped at ≈ 13:30 UTC (launchers before children); e3-confirm has one correction commit after its park tag (stop-time fix). Interrupted jobs are marked INTERRUPTED/PARTIAL, never complete or negative.
No generated or trained artifact is hosted. To resume a thread, regenerate its inputs with `handoff/REGENERATION.md` §B (the old VM's files and hashes are listed in REFERENCE_HASHES.json). Lines below that say "Assets: <group>" name the old archive grouping, kept for reference only.

## Branch heads at parking (immutable tags)
| branch | tag | commit |
|---|---|---|
| temporal-oracles | park/2026-10-29/temporal-oracles | 216e4e6d0ba01ceaba245e1b6562fe060f10434b |
| e3-confirm | park/2026-10-29/e3-confirm | 21f0c552db8204a16b08f92efb4932caa7ea4b99 |
| dns-cov | park/2026-10-29/dns-cov | b06404f5bd721b1a853c206475dccc9ac0aaacb1 |
| followup-weights | park/2026-10-29/followup-weights | c0edc15de4029783c3d2196b43ffa0e210b8dd64 |
| followup-act | park/2026-10-29/followup-act | dae47113bc8f28d47aefdc27071969b644ee4410 |
| weights-joint | park/2026-10-29/weights-joint | 1244c6fc0270a7cb3378c722c80e9914e0ebbd16 |
| main | park/2026-10-29/main | 116460b713c3919f2508b6b326f4f230f7f0d679 |

Uncommitted state left untouched in the old main checkout (temporal-oracles worktree): deleted `results/attn_share/artifacts/V1_maps_{cond,uncond}_s0-19.pt` (LFS)
and a modified `results/w4a4/quant/hsvd_r4_b0.5_kv16_u2_rs.json`. Both predate this session and are not committed; the committed versions are at the tag.

**Parked code paths.** The parked branches still contain absolute paths: `/home/boa-dit*/...` (ckpt symlinks, gpu_hours CSV globs), `/home/jl_fs/pixart_sigma_1024`,
`/home/jl_fs/hf`, `/root/ext/{deepcompressor,mjhq}`, and the ImageReward venv `$IRPY`. To resume one, either recreate those paths as symlinks into the restored
`data/` tree, or cherry-pick the path-configuration commit from `dit-joint-feedback` (env vars BOA_HF_HOME, BOA_PIXART_DIR, BOA_EXT_DIR, BOA_EXTRA_CSVS).

## 1. PixArt rank-4 branch refit (e3-confirm, `results/e3c/`)
- **Done.**
  - Part 2 design: every lr fails the endpoint guard (design endpoint 0.206 for P0).
    - P1: 0.36 / 0.91 / 1.28 / 1.30 at lr 1e-5 / 3e-5 / 1e-4 / 3e-4.
    - P2 at lr 1e-4: 0.59 (λ 0.5) / 0.79 (λ 2). P2 at lr 1e-5: 0.216 (λ 0.5) / 0.209 (λ 2).
    - Nothing promoted (`final()` gated on per-arm guards or a dated `promote.json`).
  - Diagnostics v1 (norm, branch, interv, alpha) and v2 (`steps`, `branch2`), in `results/e3c/pixart/diag.json`.
- **Latest findings (8 design prompts):**
  - Low-rate P1 lowers prediction error on cached **and** own states (0.982 → 0.881), but its endpoint worsens (0.206 → 0.362).
  - Low-rate P2 lowers held-out 5-step rollout error (0.872 → 0.784 / 0.808) with no clear endpoint change.
  - The rollout gradient concentrates at start steps 10–11.
  - Functional drift is dominated by `attn2.to_q` (8 prompts × 5 steps).
  - `attn2.to_out` transplants never repair the endpoint, so the updates interact across layers.
  - α-interpolation: P1's damage grows smoothly with α. P2 at λ 0.5, lr 1e-4, α 0.1 gave 0.188 (exploratory, selected, unconfirmed).
- **PARTIAL:** `interv2` (q / q+o transplants with endpoint detail).
  - P0 and P1_lr1e-5 rows are in `results/e3c/logs/f5_px_diag_v2.log.txt`; P1_lr3e-5 was partway (grids exist).
  - P1_lr1e-5 `restore_q`: mean 2.09, cos 0.54, RMS ratio 1.78, i.e. a magnitude increase, not only a direction change.
  - Rerun: `python -m boa_dit.e3c.px_diag interv2` (≈ 1.7 A30-h).
- **NOT STARTED, pending in protocol order** (each command is resumable by rerunning):
  1. `python -m boa_dit.e3c.px_screen` (lr 1e-4 matched screen with the drift arm);
  2. `python -m boa_dit.e3c.px_diag alpha2` (paired α + reproducibility repeats);
  3. `python -m boa_dit.e3c.px_screen --lr 1e-5 --tag lr1e-5 --cks 10,25,50,100,200 --no-drift`;
  4. `python -m boa_dit.e3c.px_fresh` (fresh dev mjhq1k[400:432], **unopened**).
  - Then the conditional trajectory-anchored target (registered, unimplemented; reserve dev set mjhq1k[432:464]).
- **Held:** the Part 3 pilot (user spec, K = 3, 0.4 h/arm) never ran; its single check run was aborted and discarded.
- **Assets:**
  - C_pixart_base (base tables, embeddings, prompt files) and D_pixart_analysis (training cache, branch files, diag endpoints, grids, images);
  - the full refit checkpoints are rebuilt with `--rebuild-pixart-refits`;
  - the PixArt upstream comes from `--upstream pixart`.
- **Dependencies:** ImageReward env (ENVIRONMENT.md), the absolute-path note above.

## 2. DiT carry confirmation E3C (e3-confirm, `results/e3c/REPORT.md`): COMPLETE
- Rank-8 rollout refit (C2) vs H16 (rank 16): FID −2.25 [−3.03, −1.67]; paired fidelity tied.
- C2 does not beat update-matched local refitting (C1u); C1 degrades with training length, while C2 saturates.
- M adds nothing.
- Assets: B_dit_analysis.

## 3. DNS reproduction and C-DNS (dns-cov, `results/dns/REPORT.md`): COMPLETE
- Faithful DNS FID −0.39 [−0.61, −0.21]; Q-Drift −0.02.
- C-DNS: the full covariance model passes its predictive gate (0.18 vs 2.56) but the sampler **fails** (FID +16.3). The covariance-multiplier construction failed; it is not relabelled.
- The ablation beats DNS by −0.38 on FID, with slightly worse paired latent error.
- **Deferred proposal (UNRUN, PARKED):**
  - ablate DNS components;
  - include the signed marginal discrepancy Var(d) + 2·Cov(x_FP, d);
  - calibrate on corrected trajectories;
  - test small expansion/contraction;
  - learn a few correction coefficients from their measured downstream response.
- Assets: B_dit_analysis (dns/*).

## 4. Activation feedback (temporal-oracles, `results/temporal/`): COMPLETE for the c3 allocation
- DiT static-A8 DCT-4 feedback: D3 FID-2k 26.18.
- PixArt DCT-16 trade-off read FAILED; ImageReward sign unconfirmed; §5 memory engineering dropped (`mem.py` untested).
- Assets: A_dit_base (static tables, smoothing), D_pixart_analysis (images, tables); regenerable histograms listed in ASSETS.json.

## 5. Follow-up E1–E3 (followup-weights / followup-act, `results/followup/REPORT.md`): COMPLETE
- Assets: B_dit_analysis (followup/e1, e3), D_pixart_analysis (followup/e2 images).

## 6. weights-joint / weights-scale (weights-joint branch): COMPLETE (results in git)
- Caches are regenerable (ASSETS.json).

## 7. Shared / nested weight quantization (weight-anchor-shared, `results/weight_anchor_shared/REPORT.md`): COMPLETE, PARKED BY USER DECISION (2026-10-30)
- Parked by user decision in favour of the activation-anchor pilot; **not** a failed read and not a closure of nested quantization.
- State at parking: every pre-registered stage ran (selftest, build+pack, checks, dev, A-RTN dev control, traces, freeze, confirmation); head 80db0aa.
  No job was running or queued when the stop instruction arrived. GPU 0.367 of 2.0 A30-h (ledger tag ws1).
- Result: reuse useful (R_shared vs A_shared −54%); R_shared 1.95× R8 (degradation beyond the 1.10 / 0.01 margins); shared artifact 223.0 MB (23.9%) smaller.
- Artifacts (not hosted; regenerate): `ckpt/weight_anchor_shared/{q4,h8,pair,A_gptq,A_rtn,B_gptq}.safetensors` (sha256 in `results/weight_anchor_shared/storage.json`),
  `build.pt`, `conf_latents.pt`, `checks_dev_latents_*.pt`; logs `logs/ws1_*.log` in the /home/boa-dit-ws worktree.
- Resume: `GIT_LFS_SKIP_SMUDGE=1 git worktree add <dir> weight-anchor-shared`; symlink external/DiT, external/boa, ckpt, data; then
  `python -m boa_dit.weight_anchor.shared selftest|build|checks|dev|conf` (prerequisites: `temporal.rebuild`, `weight_anchor.w8 build`). See `implementation.md`.
- Not started (would need a new protocol and budget): joint (MatGPTQ-style) nested construction with its own low-model baseline.
