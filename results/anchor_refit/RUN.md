# dar1 — run / regenerate (worktree /home/boa-dit-dar, branch dit-anchor-refit)
Local-only dependencies: `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` (b4a537b0…, see handoff/REGENERATION.md), `results/stage0/ref_stats.npz`
(provenance check only), `ckpt/dit_joint/{static_single8.pt, norm.pt, carry_C_L_u200.pt}`, abn1 latents `ckpt/activation_bridge_overnight/s1_latents.pt`,
`s4_latents.pt` (checks only).

    python -m boa_dit.anchor_refit.run cache      # B0 corrected-trajectory anchor states (TRAIN, DIAG)
    python -m boa_dit.anchor_refit.run norm       # frozen den[a]
    python -m boa_dit.anchor_refit.run checks     # gate (then: fdgrad, and fd64 — the decisive fp64 FD check, see PROTOCOL amendments)
    python -m boa_dit.anchor_refit.run fd64
    python -m boa_dit.anchor_refit.run profile    # N = 330, checkpoints 82/165/330
    python -m boa_dit.anchor_refit.run train_ba
    python -m boa_dit.anchor_refit.run train_bl   # update-matched + time-matched (BL-tm)
    python -m boa_dit.anchor_refit.run dev
    python -m boa_dit.anchor_refit.run diag
    python -m boa_dit.anchor_refit.run conf
Queue helper: `bash scripts/dar1_queue.sh <pid-to-wait-for> <stages...>` (commit/push per stage).
