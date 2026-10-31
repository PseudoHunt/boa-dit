# bar1 — run / regenerate (worktree /home/boa-dit-bar, branch ba-replication)
Local-only dependencies:
- `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` (b4a537b0…, see handoff/REGENERATION.md);
- `results/stage0/ref_stats.npz`;
- shared `ckpt/` (symlink to /home/boa-dit/ckpt) with `dit_joint/`, `anchor_refit/` (dar1 pilot checkpoints, anchor_cache.pt, den.pt, conf/dev latents)
  and `activation_bridge_overnight/`;
- `external/DiT`, `external/boa` symlinks to /home/boa-dit/external.

    python -m boa_dit.ba_replication.audit audit        # 1A (CPU)
    python -m boa_dit.ba_replication.audit power        # 1B (CPU) -> N = 256
    python -m boa_dit.ba_replication.mech mech          # step 2 (GPU, ~0.26 h)
    python -m boa_dit.ba_replication.mech analyze       # step 2 analysis (CPU)
    python -m boa_dit.ba_replication.rep repchecks
    python -m boa_dit.ba_replication.rep train 1 ; python -m boa_dit.ba_replication.rep dev 1
    python -m boa_dit.ba_replication.rep train 2 ; python -m boa_dit.ba_replication.rep dev 2
    python -m boa_dit.ba_replication.rep plan           # pilot-arm budget decision
    python -m boa_dit.ba_replication.rep eval           # resumable per batch (ckpt/ba_replication/eval_part.pt); then analysis
    python -m boa_dit.ba_replication.rep analyze        # CPU re-analysis of eval.json

Queue helper: `bash scripts/bar1_queue.sh <pid-to-wait-for> mod:stage[:arg] ...` (commit/push per step). Training resumes from
`ckpt/ba_replication/resume_{BA,BL}_s{1,2}.pt`. Budget: `python scripts/h1031b_budget.py`.
