# cfb1 — how to run / resume

From this worktree (branch `coupled-feedback-bridge-pilot`), with the artifacts in `handoff/REGENERATION.md` present:

    python -m boa_dit.feedback_bridge.cfb profile     # batch-32 memory and per-trajectory timing (frozen batch rule) and the plan
    bash scripts/cfb1_queue.sh                        # checks -> fit (27-point grid) -> val -> conf (only if the validation gate passes) -> diag -> report
    python scripts/make_coupled_feedback.py           # REPORT.md, STATUS.md, artifacts.json (no GPU)

The queue holds `ckpt/feedback_bridge_gpu.lock` (shared with rfb1). Generation checkpoints per (batch, configuration) in
`ckpt/coupled_feedback_bridge/<stage>_part.pt` and resumes; completed stages are skipped. Latents stay local (sha256 in artifacts.json).
