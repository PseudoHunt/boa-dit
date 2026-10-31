# rfb1 — how to run / resume

From the worktree root (branch `reference-feedback-bridge-pilot`), with the artifacts named in `handoff/REGENERATION.md` (common base, carry, tables,
smoothing; FP DiT) present:

    python -m boa_dit.feedback_bridge.rfb profile     # R-full memory and timing at batch 32/16/8 (sets the frozen batch by the protocol rule)
    bash scripts/rfb1_queue.sh                        # checks -> dev -> conf (only if the dev gate passes) -> diag -> report; resumable
    python scripts/make_reference_feedback.py         # REPORT.md, STATUS.md, artifacts.json (no GPU)

The queue holds `ckpt/feedback_bridge_gpu.lock` (shared with the cfb1 queue, so the two pilots never share the GPU). Each stage resumes from its
per-batch checkpoint `ckpt/reference_feedback_bridge/<stage>_part.pt`; completed stages are skipped. Latents stay local (sha256 in artifacts.json).
