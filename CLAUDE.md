# boa-dit: notes for Claude Code sessions

## Active task (2026-10-31): BOTH pilots COMPLETE, nothing running. rfb1 (reference-feedback-bridge-pilot, /home/boa-dit-abn): 1.324 of 3.0 A30-h, primary RDB/GB 0.988 inconclusive. cfb1 (coupled-feedback-bridge-pilot, /home/boa-dit-cfb): 1.158 of 3.0 A30-h, validation gate failed (J/B 1.262), confirmation 121000–121255 unopened. No further runs without user approval.
- Completed 2026-10-30: overnight abn1 (feedback × bridge, W8 anchors, K8, local refit + bridge, diagnostics, FID-2k if it fits, timing). Protocol
  `results/activation_bridge_overnight/PROTOCOL.md`; cap 6.0 A30-h + 8 h wall (tag abn1); queue `bash scripts/abn1_queue.sh` (resumable, flock); see RUN.md.
- Completed 2026-10-30: anchor-drift ad1 (anchor-drift-pilot b42f065; drift vs local inconclusive, 1.518 A30-h).
- **Completed 2026-10-30 (no further runs):** retrospective anchor-bridge correction (br1), branch anchor-bridge-pilot; `results/anchor_bridge/REPORT.md`;
  0.302 of 0.5 A30-h; confirmed useful (CBlam/C_last49 0.292, CB1/C 0.213 on 98100–98163). AC + bridge not run (needs its own registration).
- Completed 2026-10-30: local-refresh ler1 (local-error-refresh-pilot, stopped at E1 headroom rule, 0.331 A30-h); aa1 (activation-anchor-pilot, 0.532 A30-h).
- **Parked by user decision (2026-10-30, not a failed read):** shared/nested weight quantization (ws1, weight-anchor-shared; complete through confirmation). See `handoff/PARKED_WORK.md` §7.
- Worktrees: create with `GIT_LFS_SKIP_SMUDGE=1 git worktree add ...` (the LFS smudge hangs otherwise).
- Do NOT delete the GitHub release park-assets-2026-10-29 (user instruction 2026-10-30).
- Completed (do not rerun): shared-anchor ws1 (weight-anchor-shared, parked), W8 anchors wa8 (weight-anchor-w8), weight-anchor controls wac1 (weight-anchor-controls), Nunchaku SDXL feasibility nk1 (nunchaku-sdxl-feasibility, no margin), weight-anchor pilot wa1 (weight-anchor-pilot), quantizer observer pilot (quantizer-observer-pilot, gate failed), joint pilot (`results/dit_joint/`), DNS response pilot (dns-response-pilot), analytic carry (analytic-carry-pilot, parked method).
- **Parked:** everything else (PixArt branch refit and its pilot, DNS / C-DNS and the deferred DNS proposal, activation-feedback extensions, weights-joint, follow-ups).
  Do not resume parked work unless the user explicitly asks. Its status and resume commands are in `handoff/PARKED_WORK.md`.

## Working rules carried over from the project
- Pre-register (dated protocol or amendment) before any result of the affected step. Keep confirmation sets untouched; consult `handoff/SEED_REGISTRY.md` before choosing seeds.
- Commit and push after every job. Long jobs report visible intermediate progress.
- Report negative, partial and interrupted results as such. Use "no demonstrated advantage under these conditions" instead of broad closure claims.
- Never use `pkill -f`/`pgrep -f` with a pattern that also matches the calling shell's command line: kill by explicit PID.
- Time-matched training runs must not share the GPU with other jobs.
- The user's email is for git identity only.
