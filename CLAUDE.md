# boa-dit: notes for Claude Code sessions

## Active task (since 2026-10-30, branch analytic-carry-pilot)
- **Active:** analytic carry correction + trajectory-response calibration (E1/E2). Protocol: `results/analytic_carry/PROTOCOL.md`; launcher `scripts/run_ac1.sh`; 4.0 GPU-h ceiling.
- Completed (do not rerun): the DiT activation-feedback + weight-correction pilot (`results/dit_joint/`), the DNS response pilot (branch dns-response-pilot).
- **Parked:** everything else (PixArt branch refit and its pilot, DNS / C-DNS and the deferred DNS proposal, activation-feedback extensions, weights-joint, follow-ups).
  Do not resume parked work unless the user explicitly asks. Its status and resume commands are in `handoff/PARKED_WORK.md`.

## Working rules carried over from the project
- Pre-register (dated protocol or amendment) before any result of the affected step. Keep confirmation sets untouched; consult `handoff/SEED_REGISTRY.md` before choosing seeds.
- Commit and push after every job. Long jobs report visible intermediate progress.
- Report negative, partial and interrupted results as such. Use "no demonstrated advantage under these conditions" instead of broad closure claims.
- Never use `pkill -f`/`pgrep -f` with a pattern that also matches the calling shell's command line: kill by explicit PID.
- Time-matched training runs must not share the GPU with other jobs.
- The user's email is for git identity only.
