# boa-dit: notes for Claude Code sessions

## Active task (since 2026-10-30, branch weight-anchor-controls)
- **Active:** weight-anchor controls (wac1). Protocol `results/weight_anchor_controls/PROTOCOL.md`; 0.5 A30-h hard cap; stage CLI `python -m boa_dit.weight_anchor.controls <stage>`.
- Do NOT delete the GitHub release park-assets-2026-10-29 (user instruction 2026-10-30).
- Completed (do not rerun): weight-anchor pilot wa1 (weight-anchor-pilot), quantizer observer pilot (quantizer-observer-pilot, gate failed), joint pilot (`results/dit_joint/`), DNS response pilot (dns-response-pilot), analytic carry (analytic-carry-pilot, parked method).
- **Parked:** everything else (PixArt branch refit and its pilot, DNS / C-DNS and the deferred DNS proposal, activation-feedback extensions, weights-joint, follow-ups).
  Do not resume parked work unless the user explicitly asks. Its status and resume commands are in `handoff/PARKED_WORK.md`.

## Working rules carried over from the project
- Pre-register (dated protocol or amendment) before any result of the affected step. Keep confirmation sets untouched; consult `handoff/SEED_REGISTRY.md` before choosing seeds.
- Commit and push after every job. Long jobs report visible intermediate progress.
- Report negative, partial and interrupted results as such. Use "no demonstrated advantage under these conditions" instead of broad closure claims.
- Never use `pkill -f`/`pgrep -f` with a pattern that also matches the calling shell's command line: kill by explicit PID.
- Time-matched training runs must not share the GPU with other jobs.
- The user's email is for git identity only.
