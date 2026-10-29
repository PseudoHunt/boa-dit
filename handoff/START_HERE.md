# START HERE (fresh VM handoff, 2026-10-29)

**Current objective:** implement and run only the DiT activation-feedback + weight-correction pilot, `results/dit_joint/PROTOCOL.md`.
Everything else is **parked** (`handoff/PARKED_WORK.md`). Nothing parked runs unless the user explicitly asks.

- Repository: https://github.com/PseudoHunt/boa-dit. Branch **`dit-joint-feedback`**. The handoff commit is identified by tag **`handoff/2026-10-29`** (pushed with this file).
- Parked branch heads: immutable tags `park/2026-10-29/<branch>` (table in PARKED_WORK.md).
- Non-Git assets: release **`park-assets-2026-10-29`** (tar parts), manifest `handoff/ASSETS.json`, checksums `handoff/SHA256SUMS`.
- Environment: `handoff/ENVIRONMENT.md`. Seeds and prompts: `handoff/SEED_REGISTRY.md`. Code provenance: `handoff/INTEGRATION.md`.

## Restore order
1. **System tools** (Ubuntu/Debian GPU VM with a working NVIDIA driver; check `nvidia-smi`):
   ```
   sudo apt-get update && sudo apt-get install -y git curl ca-certificates python3 python3-venv tmux
   # GitHub CLI (needed to download release assets):
   (type -p wget >/dev/null || sudo apt-get install -y wget) && sudo mkdir -p -m 755 /etc/apt/keyrings \
     && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg >/dev/null \
     && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null \
     && sudo apt-get update && sudo apt-get install -y gh
   gh auth login          # an account authorized for PseudoHunt/boa-dit (HTTPS + browser/device login; no token in URLs)
   ```
2. **Clone and bootstrap** (≈ 10–20 min, plus downloads: about 2.9 GB of archive parts and 2.6 GB of DiT weights for the pilot):
   ```
   git clone --branch dit-joint-feedback https://github.com/PseudoHunt/boa-dit.git && cd boa-dit
   bash scripts/bootstrap_vm.sh          # venv + locked stack + submodules + A_dit_base + DiT/VAE upstream + smoke check
   . .venv/bin/activate
   ```
   Defaults: data root `./data` (override with `BOA_DATA`), HF cache `./data/hf` (override with `BOA_HF_HOME`).
   `bootstrap_vm.sh` ends with `BOOTSTRAP_OK` after `scripts/verify_restore.py --smoke` passes.
3. **Optional, for analysis or parked work only** (not needed for the pilot):
   `python scripts/restore_assets.py --groups B_dit_analysis,C_pixart_base,D_pixart_analysis --upstream pixart --rebuild-pixart-refits`
   (≈ 17 GB of parts plus the ≈ 12 GB PixArt upstream). On an A30 with the ENVIRONMENT.md stack, `python scripts/verify_restore.py --bitwise` checks bitwise agreement with archived latents.
   On different hardware, do **not** reuse old baselines: the pilot's protocol re-measures its own controls on the new machine.

## Exact next command (after BOOTSTRAP_OK)
The pilot module does not exist yet. Implement `boa_dit/dit_joint/run.py` with the stages listed in `scripts/run_dit_joint.sh`, following PROTOCOL.md §D and the §C5 checks. Then run:
```
scripts/run_dit_joint.sh        # resumable; each stage commits/pushes small results; checkpoints are backed up by scripts/backup_checkpoints.py
```

## Rules for the agent on the new VM
- Read `CLAUDE.md`, this file and `results/dit_joint/PROTOCOL.md`. Keep every parked experiment parked.
- Implement and execute only the pilot. Proceed through the correctness checks and the bounded pilot without asking for routine reconfirmation.
- Never tune on an old confirmation set. The sealed pilot set (61000–61127) is opened once, after the selections are frozen.
- Record implementation decisions and amendments in PROTOCOL.md, dated, before the affected step runs.
- Commit/push code and small results at milestones and back up new checkpoints remotely. Stop after the pilot report.
