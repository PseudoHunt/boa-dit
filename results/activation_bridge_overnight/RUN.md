# abn1 — how to run / resume

Worktree setup (fresh VM): `GIT_LFS_SKIP_SMUDGE=1 git worktree add /home/boa-dit-abn activation-bridge-overnight`, then link `ckpt`, `data`,
`external/DiT`, `external/boa` as the other worktrees, and place `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt` (regenerate with
`python -m boa_dit.dit_joint.run base`) and `results/stage0/ref_stats.npz` (`python -m boa_dit.eval_fid --ref`). Hashes: `frozen.json`.
Other prerequisites (regenerable, hash-checked at every stage): base model, smoothing, static_single8 tables (dit_joint), `ckpt/weight_anchor_w8/h8.pt`
(`python -m boa_dit.weight_anchor.w8 build`), joint-pilot carries `carry_{C_L,E_L}_u200.pt` (`python -m boa_dit.dit_joint.run train C_L|E_L`),
caches used only by the gate (aa1 `checks_dev_latents.pt`, br1 `dev_latents.pt`, wa8 `dev_latents.pt`).

## Commands (implemented)
```
bash scripts/abn1_queue.sh                         # whole queue: checks s1 s2 s3 s4 s5 s6 s7, commit/push per stage, report; resumable
python -m boa_dit.activation_bridge.run <stage>    # one stage: checks | s1 | s2 | s3 | s4 | s5 | s6 | s7
python scripts/make_activation_bridge.py           # REPORT.md + MORNING_STATUS.md from the JSONs
```
Resume points: a stage whose JSON has status complete/skipped/blocked is skipped by the queue; generation stages checkpoint every 32-image batch to
`ckpt/activation_bridge_overnight/<stage>_part.pt` and resume from it. Logs: `logs/abn1_<stage>.log` (local). Lock: `ckpt/activation_bridge_overnight/queue.lock`.
