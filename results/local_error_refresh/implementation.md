# ler1 implementation notes

Code `boa_dit/local_refresh/run.py` (stage tag ler1); report `scripts/make_local_refresh.py`; C4 diagnosis `scripts/ler1_c4_diagnosis.py`.
Latents `ckpt/local_error_refresh/` (not in Git; regenerate). Branch parent: activation-anchor-pilot d1eab46.

- `LocalQuant(dit_joint.JointQuant)`: parent arithmetic unchanged; `_fn` records h (JointQuant input, after the smoothing pre-hook) and q (its output) at
  `cap_keys`; `zero_state` drops `state_in[key]` before each call so the parent builds a zero state (G); `acct` adds the quantizer-observer guarded-feedback
  accounting (energies/rates, copied arithmetic from quantizer-observer-pilot `InstrQuant`).
- `Refresh`: forward hooks registered after the carry hooks at the selected linears. `measure(k, h)` = F.linear(h, W) − [F.linear(h, Q) + (h V) δᵀ] in fp32 with
  autocast disabled, no module call. At i mod K = 0 the result is stored fp16; every step out ← (out.float() + b.float()).to(out.dtype). Options: scale (0 for
  C6), perm (shuffle control), ref_self (M = W test), trace (E1 eW statistics with an 8-step ring buffer), diag (a / v / δb accounting), debug (C5 capture).
- Reference weights: FP weight × s at fc1/fc2 (same expression as `dit_joint.apply_smoothing`), FP weight elsewhere; cloned plain tensors.
- MP: separate cheap instance, selected weights := W, their carry hooks removed.
- Sampler: aa1 loop (DDIM-50, CFG 1.5 via `weight_anchor.run.guided`); resume state = latent, step, counts, RNG, feedback state, held b, last-refresh index, refresh counts.

## Commands
```
python -m boa_dit.local_refresh.run checks    # C1-C10 (0.18 h on an A30)
python -m boa_dit.local_refresh.run screen    # E0 (selection.json)
python -m boa_dit.local_refresh.run e1        # traces + headroom (trace.json, dev_e1.json)
python -m boa_dit.local_refresh.run e2|diag|conf|cost   # not run: E1 headroom rule failed
python scripts/make_local_refresh.py
```
Prerequisites: `ckpt/ckpt_deflate_heig_r8_w4_s/model.pt`, `results/deflate/quant/deflate_heig_r8_w4_s/carry.pt`, `ckpt/temporal/dit_smooth_s.pt`,
`ckpt/dit_joint/static_single8.pt`, `ckpt/dit_joint/eval_latents.pt` (C1 only); hashes in `provenance.json`; regeneration via `python -m boa_dit.dit_joint.run base`.
