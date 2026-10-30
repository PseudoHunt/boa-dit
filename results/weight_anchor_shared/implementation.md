# ws1 implementation notes

Code: `boa_dit/weight_anchor/shared.py`. Base: weight-anchor-w8 dc76c1e. Artifacts: `ckpt/weight_anchor_shared/` (not in Git; regenerate).

## Constructions
- **Q4 (frozen):** heig-8 weights from `ckpt/ckpt_deflate_heig_r8_w4/model.pt`; c4 = round(Q4/s4 + z4) with s4, z4 from `ckpt/temporal/heig8_layers.pt`;
  s4·(c4 − z4) == Q4 asserted bitwise; δ4 = stored heig-8 carry (re-derived and asserted equal).
- **A-GPTQ:** `qutils.boa_gptq_rows` (unchanged) on D = W − Q4, bits 4, saved P H P, replace 1/256; grid captured from BoA's quantizer
  (`w8.capture_grid`); QA = s4(c4 − z4) + sr(cr − zr) in fp32 (asserted == Q4 + D̂). Solver matrix logged per layer (`solver_matrix` in build.json:
  saved-H vs effective-H diagonal mean, added damping, dead columns).
- **A-RTN:** `qutils.rtn_weight` rule on CPU (per-row min/max including 0, no search).
- **B-GPTQ:** `gptq_cell` = BoA's GPTQ column loop (`weights_joint.joint.prep` → Wp, U), rounding c8 = clamp(round(w/s8) + z8, 16c4, 16c4 + 15),
  q = s8(c8 − z8), error feedback (w − q)/U_jj on the full post-clip error. s8 = s4/16 (exact), z8 = 16 z4 + 8 (exact integers in fp32; c8 ≤ 255).
  Validated by `selftest` (enumeration, ties, unblocked reference) and by reproducing Q4 on layer 0 with the full 4-bit grid.
- **Carry:** `deflate.carry.make_carry(W, QH, V_fp32)` for every candidate (fp16 storage, as all evaluated carries).
- **adaLN / other:** adaLN W8 codes recovered on CPU with the RTN rule (bitwise; the same computation on GPU does not reproduce the stored
  weights, so CPU is the reference); 151 other tensors (3,615,008 elements) fp32, identical to FP.

## Artifact format (safetensors, header key `ws1` = JSON metadata incl. per-tensor sha256 and layout)
Per layer `b{l}.{attn.qkv|attn.proj|mlp.fc1|mlp.fc2}`: `V` (fp16), `base_plane` (uint8 [rows, d_in/2], low nibble = even column), `s4` (fp32),
`z4` (uint8), `delta4` (fp16); shared A adds `refine_plane`, `sr` (fp32), `zr` (uint8), `deltaH`; shared B adds `refine_plane`, `deltaH`;
pair adds `h8_codes` (uint8 [rows, d_in]), `s8` (fp32), `z8` (uint8), `delta8`. `aux.adaln.{l}.{codes,scale,zero}`, `aux.other.<state-dict key>`.
Unpack: `shared.unpack(kind, 'low'|'high')` → (fp32 state dict, carry dict); `shared.model_from(kind, which, device)` builds a model (strict load +
bitwise verification) and installs the carry. Checksums are verified on every unpack.

## Commands
```
GIT_LFS_SKIP_SMUDGE=1 git worktree add /home/boa-dit-ws weight-anchor-shared     # then symlink external/DiT, external/boa, ckpt, data as other worktrees
python -m boa_dit.weight_anchor.shared selftest   # CPU
python -m boa_dit.weight_anchor.shared build      # constructions + pack + round-trip + storage (needs heig8_layers.pt, heig-8 model + carry, wa8 h8.pt)
python -m boa_dit.weight_anchor.shared checks
python -m boa_dit.weight_anchor.shared dev | dev_rtn | traces
python -m boa_dit.weight_anchor.shared conf       # requires frozen.json
python scripts/make_weight_anchor_shared.py
```
Prerequisites regenerate as: `python -m boa_dit.temporal.rebuild` (heig-8 model, layer cache, carry) and `python -m boa_dit.weight_anchor.w8 build` (h8.pt);
compare file hashes with `storage.json` / `provenance.json`.

## Resume points for later kernel work
- Packed inputs: `ckpt/weight_anchor_shared/{A_gptq,B_gptq,pair,q4,h8}.safetensors` (layout above). A's high layer = Q4 GEMM + refinement GEMM
  (two 4-bit products, or on-the-fly reconstruction) + carry; B's high layer is one 8-bit-grid GEMM with c8 = (base << 4) | refine and a derived
  scale/zero; neither matches arbitrary INT8 kernels or bit slicing without the prescribed s8/z8.
- Comparators any speed study must include: H8-50 and H_shared-50 at matched latency, not only Q4/A_shared.
