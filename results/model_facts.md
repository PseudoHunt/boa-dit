# Model facts (verified against code before Stage 1)

## DiT-XL/2 (external/DiT @ ed81ce2)
All handoff claims confirmed unless marked.

| Fact | Value | Where |
| --- | --- | --- |
| depth / hidden / heads / head dim | 28 / 1152 / 16 / 72 | `models.py:DiT_XL_2`, verified at runtime |
| patch / tokens / latent | 2 / 256 / 4×32×32 | runtime |
| output channels | 8 (`learn_sigma=True`: 4 eps + 4 variance) | runtime |
| `DiTBlock.forward` | `shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = adaLN_modulation(c).chunk(6, dim=1)`; `x += gate_msa.unsqueeze(1) * attn(modulate(norm1(x), …))`; same for MLP | `models.py:118-122` ✔ |
| `adaLN_modulation` | `Sequential(SiLU, Linear(1152, 6912))` | `models.py:113-116` ✔ |
| `modulate` | `x * (1 + scale) + shift` | `models.py:19-20` |
| attention | timm 1.0.30 `Attention`, `qkv = Linear(1152, 3456, bias=True)` fused, `proj = Linear(1152,1152)`; `fused_attn=True` (SDPA) | runtime ✔ |
| MLP | `fc1 = Linear(1152, 4608)`, `GELU(approximate="tanh")`, `fc2 = Linear(4608, 1152)` | runtime ✔ |
| CFG | `forward_with_cfg` applies CFG to the first **3** eps channels only (comment at `models.py:258-262`); the 4th eps channel and the variance channels come from the cond half. **Kept as-is** for comparability; reproduced exactly in `boa_dit/sample.py`. Branch weights s² / (s−1)² assume CFG on all channels; ch. 3 deviation is second order. | `models.py:250-266` ✔ |
| null class | 1000 (`y_embedder` has 1001 rows, CFG dropout) | ✔ |
| VAE scale | 0.18215 (`sample.py` decode) | ✔ |
| sampler | `create_diffusion("ddim50")` → `space_timesteps` "ddimN" branch (`respace.py:32`), timestep_map = 0,20,…,980; `ddim_sample(…, eta=0.0)` | ✔ |
| variance channels | `p_mean_variance` splits them off (`gaussian_diffusion.py:285-287`); DDIM uses only `pred_xstart` from eps → ignored at eta = 0 | ✔ |
| **clip_denoised** | `ddim_sample` defaults to `clip_denoised=True`, which clips latent x̂₀ to [−1,1]. The official `sample.py` passes `clip_denoised=False`; **we pass False.** (not in the handoff) | `gaussian_diffusion.py:513-520` |

Step-index convention: `i = 0..49` in sampling order; `i = 0` ↔ t = 980 (noisiest), `i = 49` ↔ t = 0.

### Parameter counts (675,129,632 total)

| Layer type (×28 blocks) | Params | Share |
| --- | ---: | ---: |
| adaLN_modulation (W+b) | 223,147,008 | 33.05% |
| mlp.fc1 (W) | 148,635,648 | 22.02% |
| mlp.fc2 (W) | 148,635,648 | 22.02% |
| attn.qkv (W) | 111,476,736 | 16.51% |
| attn.proj (W) | 37,158,912 | 5.50% |
| block biases (qkv, proj, fc1, fc2) | 290,304 | 0.04% |
| final_layer | 2,693,408 | 0.40% |
| t_embedder / y_embedder / x_embedder / pos_embed | 1,624,320 / 1,153,152 / 19,584 / 294,912 | 0.46% |

adaLN is one third of the model; the four quantized block linears are 66.0%.

## BoA fork (external/boa @ 91813d5)

| Item | Finding | file:line |
| --- | --- | --- |
| H accumulation | `CovarianceCollector.compute_cov_in_batch` forward hook on each linear; running mean `XXT = (2/n) Σ x xᵀ` | `utils/hessian_utils.py:13-19, 47-55`; wired in `quantize.py:291+` (`compute_Hessian`) |
| solver entry | `BoA(layer, opts, hyperparams)`; set `.quantizer` (a configured `MinMaxQuantizer`) and `.H_col`; `.quant()` writes fake-quant weights into `layer.weight` | `quantizers/boa.py:10-69`; driven at `quantize.py:73-76, 119` |
| layer-output (GPTQ) path | `H_row is None` → `BoA.gptq(W, H_col, scale, zero)` | `quantizers/boa.py:48-49, 72-90` |
| vectorized over rows? | yes: column loop, all rows updated together (`W[..., idx_col:] -= err @ U_col[...]`); no lazy-batch blocking | `quantizers/boa.py:76-85` |
| quant grid | per-channel (per output row); asymmetric unless `--w_sym` (default asym); `qparam_comput="Hessian"` default → grid search (100 shrinks × floor/ceil) minimizing `qᵀ H_col q` per row | `utils/process_args.py:28,31`; `utils/quant_utils.py:13-54` |
| damping | 1% of mean diagonal (`percdamp=.01`), applied in `filter_dead_neuron`; dead input columns get diagonal `replace = 1/seqlen` | `utils/quant_utils.py:57-87`; `utils/process_args.py:90` |
| act_order | off by default (`--act_order_col`, `--act_order_row` are store_true) | `utils/process_args.py:33-34` |
| groupsize | not supported in the `find_params_H` path (per-row only) | `quantizers/minmax.py:182-186` |
| TF32 | the module disables TF32 on import | `quantizers/boa.py:7-8` |

**Wrapping plan (no solver change):** the H_col for a row subset is passed by building a temporary `nn.Linear` whose
weight is `W[rows]`, running `BoA(...).quant()` on it with that subset's H_col, and writing the result back into
`W[rows]`. Because the GPTQ path treats rows independently (per-row scale/zero, per-row error propagation), this is
exactly equivalent to a per-row-group Hessian. `replace` = 1/256 (one DiT image = 256 tokens, the analogue of the
fork's 1/seqlen).
