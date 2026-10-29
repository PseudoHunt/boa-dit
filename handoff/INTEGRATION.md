# Code integration record (branch dit-joint-feedback)

Base: `park/2026-10-29/e3-confirm` = 21f0c552db8204a16b08f92efb4932caa7ea4b99. It contains the rank-8 carry training code (`boa_dit/followup/carry_opt.py`, `boa_dit/e3c/`)
and the deflate/heig builder.

Selectively taken from `park/2026-10-29/temporal-oracles` = 216e4e6d0ba01ceaba245e1b6562fe060f10434b. These are the authoritative static-A8 / DCT-4 feedback implementation:
- `boa_dit/temporal/dit_act.py`: `Quantize` (static A8 tables, smoothing, `fb_step`, `token_basis("dct4")`, guard, clamp, state storage);
- `boa_dit/temporal/c3.py`: the c3 budget/runner helpers used by the DCT-4 FID run (D3);
- `boa_dit/temporal/fid_act.py`, `boa_dit/temporal/mem.py`, `boa_dit/temporal/dist.py`: activation FID / memory / distribution runners (reference only).

Other changes on this branch (2026-10-29, at parking):
- absolute paths made configurable (`BOA_HF_HOME`, `BOA_PIXART_DIR`, `BOA_EXT_DIR`, `BOA_EXTRA_CSVS`, `BOA_FOLLOWUP_ACT_RESULTS`; repo-relative defaults);
- handoff documents; `scripts/fetch_upstream.py`, `scripts/regenerate_base.sh`, `scripts/verify_restore.py`, `scripts/bootstrap_vm.sh`; `assets_static/deepcompressor_qdiff.yaml` (BOA_QDIFF).

No research branch was merged wholesale. The joint pilot's module (`boa_dit/dit_joint/`) is new; record its sources here as it is written.
