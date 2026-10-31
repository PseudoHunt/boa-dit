# snk1 — how to run

    # models (pinned; public, non-gated)
    /home/nk_env/bin/python -c "from huggingface_hub import hf_hub_download as d, snapshot_download as s; \
      d('nunchaku-ai/nunchaku-sana','svdq-int4_r32-sana1.6b.safetensors',revision='437e41b979ec875f63c18e67d55373aa5230c308'); \
      s('Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers',revision='e2b3c0cbffebcd09d83805e88b9f5f106afc74ac')"
    P=/home/nk_env/bin/python
    $P scripts/sana_a30.py env        # environment / hashes / prompts manifest
    $P scripts/sana_a30.py smoke      # official example on both models, repeatability, router vs direct
    $P scripts/sana_a30_det.py; $P scripts/sana_a30_det2.py     # INT4 nondeterminism diagnostics
    $P scripts/sana_a30.py kernels    # profiler, BF16 inventory, memory, interface
    $P scripts/sana_a30.py timing     # per-call, loops, hybrid K4/K8, loaded pipeline; gate
    $P scripts/sana_a30.py checks     # reuse wrapper checks (only if the gate passed)
    $P scripts/sana_a30.py dev        # 16-prompt screen + same-state diagnostics
    /root/miniconda3/envs/py3.10/bin/python scripts/sana_a30_metrics.py dev    # LPIPS/PSNR/bootstrap in the main env
    $P scripts/sana_a30.py conf       # refuses unless the development rule passed (it did not)
    $P scripts/sana_a30_rep.py loc      # snk1b: repeatability / mutation / RNG checks, per-layer localization
    $P scripts/sana_a30_rep.py split    # snk1b: split-half persistent vs repeat noise
