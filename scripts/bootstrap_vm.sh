#!/bin/bash
# Fresh-VM bootstrap for boa-dit (handoff/START_HERE.md). Run from the repo root after cloning branch dit-joint-feedback.
# Creates .venv (Python 3.10), installs the locked stack, initializes submodules, downloads the pinned PUBLIC upstream inputs, REGENERATES the
# project-generated inputs of the DiT pilot (nothing trained is downloaded), and verifies. Needs: NVIDIA driver, git, python3.10 + venv.
# Optional env: BOA_DATA (data root, default ./data), BOA_HF_HOME (HF cache, default $BOA_DATA/hf).
set -euo pipefail
cd "$(dirname "$0")/.."
command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || { echo "nvidia-smi missing: install the NVIDIA driver first"; exit 1; }
git submodule update --init --recursive
PY=${PYTHON:-python3.10}; command -v $PY >/dev/null || PY=python3
$PY -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cu130
pip install -q -r requirements-lock.txt
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available to torch'; print('torch', torch.__version__, torch.cuda.get_device_name(0))"
python scripts/fetch_upstream.py --what "${BOA_UPSTREAM:-dit,vae,fidref}"
bash scripts/regenerate_base.sh
python scripts/verify_restore.py --smoke
echo "BOOTSTRAP_OK: activate with '. .venv/bin/activate'; next: see handoff/START_HERE.md"
