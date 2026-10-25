#!/usr/bin/env bash
# Usage: run_q_eval.sh <device> <stage> <quantize.py args...>
# Quantize (resumable), then latent-MSE eval (resumable). Name is derived exactly as quantize.py does.
set -euo pipefail
cd "$(dirname "$0")/.."
dev=$1; stage=$2; shift 2
name=$(python - "$@" <<'EOF'
import sys, argparse
sys.argv = ["x"] + sys.argv[1:]
from boa_dit.quantize import run_name
ap = argparse.ArgumentParser()
for k, v in [("--bits", 4), ("--K", 4), ("--lam", 1.0), ("--tau", 1.0), ("--clip_ratio", 0.0)]:
    ap.add_argument(k, type=type(v), default=v)
for k, v in [("--weighting", "uniform"), ("--branch", "cfg"), ("--gain_tag", "gain_iso_rho0.01"),
             ("--gain_stat", "mean"), ("--suffix", "")]:
    ap.add_argument(k, default=v)
a, _ = ap.parse_known_args()
print(run_name(a))
EOF
)
echo "[$(date -u +%T)] $name on $dev"
if [ ! -f "results/stage2/quant/$name.json" ]; then
  python -m boa_dit.quantize --device "$dev" --resume "$@" 2>&1 | grep -v Warn
fi
if [ ! -f "results/$stage/eval/latent_${name}${EVAL_TAG:-}.json" ]; then
  python -m boa_dit.eval_latent_mse --name "$name" --device "$dev" --stage "$stage" ${EVAL_ARGS:-} 2>&1 | grep -v Warn | tail -n 1
fi
