#!/bin/bash
# after the running P2 low-rate design step: push its result, then the repair diagnostics (no training)
D=/home/boa-dit-e3c; cd $D
while kill -0 136965 2>/dev/null; do sleep 30; done
git add results/e3c results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 checkpoint: px_design2 (low-rate P2 screen)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -q && git push -q && echo "PUSHED: px_design2"
grep -E "design latent|frozen" logs/f5_px_design2.log
python -m boa_dit.e3c.px_diag all > logs/f5_px_diag.log 2>&1; echo "diag rc $?"
git add results/e3c results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 checkpoint: PixArt repair diagnostics

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -q && git push -q && echo "PUSHED: px_diag"
echo DIAG_DONE
