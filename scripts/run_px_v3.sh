#!/bin/bash
# third review: after the v2 diagnostics + lr 1e-4 screen -> paired α + reproducibility -> lr 1e-5 matched screen -> fresh-dev validation
D=/home/boa-dit-e3c; cd $D
ck() { git add results/e3c results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -q && git push -q && echo "PUSHED: $1"; }
while kill -0 142291 2>/dev/null; do sleep 30; done
python -m boa_dit.e3c.px_diag alpha2 > logs/f5_px_alpha2.log 2>&1; echo "alpha2 rc $?"; ck "PixArt paired α analysis + reproducibility"
python -m boa_dit.e3c.px_screen --lr 1e-5 --tag lr1e-5 --cks 10,25,50,100,200 --no-drift > logs/f5_px_screen_lr1e-5.log 2>&1; echo "screen lr1e-5 rc $?"; ck "PixArt low-rate matched screen"
python -m boa_dit.e3c.px_fresh > logs/f5_px_fresh.log 2>&1; echo "fresh rc $?"; ck "PixArt fresh-dev validation"
echo V3_DONE
