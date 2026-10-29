#!/bin/bash
# stop the old diag run at its (old) steps stage, then diagnostics v2 and the update-matched screen
D=/home/boa-dit-e3c; cd $D
ck() { git add results/e3c results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f5 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -q && git push -q && echo "PUSHED: $1"; }
until grep -q "^== steps" logs/f5_px_diag.log || ! kill -0 140277 2>/dev/null; do sleep 15; done
kill 140277 2>/dev/null; sleep 5
ck "PixArt diagnostics v1 (norm, branch, interv, alpha); old steps stage stopped by design"
python -m boa_dit.e3c.px_diag v2 > logs/f5_px_diag_v2.log 2>&1; echo "v2 rc $?"; ck "PixArt diagnostics v2"
python -m boa_dit.e3c.px_screen > logs/f5_px_screen.log 2>&1; echo "screen rc $?"; ck "PixArt update-matched screen"
echo V2_SCREEN_DONE
