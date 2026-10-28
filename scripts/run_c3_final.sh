#!/bin/bash
# user instruction 2026-10-28 (rule_distribution.md): finish DCT-16 -> score -> resume D3 from its 128-image partial; §5 dropped.
cd "$(dirname "$0")/.."
ck() { git add results/temporal/dist results/temporal/stage3 results/temporal/rule_distribution.md results/gpu_hours.csv boa_dit/temporal scripts >/dev/null 2>&1
       git commit -qm "c3 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin temporal-oracles && echo "PUSHED: $1"; }
while kill -0 "$GENPID" 2>/dev/null; do sleep 20; done
grep -E "pilot|budget|done|Error" logs/c3_dct16_gen.log
grep -q Traceback logs/c3_dct16_gen.log && { echo "gen16 FAILED"; ck "DCT-16 images (FAILED)"; exit 1; }
ck "DCT-16 feedback images, both sets"
python -m boa_dit.temporal.dist score > logs/c3_score16.log 2>&1 && grep -v -i warn logs/c3_score16.log || { echo "score FAILED"; exit 1; }
ck "DCT-16 trade-off read (PSNR/LPIPS/latent, both sets)"
echo DCT16_DONE
# D3 resume: approved overrun to ~3.3 h; hard stop if the projection at launch exceeds 3.45 h
python - <<'PY' || { echo "D3 UNRUN (projection above 3.45 h)"; exit 0; }
import sys; from boa_dit.temporal.c3 import spent
s = spent(); p = s + 59 * 79 / 3600 + 0.04
print(f"[c3 budget] D3 resume: spent {s:.3f} h, projected end {p:.2f} h (approved overrun to ~3.3 h; hard stop 3.45)"); sys.exit(0 if p <= 3.45 else 1)
PY
python -m boa_dit.temporal.fid_act --kind single8 --dither none --feedback --compress dct4 --n 2000 --stage c3 > logs/c3_fid_dct4.log 2>&1 && tail -1 logs/c3_fid_dct4.log || echo "D3 FAILED"
ck "D3 DiT FID-2k DCT-4 feedback (resumed from 128/2000)"
python -c "from boa_dit.temporal.c3 import spent; print(f'c3 final spend {spent():.3f} h')"
echo C3_ALL_DONE
