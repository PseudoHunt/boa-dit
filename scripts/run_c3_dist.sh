#!/bin/bash
# consolidated §3 queue (rule_distribution.md): D2 -> IR untouched -> score -> D3; commit + push after each job.
cd "$(dirname "$0")/.."
IRPY=/tmp/claude-0/-home/9a42aff5-4872-4f32-ba52-5b677aa2b906/scratchpad/irenv/bin/python
ck() { git add -A results/temporal/dist results/temporal/stage3 results/gpu_hours.csv boa_dit scripts >/dev/null 2>&1
       git commit -qm "c3 §3 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q origin temporal-oracles && echo "PUSHED: $1"; }
fits() { python - "$1" <<'PY'
import sys; from boa_dit.temporal.c3 import spent, CAP
est = float(sys.argv[1]); s = spent(); ok = s + est + 0.30 <= CAP      # keep the 0.30 h §5 reserve
print(f"[c3 budget] spent {s:.3f} + est {est:.2f} + reserve 0.30 vs {CAP} -> {'OK' if ok else 'UNRUN'}"); sys.exit(0 if ok else 1)
PY
}
# (gen2 already finished; the earlier pgrep wait matched its own launcher)
grep -q "Traceback" logs/c3_dist_gen2.log && { echo "gen2 FAILED"; exit 1; }
ck "D2 images on untouched prompts mjhq1k[64:96]"
$IRPY -m boa_dit.temporal.dist ir --set untouched > logs/c3_ir_untouched.log 2>&1 && echo "IR untouched done" || { echo "IR FAILED"; exit 1; }
python -m boa_dit.temporal.dist score > logs/c3_score.log 2>&1 && cat logs/c3_score.log | grep -v -i warn
ck "D1/D2 PixArt scores (ImageReward, PSNR, LPIPS, latent)"
if fits 1.35; then
  python -m boa_dit.temporal.fid_act --kind single8 --dither none --feedback --compress dct4 --n 2000 --stage c3 > logs/c3_fid_dct4.log 2>&1 && tail -1 logs/c3_fid_dct4.log
  ck "D3 DiT FID-2k DCT-4 feedback"
fi
echo C3_DIST_DONE
