#!/bin/bash
# follow-on (rule_extension.md decision rule, fixed before any extension result); commit + push after every job
cd "$(dirname "$0")/.."
L=results/temporal/logs
while kill -0 32435 2>/dev/null; do sleep 30; done
export TEMPORAL_CAP=$(cat results/temporal/ext_ceiling.txt)
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est 2>/dev/null || { echo "CEILING: skipped $log"; return 1; }
        echo "START $log ($(python -m boa_dit.temporal.decide budget 0 2>&1 | sed 's/ + est.*//'))"; "$@" > $L/$log.log 2>&1; rc=$?
        tail -n 12 $L/$log.log | grep -E "pilot|ABORT|latent|R =|psnr" | sed 's/^/  /'
        [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log"; scripts/temporal_checkpoint.sh "follow-on: $log (rc $rc)"; return $rc; }
decide() { python - <<'PY'
import json, os
from boa_dit.temporal.common import boot_reduction
R = "results/temporal/"
def J(p): return json.load(open(R + p)) if os.path.exists(R + p) else None
b, f = J("ext/e1_w4a4_base_screen.json"), J("ext/e1_w4a4_feedback_screen.json")
e1 = bool(b and f and boot_reduction(f["latent_rel_mse_per_image"], b["latent_rel_mse_per_image"])["ci"][0] > 0)
ps, pf = J("stage3/pixart_s3_single8_none.json"), J("stage3/pixart_s3_single8_feedback.json")
pix_flat = bool(ps and pf and boot_reduction(pf["latent_rel_mse_per_image"], ps["latent_rel_mse_per_image"])["ci"][0] <= 0)
e2 = J("ext/e2_screen.json")
best = None
if e2:
    c = {k: v for k, v in e2["arms"].items() if k in ("mean", "dct4", "rand4") and "gain_retained_R_ci" in v}
    if c:
        k = max(c, key=lambda k: c[k]["gain_retained_R"]); best = k if c[k]["gain_retained_R_ci"][0] > 0 else None
print(f"E1={int(e1)} PIXFLAT={int(pix_flat)} E2BEST={best or ''}")
PY
}
eval $(decide); echo "decision inputs: E1 promising=$E1, PixArt stage-3 feedback flat=$PIXFLAT, E2 best compressed with R CI>0=${E2BEST:-none}"
echo "- $(date -u +%H:%M) UTC follow-on decision: E1 promising=$E1, PixArt flat=$PIXFLAT, E2 best=${E2BEST:-none}" >> results/temporal/rule_extension.md
if [ "$E1" = 1 ]; then for m in base feedback; do job 0.05 ext_e1c_$m python -m boa_dit.temporal.ext_w4a4 arm --mode $m --set confirm; done
elif [ -f boa_dit/temporal/ext_e4.py ] && [ "$PIXFLAT" = 1 ]; then job 0.05 ext_e4 python -m boa_dit.temporal.ext_e4
else [ "$PIXFLAT" = 1 ] && echo "E4 triggered but ext_e4.py not written yet: skipped for now (logged)"; fi
[ -n "$E2BEST" ] && job 0.05 ext_e2_confirm python -m boa_dit.temporal.ext e2 --seeds confirm --arms none,full,$E2BEST
job 0.05 ext_e5_screen python -m boa_dit.temporal.ext e5 --seeds screen
scripts/temporal_checkpoint.sh "follow-on finished"
echo FOLLOWON_DONE
