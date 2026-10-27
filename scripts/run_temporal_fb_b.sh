#!/bin/bash
# rule_feedback.md amendment (b): recalibrate (independent lo/hi, nudged zero point) both beds, one-sided post-GELU split
# where fc2 clips > 1%, B1′ both beds, smoke gate, then DiT stage-3 arms if the smoke gate passes.
cd "$(dirname "$0")/.."
export TEMPORAL_CAP=7.54
L=results/temporal/logs
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est || { echo "CAP: skipped $log"; return 1; }
        "$@" > $L/$log.log 2>&1; rc=$?; [ $rc = 0 ] && echo "done $log" || echo "FAILED($rc) $log"; return $rc; }
while pgrep -f "boa_dit.temporal.pixart_b b1" > /dev/null; do sleep 15; done
for f in bp_b1_pixart bp_b1_pixart_layers_partial bp_calib_pixart; do [ -f results/temporal/$f.json ] && mv results/temporal/$f.json results/temporal/${f}_SUPERSEDED_alpha_shrink.json; done
echo "old PixArt B1′ archived as superseded"
job 0.05 bpb_calib_dit python -m boa_dit.temporal.dit_act calib || exit 1
job 0.20 bpb_calib_pixart python -m boa_dit.temporal.pixart_b calib || exit 1
python - <<'PY'
import json, subprocess
for bed, f, t, mod in (("dit", "bp_calib_dit.json", "mlp.fc2", "boa_dit.temporal.dit_act"),
                       ("pixart", "bp_calib_pixart.json", "ff.net.2", "boa_dit.temporal.pixart_b")):
    c = json.load(open(f"results/temporal/{f}"))["clip_rate_by_type"]["single8"][t]
    print(f"{bed} {t} single8 clip rate {c:.4f}", flush=True)
    if c > 0.01:
        print(f"{bed}: > 1% -> one-sided post-GELU split (lo = min)", flush=True)
        subprocess.run(["python", "-m", mod, "retable_lomin"], check=True)
PY
job 0.12 bpb_b1_dit python -m boa_dit.temporal.dit_act b1
job 0.25 bpb_b1_pixart python -m boa_dit.temporal.pixart_b b1
job 0.02 fb_smoke python -m boa_dit.temporal.smoke || { echo "SMOKE GATE FAILED: stage 3 not started"; exit 0; }
for d in none sym2 unif cyc4a cyc4b; do job 0.10 s3_dit_$d python -m boa_dit.temporal.dit_act arm --kind single8 --dither $d; done
job 0.11 s3_dit_feedback python -m boa_dit.temporal.dit_act arm --kind single8 --feedback
job 0.10 s3_dit_step8 python -m boa_dit.temporal.dit_act arm --kind step8 --dither none
echo B_DONE
