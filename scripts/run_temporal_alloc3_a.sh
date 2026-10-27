#!/bin/bash
# allocation 3.1 h (hard stop at a logged total of 11.05 h), part A: arm 4a, FID-2k of arm 1
cd "$(dirname "$0")/.."
export TEMPORAL_CAP=11.05
L=results/temporal/logs
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est 2>/dev/null || { echo "CAP: skipped $log"; return 1; }
        echo "START $log ($(python -m boa_dit.temporal.decide budget 0 2>&1 | sed 's/ + est.*//'))"; "$@" > $L/$log.log 2>&1; rc=$?
        [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log (see $L/$log.log)"; return $rc; }
job 0.23 s3_dit_cyc4a python -m boa_dit.temporal.dit_act arm --kind single8 --dither cyc4a &&
python - <<'PY'
import json, numpy as np
from boa_dit.temporal.common import boot_reduction
L = lambda n: json.load(open(f"results/temporal/eval/dit_bp_{n}.json"))
a, b, f = L("single8_cyc4a"), L("single8_none"), L("single8_feedback")
r1 = boot_reduction(a["rel_mse_per_image"], b["rel_mse_per_image"]); r5 = boot_reduction(f["rel_mse_per_image"], a["rel_mse_per_image"])
print(f"  arm cyc4a: latent {a['rel_mse_mean']:.5f}, vs single {-100*r1['reduction']:+.1f}% [{-100*r1['ci'][1]:+.1f}, {-100*r1['ci'][0]:+.1f}], "
      f"feedback vs it {-100*r5['reduction']:+.1f}% [{-100*r5['ci'][1]:+.1f}, {-100*r5['ci'][0]:+.1f}], pred C/I {a['diag']['C_over_I']:.4f}, gpu_h {a['gpu_hours']:.3f}")
PY
job 0.75 fid_dit_single8 python -m boa_dit.temporal.fid_act --kind single8 --dither none && python -c "
import json; r=json.load(open('results/temporal/stage3/fid2000_dit_bp_single8_none.json')); print('  FID-2k arm 1 (single table):', round(r['fid'],3), 'gpu_h', round(r['gpu_hours'],3))"
echo ALLOC_A_DONE
