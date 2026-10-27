#!/bin/bash
# rule_feedback.md stage 3 on DiT (smoothed fc1/fc2, guarded feedback), with a one-line summary after every job.
cd "$(dirname "$0")/.."
export TEMPORAL_CAP=7.54
L=results/temporal/logs
job() { est=$1; log=$2; shift 2; python -m boa_dit.temporal.decide budget $est 2>&1 | sed 's/^/  budget: /' || { echo "CAP: skipped $log"; return 1; }
        python -m boa_dit.temporal.decide budget $est 2>/dev/null || { echo "CAP: skipped $log"; return 1; }
        echo "START $log"; "$@" > $L/$log.log 2>&1; rc=$?; [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log (see $L/$log.log)"; return $rc; }
summ() { python - "$@" <<'PY'
import json, sys, numpy as np
what = sys.argv[1]
R = "results/temporal/"
if what == "calib":
    c = json.load(open(R + "bp_calib_dit.json"))
    print("  calib: clip rate single8 by type", {k: f"{v:.4%}" for k, v in c["clip_rate_by_type"]["single8"].items()}, "| alpha", c["summary"]["single8"], "| gpu_h", round(c.get("gpu_hours", 0), 3))
elif what == "b1":
    b = json.load(open(R + "bp_b1_dit.json"))
    print("  B1':", {k: round(v["C_over_I"], 4) for k, v in b["net"].items()}, "| read 2 excess", round(b["net"]["single8"]["C_over_I"] - b["net"]["dynamic8"]["C_over_I"], 4),
          "| top-half rho (single8 none / sym2)", round(b["layer"]["top_half_weighted"]["single8_none"]["rho_adj"], 3), round(b["layer"]["top_half_weighted"]["single8_sym2"]["rho_adj"], 3))
elif what == "smoke":
    s = json.load(open(R + "stage3/smoke_gate.json"))
    print("  smoke (seeds 0-1): single8", round(s["single8"]["rel_mse"], 4), "| guarded feedback", round(s["feedback"]["rel_mse"], 4), "| heig-8 ref 0.068 (sanity, not a gate)")
elif what == "precheck":
    p = json.load(open(R + "stage3/precheck_dit.json"))
    print("  precheck: max cast/round", f"{p['max_cast_ratio_any_type']:.2e}", "| scorer worst", p["scorer_worst_layer"], round(p["scorer"][p["scorer_worst_layer"]]["max_rel_mismatch"], 4))
elif what == "fbdiag":
    f = json.load(open(R + "stage3/fb_matched_diag.json"))
    for k, v in f["arms"].items():
        print(f"  {k}: endpoint {v['endpoint_rel_mse']:.4f}, pred C/I {v['pred_C_over_I']:.4f}, clip {np.mean(v['clip_rate_per_step']):.4%}")
    for g, v in f["identity_one_seed"].items():
        print(f"  identity {g}: max discrepancy/sqrt(sum e^2) {max(l['discrepancy_over_sqrt_sum_e_energy'] for l in v['layers'].values()):.2e}, clamp rate {v['clamp_rate']:.2e}")
else:
    a = json.load(open(R + f"eval/dit_bp_{what}.json"))
    print(f"  arm {what}: latent {a['rel_mse_mean']:.5f} ± {a['rel_mse_sem']:.5f}, pred C/I {a['diag']['C_over_I']:.4f}, clip {np.mean(a['clip_rate_per_step']):.4%}, gpu_h {a['gpu_hours']:.3f}")
PY
}
job 0.25 s3_calib python -m boa_dit.temporal.dit_act calib && summ calib || exit 1
job 0.13 s3_b1 python -m boa_dit.temporal.dit_act b1 && summ b1
job 0.02 s3_smoke python -m boa_dit.temporal.smoke; summ smoke
job 0.02 s3_precheck python -m boa_dit.temporal.precheck --bed dit && summ precheck
job 0.08 s3_fbdiag python -m boa_dit.temporal.fb_diag && summ fbdiag
for d in none sym2 unif cyc4a cyc4b; do job 0.10 s3_dit_$d python -m boa_dit.temporal.dit_act arm --kind single8 --dither $d && summ single8_$d; done
job 0.11 s3_dit_feedback python -m boa_dit.temporal.dit_act arm --kind single8 --feedback && summ single8_feedback
job 0.10 s3_dit_step8 python -m boa_dit.temporal.dit_act arm --kind step8 --dither none && summ step8_none
python scripts/make_implementation.py > /dev/null 2>&1 && echo "implementation.md regenerated"
echo S3_DIT_DONE
