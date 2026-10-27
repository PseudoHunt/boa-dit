#!/bin/bash
# remaining work, no budget stop; commit + push after every job (scripts/temporal_checkpoint.sh)
cd "$(dirname "$0")/.."
L=results/temporal/logs
while kill -0 31664 2>/dev/null; do sleep 20; done
scripts/temporal_checkpoint.sh "part A finished (arm 4a, FID-2k arm 1)"
job() { log=$1; shift; echo "START $log ($(python -m boa_dit.temporal.decide budget 0 2>&1 | sed 's/ + est.*//'))"; "$@" > $L/$log.log 2>&1; rc=$?
        [ $rc = 0 ] && echo "DONE $log" || echo "FAILED($rc) $log (see $L/$log.log)"; return $rc; }
pix() { python -c "
import json; r=json.load(open('results/temporal/stage3/pixart_$1.json'))
print('  PixArt $1: PSNR %.3f, LPIPS %.4f, latent %.5f, pred C/I %.4f, gpu_h %.3f' % (r['psnr_mean'], r['lpips_mean'], r['latent_rel_mse_mean'], r['diag']['C_over_I'], r['gpu_hours'])
       + ('' if r['feedback_clip_rate'] is None else ', clip %.4%%, clamp %.2e' % (r['feedback_clip_rate'], r['feedback_residual_clamp_rate'])))"; }
job fid_dit_feedback python -m boa_dit.temporal.fid_act --kind single8 --feedback && python -c "
import json; r=json.load(open('results/temporal/stage3/fid2000_dit_bp_single8_feedback.json')); print('  FID-2k arm 5 (feedback):', round(r['fid'],3), 'gpu_h', round(r['gpu_hours'],3))"
scripts/temporal_checkpoint.sh "FID-2k arm 5 (feedback)"
if job pix_smoke python -m boa_dit.temporal.pixart_b s3 --kind single8 --feedback --n 2 --tag smoke; then
  pix s3_single8_feedback_smoke; scripts/temporal_checkpoint.sh "PixArt smoke (2 images)"
  for spec in "single8 none" "single8 feedback" "step8 none" "single8 unif"; do
    set -- $spec; k=$1; d=$2
    if [ "$d" = feedback ]; then job pix_${k}_$d python -m boa_dit.temporal.pixart_b s3 --kind $k --feedback && pix s3_${k}_feedback
    else job pix_${k}_$d python -m boa_dit.temporal.pixart_b s3 --kind $k --dither $d && pix s3_${k}_$d; fi
    scripts/temporal_checkpoint.sh "PixArt arm $k $d"
  done
else
  echo "PixArt smoke FAILED: PixArt arms not started"; scripts/temporal_checkpoint.sh "PixArt smoke failed"
fi
job s3_dit_cyc4b python -m boa_dit.temporal.dit_act arm --kind single8 --dither cyc4b && python -c "
import json; from boa_dit.temporal.common import boot_reduction
L=lambda n: json.load(open(f'results/temporal/eval/dit_bp_{n}.json')); a,b,f=L('single8_cyc4b'),L('single8_none'),L('single8_feedback')
r1=boot_reduction(a['rel_mse_per_image'],b['rel_mse_per_image']); r5=boot_reduction(f['rel_mse_per_image'],a['rel_mse_per_image'])
print(f'  arm cyc4b: latent {a[\"rel_mse_mean\"]:.5f}, vs single {-100*r1[\"reduction\"]:+.1f}%, feedback vs it {-100*r5[\"reduction\"]:+.1f}% [{-100*r5[\"ci\"][1]:+.1f}, {-100*r5[\"ci\"][0]:+.1f}]')"
scripts/temporal_checkpoint.sh "arm 4b (cyc4b); all queued work done"
echo ALL_DONE
