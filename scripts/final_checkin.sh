#!/usr/bin/env bash
# Periodic check-in every 20 min while the experiment queues run; a final check-in once all have exited.
cd "$(dirname "$0")/.." || exit 1
Q="gpu0_alloc.sh|gpu0_alloc_spectra.sh|gpu0_after.sh|w4a4_gpu1_cfgchain.sh"
n=0
while ps -eo args | grep -E "bash logs/($Q)" | grep -vq grep; do
  sleep 60; n=$((n + 1))
  [ $((n % 20)) -eq 0 ] && bash scripts/checkin.sh "checkin: code, reports and results (queues running)"
done
bash scripts/checkin.sh "final checkin: all queued experiments finished; reports regenerated"
echo "FINAL_CHECKIN_DONE $(date -u +%FT%TZ)" >> logs/checkin.log
