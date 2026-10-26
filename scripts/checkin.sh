#!/usr/bin/env bash
# One code+results check-in: copies the queue scripts into scripts/queues, regenerates the reports, commits and pushes.
# Shares the heartbeat's flock so the two never race. Usage: bash scripts/checkin.sh "<message>"
cd "$(dirname "$0")/.." || exit 1
MSG="${1:-checkin $(date -u +%FT%TZ)}"
cp -f logs/*.sh scripts/queues/ 2>/dev/null
for g in make_w4a4 make_objective make_engine make_gate1_engine; do
  timeout 600 python scripts/$g.py > /dev/null 2>>logs/checkin_err.log || echo "$(date -u +%FT%TZ) $g failed" >> logs/checkin_err.log
done
flock -w 300 /tmp/boa_dit_hb.lock bash -c '
  git add boa_dit scripts configs results README.md 2>/dev/null
  git diff --cached --name-only | while read f; do
    [ -f "$f" ] && [ $(stat -c%s "$f") -gt 100000000 ] && git reset -q -- "$f"
  done
  git diff --cached --quiet || git commit -q -m "$0

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
  for i in 1 2 3; do git push -q origin main && break; git pull -q --rebase origin main; done
' "$MSG"
