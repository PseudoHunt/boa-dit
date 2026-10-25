#!/usr/bin/env bash
# Every 2 min: commit+push results/ (flock-guarded). Run under nohup.
cd "$(dirname "$0")/.." || exit 1
while true; do
  flock -n /tmp/boa_dit_hb.lock bash -c '
    git add results >/dev/null 2>&1
    # never commit files >100MB
    git diff --cached --name-only | while read f; do
      [ -f "$f" ] && [ $(stat -c%s "$f") -gt 100000000 ] && git reset -q -- "$f"
    done
    git diff --cached --quiet || git commit -q -m "hb $(date -u +%FT%TZ)"
    git push -q origin main 2>/dev/null
  '
  sleep 120
done
