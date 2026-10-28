#!/bin/bash
# every 10 min: snapshot run logs into results/followup/logs and push both follow-up branches (survives a server loss mid-run)
while true; do
  for w in /home/boa-dit-fw /home/boa-dit-fa; do
    mkdir -p $w/results/followup/logs
    for f in $w/logs/f4_*.log; do [ -f "$f" ] && tr '\r' '\n' < "$f" | grep -v -E "it/s\]|Loading pipeline" > $w/results/followup/logs/$(basename $f).txt; done
    (cd $w && git add results/followup scripts boa_dit/followup >/dev/null 2>&1 && git diff --cached --quiet || \
      { git commit -qm "f4 autosave: logs and partial results $(date -u +%H:%MZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$(date -u +%H:%M) pushed $(basename $w)"; })
  done
  sleep 600
done
