#!/bin/bash
# every 10 min: snapshot run logs into results/<dir>/logs and push each active worktree (survives a server loss mid-run)
# usage: autopush_all.sh "<worktree>:<results subdir>:<log glob>" ...
while true; do
  for spec in "$@"; do IFS=: read w sub glob <<< "$spec"
    mkdir -p $w/results/$sub/logs
    for f in $w/logs/$glob; do [ -f "$f" ] && tr '\r' '\n' < "$f" | grep -v -E "it/s\]|Loading pipeline" > $w/results/$sub/logs/$(basename $f).txt; done
    (cd $w && git add results/$sub scripts boa_dit >/dev/null 2>&1 && git diff --cached --quiet || \
      { git commit -qm "autosave: logs and partial results $(date -u +%H:%MZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "$(date -u +%H:%M) pushed $(basename $w)"; })
  done
  sleep 600
done
