#!/bin/bash
# follow-up protocol (results/followup/PROTOCOL.md): E1 -> E2 -> E3 after the c3 queue releases the GPU; commit + push after every job.
FW=/home/boa-dit-fw; FA=/home/boa-dit-fa
ck() { (cd "$1" && git add results/followup results/gpu_hours.csv >/dev/null 2>&1 && git commit -qm "f4 checkpoint: $2

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED [$(basename $1)]: $2"); }
step() {  # dir name cmd...
  local d=$1 n=$2; shift 2
  (cd "$d" && "$@" > logs/f4_$n.log 2>&1); local rc=$?
  grep -v -i -E "warn|loading|it/s\]" "$d/logs/f4_$n.log" | tail -12
  ck "$d" "$n (rc $rc)"
  python -c "from boa_dit.f4 import spent; print(f'[f4 spent {spent():.3f} h]')" 2>/dev/null || (cd $FW && python -m boa_dit.f4 0)
  return $rc
}
while [ -n "$WAITPID" ] && kill -0 "$WAITPID" 2>/dev/null; do sleep 30; done
echo "GPU released by c3 queue: $(date -u)"
# Experiment 1
step $FW e1_live_design python -m boa_dit.followup.state_corr live_design && \
step $FW e1_live_confirm python -m boa_dit.followup.state_corr live_confirm || echo "E1 STOPPED"
# Experiment 2
step $FA e2_bench python -m boa_dit.followup.spatial2 bench && \
step $FA e2_probe python -m boa_dit.followup.spatial2 probe && \
step $FA e2_screen python -m boa_dit.followup.spatial2 screen && \
step $FA e2_confirm python -m boa_dit.followup.spatial2 confirm || echo "E2 STOPPED"
# Experiment 3
step $FW e3_check python -m boa_dit.followup.carry_opt check && \
step $FW e3_cache python -m boa_dit.followup.carry_opt cache && \
step $FW e3_profile python -m boa_dit.followup.carry_opt profile && \
step $FW e3_plan python -m boa_dit.followup.carry_opt plan && \
step $FW e3_design python -m boa_dit.followup.carry_opt design && \
step $FW e3_final python -m boa_dit.followup.carry_opt final && \
step $FW e3_eval python -m boa_dit.followup.carry_opt eval || echo "E3 STOPPED"
echo F4_DONE
