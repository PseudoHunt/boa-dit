#!/bin/bash
# DNS / C-DNS main queue (results/dns/PROTOCOL.md); starts after the E3C and PixArt queues; commit + push after every job.
D=/home/boa-dit-dns; cd $D
ck() { git add results/dns results/gpu_hours.csv >/dev/null 2>&1; git commit -qm "f6 checkpoint: $1

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: $1"; }
step() { local n=$1; shift; "$@" > logs/f6_$n.log 2>&1; local rc=$?; grep -v -i -E "warn|it/s\]" logs/f6_$n.log | tail -8; ck "$n (rc $rc)"; return $rc; }
until grep -q EARLY_DONE logs/f6_early.log 2>/dev/null; do sleep 60; done
python -c "import json;assert json.load(open('results/dns/checks.json'))['pass']" || { echo "DNS CHECKS NOT PASSED: stop"; exit 6; }
(cd /home/boa-dit-e3c && python scripts/e3c_fidboot.py > logs/f5_fidboot.log 2>&1) & BOOT=$!   # E3C bootstrap: CPU, during untimed DNS work
step design python -m boa_dit.dns.run design
for a in dns qdrift; do step sample_${a}_paired python -m boa_dit.dns.run sample --arm $a --set paired; done
for a in dns qdrift; do step sample_${a}_fid python -m boa_dit.dns.run sample --arm $a --set fid; done
step cov python -m boa_dit.dns.cov
if python -c "import json,sys;sys.exit(0 if json.load(open('results/dns/cov_fit.json'))['gate']['pass'] else 1)"; then
  echo "C-DNS GATE PASSED: running the frozen ablation and full arms"
  for a in cdns_ind cdns_full; do step sample_${a}_paired python -m boa_dit.dns.run sample --arm $a --set paired; step sample_${a}_fid python -m boa_dit.dns.run sample --arm $a --set fid; done
else
  echo "C-DNS GATE FAILED: confirmation unrun (protocol §5)"
fi
step score python -m boa_dit.dns.run score
step fidboot python scripts/dns_fidboot.py
wait $BOOT; (cd /home/boa-dit-e3c && python scripts/make_e3c.py >/dev/null 2>&1 && git add results/e3c && git commit -qm "f5: E3C FID bootstrap and report

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" && git push -q && echo "PUSHED: e3c fidboot + report")
echo F6_DONE
