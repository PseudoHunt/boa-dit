cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_e.sh" >/dev/null; do sleep 15; done
python -m boa_dit.twosided.downstream --device cuda:0
if python -c "import json,sys; d=json.load(open('results/gate1/downstream_b2.json')); r=d['H4']['a_over_b']; print('H4 a/b', r); sys.exit(0 if r <= 0.5 else 1)"; then
  echo "downstream-first: RUN (H4 a/b <= 0.5)"
  bash scripts/run_gate1_sec4.sh cuda:0 "--base heig --v v1 --fc1 v1 --order downstream"
else
  echo "downstream-first: SKIP (H4 a/b > 0.5)"
fi
python -m boa_dit.twosided.fid5k --arm fp --device cuda:0
echo GPU0_F_DONE
