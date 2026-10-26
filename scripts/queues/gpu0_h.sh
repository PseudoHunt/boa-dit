cd /home/jl_fs/boa-dit
while pgrep -f "fid5k --arm static-8" >/dev/null; do sleep 15; done
bash scripts/run_gate1_sec4.sh cuda:0 "--base heig --v v1 --fc1 v1"
python -m boa_dit.twosided.oracle --arm deflate_heig_r8_w4 --steps 500 --device cuda:0
python -m boa_dit.twosided.downstream --device cuda:0
python -c "import json; d=json.load(open('results/gate1/downstream_b2.json')); r=d['H4']['a_over_b']; print('H4 a/b', r, '-> revised trigger (<= 0.75)', 'met' if r <= 0.75 else 'not met', '; run regardless')"
bash scripts/run_gate1_sec4.sh cuda:0 "--base heig --v v1 --fc1 v1 --order downstream"
python -m boa_dit.twosided.downstream --device cuda:0
echo GPU0_H_DONE
