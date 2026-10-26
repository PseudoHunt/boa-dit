cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu1_c.sh" >/dev/null; do sleep 15; done
bash scripts/run_gate1_sec4.sh cuda:1 "--base heig --v v1" "--base heig --fc1 rg"
until b=$(python scripts/gate1_pick2.py); do sleep 30; done
python -m boa_dit.twosided.fid5k --arm "$b" --device cuda:1
echo GPU1_D_DONE
