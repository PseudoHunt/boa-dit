cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu1_a.sh" >/dev/null; do sleep 15; done
bash scripts/run_gate1_sec4.sh cuda:1 "--base uniform --v v1 --fc1 v1" "--base uniform --fc1 v1"
until f=$(python scripts/gate1_pick.py e6); do sleep 30; done
bash scripts/run_gate1_sec4.sh cuda:1 "$f"
python -m boa_dit.twosided.fid5k --arm deflate_dc_r8_w4 --device cuda:1
until b=$(python scripts/gate1_pick.py best); do sleep 30; done
python -m boa_dit.twosided.fid5k --arm "$b" --device cuda:1
python -m boa_dit.eval_fid --name deflate_dc_r32_w3 --n 2000 --stage gate1 --carry --device cuda:1
echo GPU1_B_DONE
