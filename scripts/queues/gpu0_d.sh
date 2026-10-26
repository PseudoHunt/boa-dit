cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_c.sh" >/dev/null; do sleep 15; done
python -m boa_dit.twosided.test_solver --device cuda:0
bash scripts/run_gate1_sec4.sh cuda:0 "--base uniform --v v2 --fc1 v1 --wo v2" "--base uniform --v v1"
python -m boa_dit.twosided.fid5k --arm uniform --device cuda:0
python -m boa_dit.twosided.fid5k --arm static-8 --device cuda:0
python -m boa_dit.lowrank --bits 3 --arms static-32 --regimes static --fid --device cuda:0
echo GPU0_D_DONE
