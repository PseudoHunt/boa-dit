cd /home/jl_fs/boa-dit
while pgrep -f "logs/gpu0_d.sh" >/dev/null; do sleep 15; done
bash scripts/run_gate1_sec4.sh cuda:0 "--base heig --v v1 --fc1 v1" "--base heig --v v2 --wo v2 --fc1 v1 --gates sample"
python -m boa_dit.twosided.oracle --arm ts_heig8_vv1_fc1v1_w4 --device cuda:0
echo GPU0_E_DONE
