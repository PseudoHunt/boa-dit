cd /home/jl_fs/boa-dit
python -m boa_dit.eval_fid --ref --device cuda:1 && rm -f ckpt/VIRTUAL_imagenet256_labeled.npz
bash scripts/run_gate1_sec3.sh cuda:1 dc:8:4 heig:8:4 wsvd:8:4 dc:32:4 dc:32:3
