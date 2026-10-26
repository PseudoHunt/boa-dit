set -x
cd /home/jl_fs/boa-dit
python -m boa_dit.quantize --bits 3 --weighting uniform --K 1 --lam 1 --branch cfg --device cuda:1 --resume
python -m boa_dit.fp_grams --device cuda:1
echo GPU1_REBUILD_DONE
