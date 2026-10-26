set -x
cd /home/jl_fs/boa-dit
python -m boa_dit.quantize --bits 4 --weighting uniform --K 1 --lam 1 --branch cfg --device cuda:0 --resume
python -m boa_dit.eval_latent_mse --name uniform_w4_K1_lam1_brcfg --stage gate1/rebuild --device cuda:0
echo GPU0_REBUILD_DONE
