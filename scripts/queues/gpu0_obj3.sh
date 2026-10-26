cd /home/jl_fs/boa-dit
python -u -m boa_dit.objective.oracle_band --device cuda:0 2>&1 | grep -v -i warn
python -u -m boa_dit.w4a4.cfg_branch --device cuda:0 2>&1 | grep -v -i warn | tail -3
echo GPU0_OBJ3_DONE
