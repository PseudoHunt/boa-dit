cd /home/jl_fs/boa-dit
while pgrep -f "boa_dit.twosided.d2" >/dev/null; do sleep 15; done
bash scripts/run_gate1_sec3.sh cuda:0 dc:32:3
bash scripts/run_gate1_sec4.sh cuda:0 "--act_order"
python -c "
import torch; torch.cuda.set_device('cuda:0')
from boa_dit.common import Run
from boa_dit import lowrank
for b in (4, 3):
    with Run('gate1', f'lowrank_build_static_w{b}', 'cuda:0', {'bits': b, 'regimes': ['static']}):
        lowrank.build(b, 'cuda:0', ['static'])
"
echo GPU0_B_DONE
