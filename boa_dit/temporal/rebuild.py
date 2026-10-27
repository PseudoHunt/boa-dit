"""Setup 2 (rule.md): rebuild heig-8 (`deflate.quantize_dc --mode heig --r 8 --bits 4`, unchanged) and save, per layer,
the deflated Hessian handed to the solver (PHP), the per-row grid (scale, zero) the solver chose, and V in fp32.
Outputs ckpt/ckpt_deflate_heig_r8_w4/model.pt, ckpt/temporal/heig8_layers.pt, ckpt/temporal/heig8_carry.pt; the stored
git carry (results/deflate/quant/deflate_heig_r8_w4/carry.pt) is not overwritten, and is compared bit for bit.
With --bits 5|6 (exploratory comparators): the same heig-8 recipe at that bit-width; only the checkpoint and carry
are kept (ckpt/ckpt_deflate_heig_r8_w{b}/model.pt, ckpt/temporal/rebuild_quant/deflate_heig_r8_w{b}/carry.pt).
Run: python -m boa_dit.temporal.rebuild --device cuda:0 [--bits 4]
"""
import argparse
import os
import sys

import torch

from boa_dit import common as C
from boa_dit.deflate import quantize_dc as Q
from boa_dit.model import LINEARS
from boa_dit.temporal.common import CK, HEIG, LAYERS_FILE, OUT, STAGE
from quantizers.boa import BoA

SAVED, ORDER = {}, []


class _Done(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bits", type=int, default=4)
    a = ap.parse_args()
    os.makedirs(CK, exist_ok=True)
    tmp_quant = os.path.join(CK, "rebuild_quant")

    orig_quant, orig_gptq_groups, orig_make_carry = BoA.quant, Q.boa_gptq_groups, Q.make_carry

    def quant(self, *args, **kw):                                            # grab the grid the solver chose
        orig_quant(self, *args, **kw)
        SAVED[ORDER[-1]]["scale"] = self.quantizer.scale.detach().float().cpu().clone()
        SAVED[ORDER[-1]]["zero"] = self.quantizer.zero.detach().float().cpu().clone()

    def gptq_groups(lin, groups, bits, replace, *args, **kw):
        n = len(ORDER)
        key = f"b{n // len(LINEARS)}.{LINEARS[n % len(LINEARS)]}"
        ORDER.append(key)
        assert len(groups) == 1
        SAVED[key] = {"H": groups[0][1].detach().float().cpu().clone(), "bits": bits, "replace": replace}
        return orig_gptq_groups(lin, groups, bits, replace, *args, **kw)

    def make_carry(W_fp, W_q, V):
        SAVED[ORDER[-1]]["V"] = V.detach().float().cpu().clone()
        return orig_make_carry(W_fp, W_q, V)

    def run_stage(stage, name, dev, cfg):                                    # log under this phase's stage
        return C.Run(STAGE, f"rebuild_{name}", dev, cfg)

    def stop(*args, **kw):                                                   # the uniform-arm drift step needs a ckpt we lack
        raise _Done

    BoA.quant = quant
    Q.boa_gptq_groups, Q.make_carry, Q.Run, Q.load_quant_state = gptq_groups, make_carry, run_stage, stop
    Q.QUANT = tmp_quant
    Q.carry_path = lambda arm: os.path.join(tmp_quant, arm, "carry.pt")
    sys.argv = [sys.argv[0], "--mode", "heig", "--r", "8", "--bits", str(a.bits), "--device", a.device]
    try:
        Q.main()
    except _Done:
        pass
    assert len(SAVED) == 112, len(SAVED)
    if a.bits != 4:
        return
    C.atomic_torch_save(SAVED, LAYERS_FILE)
    new = torch.load(os.path.join(tmp_quant, HEIG, "carry.pt"), map_location="cpu")
    old = torch.load(os.path.join(C.RESULTS, "deflate", "quant", HEIG, "carry.pt"), map_location="cpu")
    eq = {k: bool(torch.equal(new[k]["delta"], old[k]["delta"]) and torch.equal(new[k]["V"], old[k]["V"])) for k in old}
    mx = max(float((new[k]["delta"].float() - old[k]["delta"].float()).abs().max()) for k in old)
    res = {"carry_bitexact_layers": sum(eq.values()), "n_layers": len(eq), "carry_max_abs_diff": mx}
    C.atomic_write_json(os.path.join(OUT, "rebuild_carry_check.json"), res)
    print(res, flush=True)


if __name__ == "__main__":
    main()
