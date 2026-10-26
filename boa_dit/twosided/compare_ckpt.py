"""Bit-exactness check between two quantized checkpoints (+ carries): python -m boa_dit.twosided.compare_ckpt A B"""
import sys

import torch

from boa_dit.common import CKPT_DIR, RESULTS, atomic_write_json
from boa_dit.deflate.carry import carry_path


def main():
    a, b = sys.argv[1], sys.argv[2]
    sa = torch.load(f"{CKPT_DIR}/ckpt_{a}/model.pt", map_location="cpu")
    sb = torch.load(f"{CKPT_DIR}/ckpt_{b}/model.pt", map_location="cpu")
    diff = [k for k in sa if not torch.equal(sa[k], sb[k])]
    ca, cb = torch.load(carry_path(a), map_location="cpu"), torch.load(carry_path(b), map_location="cpu")
    cdiff = [k for k in ca if not (torch.equal(ca[k]["V"], cb[k]["V"]) and torch.equal(ca[k]["delta"], cb[k]["delta"]))]
    mx = max((float((sa[k] - sb[k]).abs().max()) for k in diff), default=0.0)
    res = {"a": a, "b": b, "n_tensors": len(sa), "n_differ": len(diff), "differ": diff[:20], "max_abs_diff": mx,
           "n_carry": len(ca), "n_carry_differ": len(cdiff), "bit_exact": not diff and not cdiff and set(ca) == set(cb)}
    atomic_write_json(f"{RESULTS}/gate1/compare_{a}_vs_{b}.json", res)
    print(res)


if __name__ == "__main__":
    main()
