"""Exploratory tables (below the marker in rule.md), same grid / PHP / V as the pair, from heig8_layers.pt:
  Wq4, Wq5 : GPTQ of W in random column orders (torch.randperm, generator seed 1 and 2 + layer index) -> period-k
             cycling of independent orderings: k3 = (Wq1, Wq3, Wq4), k4 = (Wq1, Wq3, Wq4, Wq5)
  Wq2u     : the mirror target 2W - W_q1 without the pre-loop clamp (GPTQ's loop clips and feeds the clip error forward)
Output ckpt/temporal/pair_extra.pt (same record layout as pair.pt), results/temporal/pair_extra.json.
Run: python -m boa_dit.temporal.pair_extra --device cuda:0
"""
import argparse
import os

import torch

from boa_dit import common as C
from boa_dit.deflate.carry import make_carry
from boa_dit.model import block_linear, load_dit, load_quant_state
from boa_dit.temporal.common import CK, HEIG, LAYERS_FILE, OUT, STAGE
from boa_dit.temporal.pair import codes, gptq_fixed

EXTRA_FILE = os.path.join(CK, "pair_extra.pt")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    S = torch.load(LAYERS_FILE, map_location="cpu")
    W_fp = load_dit("cpu")
    W_q = load_quant_state(load_dit("cpu"), os.path.join(C.CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    out, info = {}, {}
    with C.Run(STAGE, "explore_pair_extra", dev, vars(a)) as run:
        for li, (key, s) in enumerate(S.items()):
            l, ln = key.split(".", 1)
            Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            Wq1 = block_linear(W_q.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            scale, zero, H, V = s["scale"].to(dev), s["zero"].to(dev), s["H"].to(dev), s["V"].to(dev)
            maxq = 2 ** s["bits"] - 1
            d = Wf.shape[1]
            tabs = {}
            for name, sd in (("Wq4", 1), ("Wq5", 2)):
                perm = torch.randperm(d, generator=torch.Generator().manual_seed(1000 * sd + li))
                tabs[name] = gptq_fixed(Wf, H, scale, zero, s["bits"], s["replace"], perm=perm)
            tabs["Wq2u"] = gptq_fixed(2 * Wf - Wq1, H, scale, zero, s["bits"], s["replace"])
            Hd = H.double()
            P = torch.eye(d, device=dev, dtype=torch.float64) - V.double() @ V.double().T
            E = lambda t: (Wf - t).double() @ P
            tr = lambda X: float(torch.einsum("ij,jk,ik->", X, Hd, X))
            e1 = tr(E(Wq1))
            info[key] = {"err_ratio_q4": tr(E(tabs["Wq4"])) / e1, "err_ratio_q5": tr(E(tabs["Wq5"])) / e1,
                         "err_ratio_q2u": tr(E(tabs["Wq2u"])) / e1, "rho_pair_1_2u": tr(E(Wq1) + E(tabs["Wq2u"])) / e1,
                         "rho_pair_1_4": tr(E(Wq1) + E(tabs["Wq4"])) / e1, "frac_changed_q2u": float((tabs["Wq2u"] != Wq1).float().mean())}
            out[key] = {"scale": scale.cpu(), "zero": zero.cpu(), "V": s["V"].half(),
                        **{f"c_{n}": codes(t, scale, zero, maxq).cpu() for n, t in tabs.items()},
                        **{f"d_{n}": make_carry(Wf, t, V)["delta"] for n, t in tabs.items()}}
        C.atomic_torch_save(out, EXTRA_FILE)
        rec = run.record()
    summ = {f"mean_{k}": sum(v[k] for v in info.values()) / len(info) for k in next(iter(info.values()))}
    C.atomic_write_json(os.path.join(OUT, "pair_extra.json"), {"summary": summ, "layers": info, **rec})
    print(summ, flush=True)


if __name__ == "__main__":
    main()
