"""D1 (rule.md): guidance diagnostic on heig-8.
  layer level : one calibration pass along heig-8's own quantized prefix (blocks 0..l-1 heig-8 with carries; block l
                FP linears + W8 adaLN = the state quantize_dc.py collected H in). Per (layer, bucket): S_cc, S_uu, S_dd,
                S_ud (d = x_c - x_u, same image/step/token). H_cur = 0.9 S_cc + 0.1 S_uu; H_cfg = S_uu + w(S_ud + S_ud^T)
                + w^2 S_dd. With E_eff = (W - W_q)(I - V V^T): traces, r_cfg, and the shared / difference / cross split.
  network     : from D4's saved branches (eval seeds 0-31 x steps 0,5,..,45): heig-8 vs FP branch errors (3 guided ch).
Run: python -m boa_dit.objective.d1 --device cuda:0
"""
import argparse
import copy
import os

import numpy as np
import torch

from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.carry import carry_path, install
from boa_dit.model import LINEARS, block_linear, load_dit, load_quant_state, make_diffusion
from boa_dit.quantize import CALIB_SEEDS

BASE = "deflate_heig_r8_w4"
W_CFG, B = 1.5, 10
OUT = os.path.join(RESULTS, "objective")


class PairStats:
    """Hooks on block linears; cond slab first (stores x_c), then the uncond slab of the same step (pairs)."""

    def __init__(self, blk):
        self.S = {}           # (layer, bucket) -> dict of sums
        self.n = {}
        self.xc, self.bucket, self.branch = {}, None, None
        self.h = [block_linear(blk, ln).register_forward_pre_hook(self._hook(ln)) for ln in LINEARS]

    def _hook(self, ln):
        def h(mod, inp):
            with torch.autocast("cuda", enabled=False):
                x = inp[0].reshape(-1, inp[0].shape[-1]).float()
                if self.branch == 0:
                    self.xc[ln] = x
                    return
                xc, xu = self.xc.pop(ln), x
                d = xc - xu
                key = (ln, self.bucket)
                torch.backends.cuda.matmul.allow_tf32 = True
                try:
                    vals = {"cc": xc.T @ xc, "uu": xu.T @ xu, "dd": d.T @ d, "ud": xu.T @ d}
                finally:
                    torch.backends.cuda.matmul.allow_tf32 = False
                if key in self.S:
                    for k, v in vals.items():
                        self.S[key][k] += v
                    self.n[key] += x.shape[0]
                else:
                    self.S[key], self.n[key] = vals, x.shape[0]
        return h

    def remove(self):
        for h in self.h:
            h.remove()


def trq(E, S):
    return float(((E @ S) * E).sum())


@torch.no_grad()
def layer_level(dev):
    q = load_dit(dev)
    load_quant_state(q, os.path.join(CKPT_DIR, f"ckpt_{BASE}", "model.pt"))
    carry = torch.load(carry_path(BASE), map_location="cpu")
    install(q, carry)
    fp = load_dit(dev)
    diffusion = make_diffusion(50)
    S_, N = diffusion.num_timesteps, len(CALIB_SEEDS)
    X, C = capture_block0_inputs(fp, diffusion, CALIB_SEEDS, W_CFG, dev)
    per = S_ // B
    out = {}
    for l in range(28):
        blk = copy.deepcopy(fp.blocks[l])
        blk.adaLN_modulation[1].weight.data.copy_(q.blocks[l].adaLN_modulation[1].weight.data)
        ps = PairStats(blk)
        try:
            for s in range(S_):
                ps.bucket = s // per
                for br in range(2):
                    ps.branch = br
                    r0 = s * 2 * N + br * N
                    with torch.autocast("cuda", dtype=torch.float16):
                        blk(X[r0:r0 + N].float(), C[r0:r0 + N])
        finally:
            ps.remove()
        for ln in LINEARS:
            key = f"b{l}.{ln}"
            Wf = block_linear(fp.blocks[l], ln).weight.data.float()
            Wq = block_linear(q.blocks[l], ln).weight.data.float()
            V = carry[key]["V"].to(dev).float()
            E = (Wf - Wq) - (Wf - Wq) @ V @ V.T
            rec = {"buckets": []}
            tot = {k: 0.0 for k in ("cur", "cfg", "shared", "diff", "cross")}
            for b in range(B):
                st, n = ps.S[(ln, b)], ps.n[(ln, b)]
                m = {k: v / n for k, v in st.items()}
                shared, diff = trq(E, m["uu"]), W_CFG ** 2 * trq(E, m["dd"])
                cross = W_CFG * 2 * float(((E @ m["ud"]) * E).sum())
                cur = 0.9 * trq(E, m["cc"]) + 0.1 * shared
                cfg = shared + diff + cross
                rec["buckets"].append({"cur": cur, "cfg": cfg, "shared": shared, "diff": diff, "cross": cross})
                for k, v in zip(("cur", "cfg", "shared", "diff", "cross"), (cur, cfg, shared, diff, cross)):
                    tot[k] += v / B
            rec.update(tot)
            rec["r_cfg"] = tot["cfg"] / tot["cur"]
            rec["diffcross_share"] = (tot["diff"] + tot["cross"]) / tot["cfg"]
            out[key] = rec
        del ps
        torch.cuda.empty_cache()
        with torch.autocast("cuda", dtype=torch.float16):                     # heig-8 block -> next inputs
            for s in range(S_):
                for br in range(2):
                    r0 = s * 2 * N + br * N
                    X[r0:r0 + N] = q.blocks[l](X[r0:r0 + N].float(), C[r0:r0 + N]).to(torch.float16)
        print(f"block {l}: r_cfg " + " ".join(f"{ln.split('.')[1]}={out[f'b{l}.{ln}']['r_cfg']:.2f}" for ln in LINEARS), flush=True)
    return out


def network_level():
    fp = torch.load(os.path.join(CKPT_DIR, "objective", "branches_fp.pt"))["fp"]
    hq = torch.load(os.path.join(CKPT_DIR, "objective", "branches_heig8.pt"))["heig-8"]
    rows, steps = [], sorted(fp)
    for s in steps:
        n = fp[s]["raw"].shape[0] // 2
        dc = (hq[s]["raw"][:n] - fp[s]["raw"][:n]).double()
        du = (hq[s]["raw"][n:] - fp[s]["raw"][n:]).double()
        dg = W_CFG * dc + (1 - W_CFG) * du
        dd = dc - du
        e = lambda t: float(t.pow(2).sum())
        rows.append({"step": s, "g": e(dg), "c": e(dc), "u": e(du), "shared": e(du), "diff": W_CFG ** 2 * e(dd),
                     "cross": 2 * W_CFG * float((du * dd).sum()),
                     "corr_cu": float(np.corrcoef(dc.flatten().numpy(), du.flatten().numpy())[0, 1]),
                     "g4": float((hq[s]["res"] - fp[s]["res"]).double().pow(2).flatten(1).sum(1).mean())})
    tot = {k: sum(r[k] for r in rows) for k in ("g", "c", "u", "shared", "diff", "cross")}
    return {"per_step": rows, "ratio_g_over_mean_branch": tot["g"] / ((tot["c"] + tot["u"]) / 2),
            "ratio_g_over_cur_weighted": tot["g"] / (0.9 * tot["c"] + 0.1 * tot["u"]),
            "corr_cu_mean": float(np.mean([r["corr_cu"] for r in rows])),
            "split_share": {k: tot[k] / tot["g"] for k in ("shared", "diff", "cross")},
            "ratio_by_step": [r["g"] / ((r["c"] + r["u"]) / 2) for r in rows],
            "dg4_energy_per_sample_by_step": {r["step"]: r["g4"] for r in rows}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    with Run("objective", "d1_guidance", a.device, vars(a)) as run:
        net = network_level()
        lay = layer_level(a.device)
        tot_cfg = sum(v["cfg"] for v in lay.values())
        tot_cur = sum(v["cur"] for v in lay.values())
        heavy = [k for k, v in lay.items() if v["diffcross_share"] >= 0.30]
        held = sum(lay[k]["cfg"] for k in heavy) / tot_cfg
        by_type = {ln: {"r_cfg": float(np.mean([lay[f"b{l}.{ln}"]["r_cfg"] for l in range(28)])),
                        "diffcross_share": float(np.mean([lay[f"b{l}.{ln}"]["diffcross_share"] for l in range(28)])),
                        "share_of_cfg_error": sum(lay[f"b{l}.{ln}"]["cfg"] for l in range(28)) / tot_cfg} for ln in LINEARS}
        res = {"network": net, "layers": lay, "model_r_cfg": tot_cfg / tot_cur, "by_type": by_type,
               "layers_diffcross_ge_30pct": heavy, "their_share_of_cfg_error": held,
               "read1_layer_part": held >= 0.5, "read1_network_part": net["ratio_g_over_mean_branch"] >= 2,
               "toy_check": {"x_c": [1, 1], "x_u": [1, 0], "w": 4, "v": [1, 4],
                             "note": "error on the channel where the branches differ counts 16x in ||E v||^2 vs 1x per branch"}}
        res["read1_lever"] = res["read1_layer_part"] and res["read1_network_part"]
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "d1.json"), res)
    print({k: res[k] for k in ("model_r_cfg", "their_share_of_cfg_error", "read1_layer_part", "read1_network_part")},
          net["ratio_g_over_mean_branch"], net["corr_cu_mean"], net["split_share"], flush=True)


if __name__ == "__main__":
    main()
