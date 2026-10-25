"""Deflate-and-carry G1 (FP only): token-mean energy of every block linear's input Gram, its rank, the directions the
old damping erases, and whether the token mean is the adaLN shift path. See results/deflate/G1.md.

One FP calibration pass (seeds 1000-1031, both branches, 50 steps); per block, forward pre-hooks on the 4 linears
accumulate, per CFG branch: G = sum x x^T, Gm = sum_s 256 xbar_s xbar_s^T, per-bucket traces of both, cos(xbar_s, p_s)
for every sample, and sum p_s p_s^T. H / H_mean use the uniform arm's weights (1/B per bucket, branch cfg weights).
-> results/deflate/g1.json, g1_spectra.json, configs/deflate_r.json
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from boa_dit.calib import capture_block0_inputs
from boa_dit.common import ROOT, RESULTS, Run, atomic_write_json
from boa_dit.model import LINEARS, block_linear, load_dit, make_diffusion
from boa_dit.quantize import CALIB_SEEDS

OUT = os.path.join(RESULTS, "deflate")
B, CFG = 10, 1.5
SPECTRA_BLOCKS = (0, 13, 27)


def eigh64(H):
    """Ascending eigenvalues/eigenvectors in fp64; GPU first, CPU fallback (GPU eigh failed to converge before)."""
    try:
        return torch.linalg.eigh(H.double())
    except RuntimeError:
        e, V = torch.linalg.eigh(H.double().cpu())
        return e.to(H.device), V.to(H.device)


def r_of(eig_desc, frac=0.95, cap=8):
    c = eig_desc.clamp_min(0).cumsum(0) / eig_desc.clamp_min(0).sum()
    return int(min(cap, int((c < frac).sum()) + 1))


class Acc:
    def __init__(self, block, dev):
        self.dev, self.cur, self.p = dev, None, None
        self.d = {}
        self.hooks = [block_linear(block, ln).register_forward_pre_hook(self._hook(ln)) for ln in LINEARS]

    def _hook(self, ln):
        @torch.autocast("cuda", enabled=False)
        def h(mod, inp):
            x = inp[0].float()                                                    # [N, 256, d]
            b, br = self.cur
            xb = x.mean(1)                                                        # [N, d]
            p = self.p[ln]                                                        # [N, d]
            xf = x.reshape(-1, x.shape[-1])
            if ln not in self.d:
                d = x.shape[-1]
                z = lambda *s: torch.zeros(*s, device=self.dev, dtype=torch.float32)
                self.d[ln] = {"G": z(2, d, d), "Gm": z(2, d, d), "Pp": z(d, d), "trG": z(B, 2), "trGm": z(B, 2),
                              "n": z(B, 2), "cos": []}
            a = self.d[ln]
            a["G"][br] += xf.T @ xf
            a["Gm"][br] += 256.0 * (xb.T @ xb)
            a["Pp"] += p.T @ p
            a["trG"][b, br] += (xf * xf).sum()
            a["trGm"][b, br] += 256.0 * (xb * xb).sum()
            a["n"][b, br] += xf.shape[0]
            a["cos"].append(F.cosine_similarity(xb, p, dim=-1).cpu())
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def predicted(blk, c):
    """Token-constant input direction p_s for each linear, from the block's adaLN modulation (fp32)."""
    with torch.autocast("cuda", enabled=False):
        mod = blk.adaLN_modulation(c.float())
        shift_msa, _, _, shift_mlp, _, _ = mod.chunk(6, dim=1)
        C = shift_msa.shape[1]
        Wv, bv = blk.attn.qkv.weight[2 * C:].float(), blk.attn.qkv.bias[2 * C:].float()
        h1 = F.linear(shift_mlp, blk.mlp.fc1.weight.float(), blk.mlp.fc1.bias.float())
        return {"attn.qkv": shift_msa, "mlp.fc1": shift_mlp, "attn.proj": F.linear(shift_msa, Wv, bv),
                "mlp.fc2": F.gelu(h1, approximate="tanh")}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    wc, wu = CFG ** 2, (CFG - 1) ** 2
    wc, wu = wc / (wc + wu), wu / (wc + wu)
    res, spectra = {"layers": {}}, {}
    with Run("deflate", "g1", dev, vars(a)) as run:
        model, diffusion = load_dit(dev), make_diffusion(50)
        assert type(model.blocks[0].mlp.act).__name__ == "GELU" and model.blocks[0].mlp.act.approximate == "tanh"
        S, N = diffusion.num_timesteps, len(CALIB_SEEDS)
        X, C = capture_block0_inputs(model, diffusion, CALIB_SEEDS, CFG, dev)
        for l, blk in enumerate(model.blocks):
            acc = Acc(blk, dev)
            try:
                for s in range(S):
                    for br in range(2):
                        r0 = s * 2 * N + br * N
                        acc.cur, acc.p = (s // (S // B), br), predicted(blk, C[r0:r0 + N])
                        with torch.autocast("cuda", dtype=torch.float16):
                            y = blk(X[r0:r0 + N].float(), C[r0:r0 + N])
                        X[r0:r0 + N] = y.to(torch.float16)                         # FP propagation
            finally:
                acc.remove()
            for ln in LINEARS:
                d = acc.d[ln]
                n = d["n"][0, 0]
                assert torch.all(d["n"] == n)
                norm = B * n                                                       # (1/B) x per-(bucket, branch) mean
                H = (wc * d["G"][0] + wu * d["G"][1]) / norm
                Hm = (wc * d["Gm"][0] + wu * d["Gm"][1]) / norm
                Hv = H - Hm
                share_b = ((wc * d["trGm"][:, 0] + wu * d["trGm"][:, 1]) / (wc * d["trG"][:, 0] + wu * d["trG"][:, 1]))
                em, Vm = eigh64(Hm)
                em, Vm = em.flip(0), Vm.flip(1)
                eh = eigh64(H)[0].flip(0)
                r = r_of(em)
                V = Vm[:, :r]
                P = torch.eye(H.shape[0], device=dev, dtype=torch.float64) - V @ V.T
                HP = P @ H.double() @ P
                ehp = eigh64(HP)[0].flip(0)
                old = 0.01 * H.diagonal().mean()
                new = 0.01 * HP.diagonal().mean()
                ep, Vp = eigh64(d["Pp"])
                u1 = Vp[:, -1]
                cos = torch.cat(d["cos"])
                tot = em.clamp_min(0).sum()
                key = f"b{l}.{ln}"
                res["layers"][key] = {
                    "block": l, "layer": ln, "d_in": int(H.shape[0]),
                    "mean_energy_share": float(Hm.trace() / H.trace()), "mean_energy_share_per_bucket": share_b.tolist(),
                    "top1_frac": float(em[0] / tot), "top4_frac": float(em[:4].clamp_min(0).sum() / tot),
                    "top8_frac": float(em[:8].clamp_min(0).sum() / tot), "r": r,
                    "erased_old": int((eh < old).sum()), "erased_new": int((ehp < new).sum()) - r,
                    "cos_xbar_p_median": float(cos.median()), "cos_xbar_p_p10": float(torch.quantile(cos, 0.1)),
                    "abs_cos_v1_u1": float((Vm[:, 0] @ u1).abs())}
                if l in SPECTRA_BLOCKS:
                    spectra[key] = {"eig_H": eh.cpu().tolist(), "eig_Hvar": eigh64(Hv)[0].flip(0).cpu().tolist(),
                                    "damp_old": float(old), "damp_new": float(new)}
                print(key, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res["layers"][key].items()
                            if k not in ("mean_energy_share_per_bucket", "block", "layer")}, flush=True)
            del acc
            torch.cuda.empty_cache()
        res.update(run.record())
    rs = sorted(v["r"] for v in res["layers"].values())
    r_star = max(1, rs[(len(rs) - 1) // 2])
    atomic_write_json(os.path.join(ROOT, "configs", "deflate_r.json"),
                      {"r_per_layer": {k: v["r"] for k, v in res["layers"].items()}, "r_star": r_star,
                       "rule": "r_l = smallest r with top-r eig(H_mean) >= 95% of tr(H_mean), cap 8; r* = lower median, >= 1",
                       "source": "results/deflate/g1.json (FP-propagated inputs)"})
    res["r_star"] = r_star
    atomic_write_json(os.path.join(OUT, "g1.json"), res)
    atomic_write_json(os.path.join(OUT, "g1_spectra.json"), spectra)
    print("r* =", r_star)


if __name__ == "__main__":
    main()
