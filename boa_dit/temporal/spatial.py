"""Consolidated §1 (rule_spatial.md): spatial structure of the W4A4 activation rounding error on PixArt-Σ, with A8 dynamic
as a side computation on the same inputs. Trajectory = the unchanged W4A4 base; design prompts 0-7; even steps 0..18.
Run: python -m boa_dit.temporal.spatial diag
"""
import argparse
import os

import torch
import torch.nn.functional as F

from boa_dit.common import Run, atomic_write_json
from boa_dit.objective.d2 import band_masks
from boa_dit.temporal import pixart as PX
from boa_dit.temporal.c3 import STAGE, guard, spent
from boa_dit.temporal.common import OUT
from boa_dit.temporal.ext_w4a4 import QLinearFB, _groups, install
from boa_dit.w4a4 import common as WC

SP = os.path.join(OUT, "spatial")
G = 64                                                                        # token grid side (1024² / 8 / patch 2)
FIELDS = ["n", "E", "Eout", "hxy", "hxx", "hyy", "vxy", "vxx", "vyy", "hxy_u", "hxx_u", "hyy_u", "vxy_u", "vxx_u", "vyy_u",
          "xh_xy", "xh_xx", "xh_yy", "same_h", "same_v", "b1", "b2", "b3", "b4", "dc"]


def quant(xs, shift, bits):
    """dynamic per-token group quantizer of the base (bits=4) or its 8-bit analogue; returns (xq, codes-space input, Δ)."""
    if shift is not None:
        qmax = 2 ** bits - 1
        uc = (xs + shift).clamp_min(0)
        g, sh = _groups(uc)
        sc = g.amax(-1, keepdim=True).clamp_min(1e-8) / qmax
        q = torch.clamp(torch.round(g / sc), 0, qmax) * sc
        return q.reshape(sh) - shift, uc, sc.expand(g.shape).reshape(sh)
    qmax, qmin = 2 ** (bits - 1) - 1, -(2 ** (bits - 1))
    g, sh = _groups(xs)
    sc = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / qmax
    q = torch.clamp(torch.round(g / sc), qmin, qmax) * sc
    return q.reshape(sh), xs, sc.expand(g.shape).reshape(sh)


def corr_terms(a):
    """a [N, G, G, C]: horizontal / vertical neighbour cross-products and energies."""
    h0, h1, v0, v1 = a[:, :, :-1], a[:, :, 1:], a[:, :-1], a[:, 1:]
    S = lambda t: t.sum(dtype=torch.float64)
    return [S(h0 * h1), S(h0 * h0), S(h1 * h1), S(v0 * v1), S(v0 * v0), S(v1 * v1)]


class Acc:
    def __init__(self, dev):
        self.m = {k: band_masks(G)[i].to(dev) for i, k in enumerate(["b1", "b2", "b3", "b4"])}
        self.s = {}

    def add(self, key, xs, xq, uc, sc, R):
        N, T, C = xs.shape
        e = xq - xs
        st = self.s.setdefault(key, torch.zeros(len(FIELDS), dtype=torch.float64, device=xs.device))
        out = F.linear(e, R.float())
        eg = e.reshape(N, G, G, C)
        ec = eg - eg.mean((1, 2), keepdim=True)
        xg = xs.reshape(N, G, G, C)
        xc = xg - xg.mean((1, 2), keepdim=True)
        code = lambda num, den: torch.round(num / den)
        ug, sg = uc.reshape(N, G, G, C), sc.reshape(N, G, G, C)
        same_h = (code(ug[:, :, :-1], sg[:, :, :-1]) == code(ug[:, :, 1:], sg[:, :, :-1])).sum(dtype=torch.float64)
        same_v = (code(ug[:, :-1], sg[:, :-1]) == code(ug[:, 1:], sg[:, :-1])).sum(dtype=torch.float64)
        P = torch.fft.fft2(eg.permute(0, 3, 1, 2)).abs().pow(2).sum((0, 1), dtype=torch.float64)  # [G, G]
        xh = corr_terms(xc)
        vals = [float(N * T * C), e.pow(2).sum(dtype=torch.float64), out.pow(2).sum(dtype=torch.float64)] + corr_terms(ec) + corr_terms(eg) + [xh[0], xh[1], xh[2], same_h, same_v] \
            + [P[self.m[k]].sum() for k in ("b1", "b2", "b3", "b4")] + [P[0, 0]]
        st += torch.stack([torch.as_tensor(v, dtype=torch.float64, device=xs.device) for v in vals])


class QLinearSpatial(QLinearFB):
    """base W4A4 output, unchanged; error statistics at A4 and A8 on even steps for 4096-token inputs."""
    acc, name = None, None

    def forward(self, x):
        xs = x.float() if self.d is None else x.float() / self.d
        xq, uc, sc = quant(xs, self.act_shift, 4)
        t = PX.STATE.step
        if self.acc is not None and xs.shape[1] == G * G and t % 2 == 0 and t <= 18:
            self.acc.add((self.name, 4), xs, xq, uc, sc, self.R)
            q8, u8, s8 = quant(xs, self.act_shift, 8)
            self.acc.add((self.name, 8), xs, q8, u8, s8, self.R)
        y = F.linear(xq.to(torch.bfloat16), self.R, self.bias)
        if self.L1 is not None and self.L1.shape[1] > 0:
            y = y + (xs.to(torch.bfloat16) @ self.L1) @ self.L2
        return y.to(x.dtype)


def summarize(S):
    def one(v):
        d = dict(zip(FIELDS, v.tolist()))
        c = lambda xy, xx, yy: d[xy] / max((d[xx] * d[yy]) ** 0.5, 1e-30)
        bt = sum(d[b] for b in ("b1", "b2", "b3", "b4"))
        tok_h = d["n"] * (G - 1) / G
        return {"E": d["E"], "Eout": d["Eout"],
                "corr_h": c("hxy", "hxx", "hyy"), "corr_v": c("vxy", "vxx", "vyy"),
                "corr_h_uncentred": c("hxy_u", "hxx_u", "hyy_u"), "corr_v_uncentred": c("vxy_u", "vxx_u", "vyy_u"),
                "x_corr_h": c("xh_xy", "xh_xx", "xh_yy"),
                "same_cell_h": d["same_h"] / tok_h, "same_cell_v": d["same_v"] / tok_h,
                "bands": [d[b] / bt for b in ("b1", "b2", "b3", "b4")], "dc_share": d["dc"] / bt}
    res = {}
    for bits in (4, 8):
        rows = {k[0]: one(v.cpu()) for k, v in S.items() if k[1] == bits}
        tot = sum(r["Eout"] for r in rows.values())
        for r in rows.values():
            r["share"] = r["Eout"] / tot
            r["corr_mean"] = (r["corr_h"] + r["corr_v"]) / 2
        order = sorted(rows, key=lambda k: -rows[k]["share"])
        cum, top = 0.0, []
        for k in order:
            top.append(k); cum += rows[k]["share"]
            if cum >= 0.5:
                break
        res[f"A{bits}"] = {"types": rows, "order": order, "half_set": top,
                           "a_min_corr_in_half_set": min(rows[k]["corr_mean"] for k in top),
                           "b_weighted_band1": sum(r["share"] * r["bands"][0] for r in rows.values())}
    a = res["A4"]
    res["read_S"] = {"a": a["a_min_corr_in_half_set"] >= 0.3, "b": a["b_weighted_band1"] >= 0.4}
    res["read_S"]["pass"] = res["read_S"]["a"] or res["read_S"]["b"]
    return res


@torch.no_grad()
def diag(dev, prompts=range(8), bs=2):
    if not guard(0.2, "spatial diag"):
        return
    os.makedirs(SP, exist_ok=True)
    with Run(STAGE, "spatial_diag_w4a4", dev, {"prompts": list(prompts), "steps": "0,2,..,18"}) as run:
        pipe = WC.load_pipe(dev)
        mods = install(pipe.transformer, "base")
        acc = Acc(dev)
        names = {}
        for m in mods:                                                        # swap class in place: same buffers
            m.__class__ = QLinearSpatial
        for (blk_i, blk) in enumerate(pipe.transformer.transformer_blocks):
            for n, sub in blk.named_modules():
                if isinstance(sub, QLinearSpatial):
                    sub.acc, sub.name = acc, n.replace(".0", "").replace("ff.net.proj", "ff.fc1").replace("ff.net.2", "ff.fc2")
                    names[sub.name] = names.get(sub.name, 0) + 1
        print("  sites per type:", names, flush=True)
        e = WC.load_embeds("design")
        P = list(prompts)
        for i in range(0, len(P), bs):
            sc = PX.StepCounter(pipe.transformer)
            try:
                PX.sample_lat(pipe, e, P[i:i + bs], dev)
            finally:
                sc.remove()
            print(f"  prompts {P[i]}-{P[i + bs - 1]} done ({spent():.3f} h c3 spent before this job)", flush=True)
        res = summarize(acc.s)
        res.update({"prompts": P, "steps": list(range(0, 20, 2)), "base": "hsvd_r4_b0.5_kv16_u2_rs"})
        res.update(run.record())
    atomic_write_json(os.path.join(SP, "diag.json"), res)
    for bits in ("A4", "A8"):
        r = res[bits]
        print(f"{bits}: half-set {r['half_set']}, min centred corr {r['a_min_corr_in_half_set']:.3f}, weighted band-1 {r['b_weighted_band1']:.3f}")
        for k in r["order"]:
            t = r["types"][k]
            print(f"   {k:16s} share {t['share']:.3f} corr h/v {t['corr_h']:.3f}/{t['corr_v']:.3f} (unc {t['corr_h_uncentred']:.3f}) "
                  f"x-corr {t['x_corr_h']:.3f} same-cell {t['same_cell_h']:.3f}/{t['same_cell_v']:.3f} bands {[round(b, 3) for b in t['bands']]} dc {t['dc_share']:.4f}")
    print("Read S:", res["read_S"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["diag"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    diag(dev)


if __name__ == "__main__":
    main()
