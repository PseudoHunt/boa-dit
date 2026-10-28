"""Consolidated §5 (rule_distribution.md): memory engineering of the DiT static-A8 feedback state.
  rank : per-site coherent contribution C = ||Σ_t e_t||² / T of the single table (e = q̂ − x), calibration seeds 1000-1007
  eta  : η re-measured (fp16 vs INT4 state), seeds 0-31: all-site Σ||η||²/Σ||e||² and ||Ση||/√Σ||e||² on the identity sites
  eval : latent rel. MSE on seeds 0-127 for INT4 state and the site budget (full fp16 state on top-C sites, DCT-4 elsewhere)
Run: python -m boa_dit.temporal.mem rank|eta|eval
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample, sample_seeds
from boa_dit.temporal.c3 import STAGE, guard
from boa_dit.temporal.common import CK, OUT, boot_reduction
from boa_dit.temporal.dit_act import Quantize

MD = os.path.join(OUT, "mem")
IDENT = ["b13.mlp.fc2", "b13.attn.qkv", "b13.attn.proj"]
CAL = list(range(1000, 1008))                                                # 8 seeds: per-site Σe buffers fit in memory
N_EVAL = 128
BUDGET = 0.25                                                                  # full-state sites hold <= 25% of full-feedback bytes


class Rank(Quantize):
    """single table; accumulates Σ_t e and Σ_t ||e||² per site (e = consumed − x, original space)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.se, self.ee = {}, {}

    def _fn(self, key, x, mod):
        y = super()._fn(key, x, mod)
        e = y.float() - x.float()
        if CTX.step == 0 or key not in self.se:
            self.se[key] = torch.zeros_like(e)
        self.se[key] += e
        self.ee[key] = self.ee.get(key, 0.0) + float(e.double().pow(2).sum())
        return y


class Eta(Quantize):
    """feedback with η bookkeeping: η = s − r⁰, e = q̂ − x (quantizer space)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.acc = {"eta2": 0.0, "e2": 0.0, "r02": 0.0}
        self.vec = {}

    def fb_step(self, key, t, xs, D, Z, qm, lo, s_c=None):
        st = super().fb_step(key, t, xs, D, Z, qm, lo, s_c)
        eta, e = st["s"] - st["r0"], st["qh"] - xs                           # η = s − r⁰ (storage + clamp)
        self.acc["eta2"] += float(eta.double().pow(2).sum()); self.acc["e2"] += float(e.double().pow(2).sum())
        self.acc["r02"] += float(st["r0"].double().pow(2).sum())
        if key in IDENT:
            v = self.vec.setdefault(key, {"seta": torch.zeros_like(eta), "e2": 0.0})
            v["seta"] += eta; v["e2"] += float(e.double().pow(2).sum())
        return st


@torch.no_grad()
def rank(dev):
    if not guard(0.05, "mem rank"):
        raise SystemExit(5)
    with Run(STAGE, "mem_rank_sites", dev, {"seeds": [CAL[0], CAL[-1]]}) as run:
        m, d = load_dit(dev), make_diffusion(50)
        q = Rank(m, "single8", "none", dev, log=False)
        ddim_sample(m, d, noise_for_seeds(CAL, dev), labels_for_seeds(CAL), 1.5)
        C = {k: float(v.double().pow(2).sum()) / 50 for k, v in q.se.items()}
        q.remove()
        # full-feedback state bytes per site at this batch (fp16, rows = 2 x 32, shape of the input)
        fb = Quantize(m, "single8", "none", dev, feedback=True, log=False)
        ddim_sample(m, d, noise_for_seeds(CAL[:1], dev), labels_for_seeds(CAL[:1]), 1.5)  # one seed: allocate per-site state
        B = {k: v.numel() * v.element_size() for k, v in fb.r.items()}
        fb.remove()
        res = {"C": C, "E": q.ee, "state_bytes_per_image_full": B, "seeds": [CAL[0], CAL[-1]]}
        order = sorted(C, key=lambda k: -C[k])
        tot, cum, full = sum(B.values()), 0, []
        for k in order:
            if cum + B[k] > BUDGET * tot:
                break
            full.append(k); cum += B[k]
        res.update({"order": order, "full_sites": full, "full_bytes_frac": cum / tot,
                    "C_frac_covered": sum(C[k] for k in full) / sum(C.values())})
        res.update(run.record())
    os.makedirs(MD, exist_ok=True)
    atomic_write_json(os.path.join(MD, "rank.json"), res)
    print(f"top-C sites under the {BUDGET:.0%} byte budget: {len(full)} sites, {cum / tot:.3f} of bytes, {res['C_frac_covered']:.3f} of C; first 8 {full[:8]}", flush=True)


@torch.no_grad()
def eta(dev):
    if not guard(0.06, "mem eta"):
        raise SystemExit(5)
    seeds = list(range(32))
    out = {}
    with Run(STAGE, "mem_eta_fp16_int4", dev, {"seeds": [0, 31]}) as run:
        m, d = load_dit(dev), make_diffusion(50)
        for bits in (16, 4):
            q = Eta(m, "single8", "none", dev, feedback=True, log=False, state_bits=bits)
            ddim_sample(m, d, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), 1.5)
            a = q.acc
            out[f"state{bits}"] = {"eta2_over_e2": a["eta2"] / a["e2"], "eta2_over_r02": a["eta2"] / a["r02"],
                                   "ident": {k: float(v["seta"].double().norm()) / math.sqrt(v["e2"]) for k, v in q.vec.items()}}
            q.remove()
            del q
            torch.cuda.empty_cache()
            print(f"  state {bits}-bit: Σ||η||²/Σ||e||² {out[f'state{bits}']['eta2_over_e2']:.3e}, identity sites ||Ση||/√Σ||e||² "
                  f"{ {k: round(v, 4) for k, v in out[f'state{bits}']['ident'].items()} }", flush=True)
        out.update(run.record())
    atomic_write_json(os.path.join(MD, "eta.json"), out)


@torch.no_grad()
def eval_(dev):
    rk = json.load(open(os.path.join(MD, "rank.json")))
    xf, _ = fp_reference(dev)
    xf = xf[:N_EVAL]
    L = lambda n: json.load(open(os.path.join(OUT, "eval", f"dit_bp_single8_{n}.json")))["rel_mse_per_image"][:N_EVAL]
    single, full = L("none"), L("feedback")
    res = {"seeds": [0, N_EVAL - 1], "single_rel_mse": float(np.mean(single)), "full_fp16_rel_mse": float(np.mean(full))}
    for arm, kw in (("int4", dict(state_bits=4)), ("budget", dict(compress="dct4", full_sites=set(rk["full_sites"]))),
                    ("dct4", dict(compress="dct4"))):
        f = os.path.join(MD, f"eval_{arm}.json")
        if os.path.exists(f):
            res[arm] = json.load(open(f)); continue
        if not guard(0.12, f"mem eval {arm}"):
            res[arm] = "UNRUN (budget)"; continue
        with Run(STAGE, f"mem_eval_{arm}", dev, {"n": N_EVAL}) as run:
            m, d = load_dit(dev), make_diffusion(50)
            q = Quantize(m, "single8", "none", dev, feedback=True, log=False, **kw)
            part = os.path.join(CK, f"mem_{arm}.partial")
            xq = sample_seeds(m, d, list(range(N_EVAL)), 1.5, 32, partial=part, resume=True)
            rel, _, _ = per_image(xq, xf)
            ddim_sample(m, d, noise_for_seeds([0], dev), labels_for_seeds([0]), 1.5)   # one image: state bytes per image
            r = {"rel_mse_mean": float(rel.mean()), "rel_mse_per_image": rel.tolist(), "state_bytes_per_image": q.state_bytes(),
                 "vs_single": boot_reduction(rel.tolist(), single), "vs_full_fp16": boot_reduction(rel.tolist(), full)}
            q.remove()
            r.update(run.record())
        atomic_write_json(f, r)
        os.remove(part)
        res[arm] = r
        print(f"  {arm}: latent {r['rel_mse_mean']:.5f}, state {r['state_bytes_per_image'] / 2**20:.2f} MiB/image, vs single "
              f"{-100 * r['vs_single']['reduction']:+.1f}% [{-100 * r['vs_single']['ci'][1]:+.1f}, {-100 * r['vs_single']['ci'][0]:+.1f}], vs full fp16 "
              f"{-100 * r['vs_full_fp16']['reduction']:+.1f}% [{-100 * r['vs_full_fp16']['ci'][1]:+.1f}, {-100 * r['vs_full_fp16']['ci'][0]:+.1f}]", flush=True)
    res["full_fp16_state_bytes_per_image"] = sum(rk["state_bytes_per_image_full"].values())
    atomic_write_json(os.path.join(MD, "eval.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rank", "eta", "eval"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(MD, exist_ok=True)
    {"rank": lambda: rank(dev), "eta": lambda: eta(dev), "eval": lambda: eval_(dev)}[a.cmd]()


if __name__ == "__main__":
    main()
