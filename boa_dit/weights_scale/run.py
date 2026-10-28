"""Fixed-code scale modulation of heig-8 W4 (results/temporal/weights_scale/PROTOCOL.md).
Each block linear computes y_t = c_{l,t}·(x W_q^T) + b + δ(V^T x); the rank-8 carry (high-precision branch) is not scaled;
c = 1 leaves the layer bit-identical to heig-8. Commands:
  validate : c≡1 bit-exact; output scaling == weight scaling (fp32); symmetric-average identity
  refit    : static c*_l on calibration seeds 1000-1003 (heig-8 trajectory), range [0.90, 1.10]
  fit      : endpoint finite differences (seeds 5000-5007), bounded ridge fit per pattern, λ on 5008-5023, freeze
  eval     : six arms on the reserved seeds 4100-4227 (one-time use, after the freeze)
Run: python -m boa_dit.weights_scale.run validate|refit|fit|eval
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import LINEARS, block_linear, labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample
from boa_dit.weights_joint.common import boot_reduction, diagnostics, fp_traj, rel_mse

STAGE, CAP_H = "weights_scale", 1.0
OUT = os.path.join(RESULTS, "temporal", "weights_scale")
HEIG = "deflate_heig_r8_w4"
CARRY = os.path.join(CKPT_DIR, "temporal", "rebuild_quant", HEIG, "carry.pt")
CAL_REFIT, CAL_FIT, VAL, EVAL = list(range(1000, 1004)), list(range(5000, 5008)), list(range(5008, 5024)), list(range(4100, 4228))
T, EPS, ABOUND, CRANGE = 50, 0.005, 0.02, (0.90, 1.10)
PATTERNS = {"constant": [1.0] * T, "slow": [-1.0 if t < 25 else 1.0 for t in range(T)], "alternating": [(-1.0) ** t for t in range(T)]}
SHUF = (lambda p: [1.0 if p[t] < 25 else -1.0 for t in range(T)])(torch.randperm(T, generator=torch.Generator().manual_seed(0)).tolist())
KEYS = [f"b{l}.{ln}" for l in range(28) for ln in LINEARS]


def spent():
    with open(os.path.join(RESULTS, "gpu_hours.csv")) as f:
        return sum(float(r[5]) for r in csv.reader(f) if r and r[0] == STAGE)


def guard(est, what):
    s = spent()
    print(f"  budget: {s:.3f} h + {est:.3f} h vs cap {CAP_H} ({what})", flush=True)
    if s + est > CAP_H:
        print(f"  UNRUN: {what}", flush=True)
        sys.exit(5)


class Scaler:
    """Output hooks on the 112 block linears: c_t·(out - b) + b + carry; c = 1 -> out + carry (exactly heig-8's carry path)."""

    def __init__(self, model, sched=None):
        dev = next(model.parameters()).device
        car = torch.load(CARRY, map_location="cpu")
        self.sched = sched or {}
        self.hs = []
        for key in KEYS:
            l, ln = key.split(".", 1)
            lin = block_linear(model.blocks[int(l[1:])], ln)
            V, D = car[key]["V"].to(dev), car[key]["delta"].to(dev)
            self.hs.append(lin.register_forward_hook(self._hook(key, V, D)))

    def _hook(self, key, V, D):
        def hook(mod, inp, out):
            x = inp[0]
            dt = out.dtype
            carry = ((x.to(V.dtype) @ V) @ D.T).to(dt) if dt == torch.float16 else (x.float() @ V.float()) @ D.float().T
            s = self.sched.get(key)
            c = 1.0 if s is None or CTX.step is None else s[CTX.step]
            if c == 1.0:
                return out + carry
            b = mod.bias.float()
            return (((out.float() - b) * c + b) + carry.float()).to(dt)
        return hook

    def remove(self):
        for h in self.hs:
            h.remove()


def schedule(cstar, a, phi):
    """{key: [c_{l,t}]}: c*_l·(1 + a_f·φ(t)); a = {family: amplitude} (missing = 0)."""
    return {k: [cstar[k] * (1.0 + a.get(k.split(".", 1)[1], 0.0) * phi[t]) for t in range(T)] for k in KEYS}


def heig_model(dev):
    m = load_dit(dev)
    load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    return m


@torch.no_grad()
def sample(m, d, seeds, sched, dev):
    sc = Scaler(m, sched)
    x0 = ddim_sample(m, d, noise_for_seeds(seeds, dev), labels_for_seeds(seeds), 1.5).cpu()
    sc.remove()
    return x0


@torch.no_grad()
def validate(dev):
    res = {}
    m, d = heig_model(dev), make_diffusion(50)
    s = CAL_FIT[:2]
    from boa_dit.deflate.carry import install
    hs = install(m, torch.load(CARRY, map_location="cpu"))
    ref = ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5).cpu()
    for h in hs:
        h.remove()
    res["c1_bitexact_vs_heig8"] = bool(torch.equal(sample(m, d, s, {k: [1.0] * T for k in KEYS}, dev), ref))
    lin = block_linear(m.blocks[13], "mlp.fc1")
    x = torch.randn(64, lin.in_features, device=dev)
    c = 1.013
    y_out = c * (F.linear(x, lin.weight) ) + lin.bias
    y_w = F.linear(x, c * lin.weight) + lin.bias
    res["output_vs_weight_scaling_max_rel"] = float((y_out - y_w).abs().max() / y_w.abs().max())
    a = 0.01
    yp, ym, y1 = (1 + a) * F.linear(x, lin.weight), (1 - a) * F.linear(x, lin.weight), F.linear(x, lin.weight)
    res["symmetric_average_max_rel"] = float(((yp + ym) / 2 - y1).abs().max() / y1.abs().max())
    res["all_pass"] = res["c1_bitexact_vs_heig8"] and res["output_vs_weight_scaling_max_rel"] < 1e-5 and res["symmetric_average_max_rel"] < 1e-6
    atomic_write_json(os.path.join(OUT, "validation.json"), res)
    print(res, flush=True)
    if not res["all_pass"]:
        raise SystemExit(6)


@torch.no_grad()
def refit(dev):
    guard(0.05, "refit")
    m, d = heig_model(dev), make_diffusion(50)
    fp = load_dit("cpu")
    car = torch.load(CARRY, map_location="cpu")
    acc = {k: [0.0, 0.0] for k in KEYS}
    hs = []
    for key in KEYS:
        l, ln = key.split(".", 1)
        lin = block_linear(m.blocks[int(l[1:])], ln)
        Wf = block_linear(fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
        V, D = car[key]["V"].to(dev).float(), car[key]["delta"].to(dev).float()

        def hook(mod, inp, out, key=key, Wf=Wf, V=V, D=D):
            x = inp[0].float()
            yq = F.linear(x, mod.weight.float())                              # bias-free quantized output
            target = F.linear(x, Wf) - (x @ V) @ D.T                          # FP output minus the unscaled carry
            acc[key][0] += float((yq.double() * target.double()).sum()); acc[key][1] += float(yq.double().pow(2).sum())
            return out + ((inp[0].to(torch.float16) @ V.half()) @ D.half().T).to(out.dtype) if out.dtype == torch.float16 else out + (x @ V) @ D.T
        hs.append(lin.register_forward_hook(hook))
    with Run(STAGE, "refit", dev, {"seeds": CAL_REFIT}) as run:
        ddim_sample(m, d, noise_for_seeds(CAL_REFIT, dev), labels_for_seeds(CAL_REFIT), 1.5)
        rec = run.record()
    raw = {k: acc[k][0] / acc[k][1] for k in KEYS}
    cstar = {k: float(min(max(v, CRANGE[0]), CRANGE[1])) for k, v in raw.items()}
    hits = [k for k, v in raw.items() if v < CRANGE[0] or v > CRANGE[1]]
    atomic_write_json(os.path.join(OUT, "refit.json"), {"cstar": cstar, "raw": raw, "range": CRANGE, "boundary_hits": hits,
                                                        "by_family_mean": {f: float(np.mean([cstar[k] for k in KEYS if k.endswith(f)])) for f in LINEARS}, **rec})
    print({f: round(float(np.mean([cstar[k] for k in KEYS if k.endswith(f)])), 4) for f in LINEARS}, "boundary hits", len(hits), flush=True)


def solve(e, R, lam):
    """min ||e + R a||^2 + lam ||a||^2, |a| <= ABOUND (bounded least squares, scipy)."""
    from scipy.optimize import lsq_linear
    A = np.vstack([R, np.sqrt(lam) * np.eye(R.shape[1])]) if lam > 0 else R
    b = np.concatenate([-e, np.zeros(R.shape[1])]) if lam > 0 else -e
    return lsq_linear(A, b, bounds=(-ABOUND, ABOUND)).x


@torch.no_grad()
def fit(dev):
    guard(0.25, "endpoint fit")
    cstar = json.load(open(os.path.join(OUT, "refit.json")))["cstar"]
    m, d = heig_model(dev), make_diffusion(50)
    res = {"eps": EPS, "abound": ABOUND, "patterns": {}}
    with Run(STAGE, "fit", dev, {"cal": CAL_FIT, "val": VAL}) as run:
        fpm = load_dit(dev)
        xf_cal = ddim_sample(fpm, d, noise_for_seeds(CAL_FIT, dev), labels_for_seeds(CAL_FIT), 1.5).cpu()
        xf_val = ddim_sample(fpm, d, noise_for_seeds(VAL, dev), labels_for_seeds(VAL), 1.5).cpu()
        del fpm
        torch.cuda.empty_cache()
        base = sample(m, d, CAL_FIT, schedule(cstar, {}, [0.0] * T), dev)
        e = (base - xf_cal).flatten().double().numpy()
        for pname, phi in PATTERNS.items():
            cols, curv = [], {}
            for f in LINEARS:
                xp = sample(m, d, CAL_FIT, schedule(cstar, {f: EPS}, phi), dev)
                xm = sample(m, d, CAL_FIT, schedule(cstar, {f: -EPS}, phi), dev)
                cols.append(((xp - xm) / (2 * EPS)).flatten().double().numpy())
                curv[f] = float((xp + xm - 2 * base).norm() / (xp - xm).norm())
                if pname == "alternating":                                    # FD resolution: R(2ε) vs R(ε)
                    xp2 = sample(m, d, CAL_FIT, schedule(cstar, {f: 2 * EPS}, phi), dev)
                    xm2 = sample(m, d, CAL_FIT, schedule(cstar, {f: -2 * EPS}, phi), dev)
                    r2 = ((xp2 - xm2) / (4 * EPS)).flatten().double().numpy()
                    curv[f + "_R2eps_vs_Reps_rel"] = float(np.linalg.norm(r2 - cols[-1]) / np.linalg.norm(cols[-1]))
            R = np.stack(cols, 1)
            scale = float(np.mean(np.diag(R.T @ R)))
            val = {}
            for lm in (0.0, 0.01, 0.1, 1.0):
                a = solve(e, R, lm * scale)
                av = {f: float(a[i]) for i, f in enumerate(LINEARS)}
                xv = sample(m, d, VAL, schedule(cstar, av, phi), dev)
                val[str(lm)] = {"a": av, "val_rel_mse": float(rel_mse(xv, xf_val).mean()),
                                "pred_linear_cal_err": float(np.linalg.norm(e + R @ a) ** 2), "boundary": [abs(x) >= ABOUND - 1e-9 for x in a]}
            best = min(val, key=lambda k: val[k]["val_rel_mse"])
            res["patterns"][pname] = {"R_norms": {f: float(np.linalg.norm(R[:, i])) for i, f in enumerate(LINEARS)}, "fd_checks": curv,
                                      "lambda_grid": val, "chosen_lambda_rel": float(best), "a": val[best]["a"]}
            print(pname, "λ", best, {k: round(v, 5) for k, v in val[best]["a"].items()}, "val", round(val[best]["val_rel_mse"], 5), "fd", {k: round(v, 3) for k, v in curv.items()}, flush=True)
        xb_val = sample(m, d, VAL, schedule(cstar, {}, [0.0] * T), dev)
        res["val_static_refit_rel_mse"] = float(rel_mse(xb_val, xf_val).mean())
        res["base_cal_err"] = float(np.linalg.norm(e) ** 2)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "fit.json"), res)
    frozen = {"cstar": cstar, "patterns": {p: res["patterns"][p]["a"] for p in PATTERNS}, "shuffled_schedule": SHUF,
              "shuffled_uses": "alternating amplitudes, not refit", "frozen_before_eval_seeds": True}
    atomic_write_json(os.path.join(OUT, "frozen.json"), frozen)


@torch.no_grad()
def evaluate(dev):
    guard(0.35, "evaluation (128 reserved seeds, 6 arms)")
    fr = json.load(open(os.path.join(OUT, "frozen.json")))
    cstar = fr["cstar"]
    arms = {"original": {k: [1.0] * T for k in KEYS}, "static_refit": schedule(cstar, {}, [0.0] * T),
            "constant": schedule(cstar, fr["patterns"]["constant"], PATTERNS["constant"]),
            "slow": schedule(cstar, fr["patterns"]["slow"], PATTERNS["slow"]),
            "alternating": schedule(cstar, fr["patterns"]["alternating"], PATTERNS["alternating"]),
            "shuffled": schedule(cstar, fr["patterns"]["alternating"], fr["shuffled_schedule"])}
    out = {"seeds": [EVAL[0], EVAL[-1]], "arms": {}}
    with Run(STAGE, "eval", dev, {"seeds": "4100-4227"}) as run:
        tr = fp_traj(dev, EVAL[:32], True, "scale_eval_fp16")
        fpm = load_dit(dev)
        xf = torch.cat([ddim_sample(fpm, make_diffusion(50), noise_for_seeds(EVAL[i:i + 32], dev), labels_for_seeds(EVAL[i:i + 32]), 1.5).cpu()
                        for i in range(0, len(EVAL), 32)])
        del fpm
        torch.cuda.empty_cache()
        m, d = heig_model(dev), make_diffusion(50)
        for name, sched in arms.items():
            sc = Scaler(m, sched)
            torch.cuda.synchronize(); t0 = time.time()
            x = torch.cat([ddim_sample(m, d, noise_for_seeds(EVAL[i:i + 32], dev), labels_for_seeds(EVAL[i:i + 32]), 1.5).cpu()
                           for i in range(0, len(EVAL), 32)])
            torch.cuda.synchronize(); ts = time.time() - t0
            dg = diagnostics(m, d, dev, tr)
            sc.remove()
            r = rel_mse(x, xf)
            out["arms"][name] = {"rel_mse_mean": float(r.mean()), "rel_mse_per_seed": r.tolist(), "sampling_s": ts, "diag": dg}
            print(f"  {name}: latent {float(r.mean()):.5f}, C {dg['C']:.4g}, C/I {dg['C_over_I']:.4f}, {ts:.1f} s", flush=True)
        A = out["arms"]
        for name in A:
            for ref in ("static_refit", "constant", "original"):
                if name != ref:
                    A[name][f"vs_{ref}"] = boot_reduction(A[name]["rel_mse_per_seed"], A[ref]["rel_mse_per_seed"])
        out["storage"] = {"static_multipliers": "112 x fp16 = 224 bytes (folded into the stored per-row scales)",
                          "amplitudes": "4 x fp16 per pattern = 8 bytes", "pattern": "fixed rule (constant/slow/alternating); shuffled: 50 bits",
                          "packed_weight_table": "unchanged (one W4 table)"}
        out.update(run.record())
    atomic_write_json(os.path.join(OUT, "eval.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["validate", "refit", "fit", "eval"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    {"validate": lambda: validate(dev), "refit": lambda: refit(dev), "fit": lambda: fit(dev), "eval": lambda: evaluate(dev)}[a.cmd]()


if __name__ == "__main__":
    main()
