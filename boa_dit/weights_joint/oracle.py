"""Part 0 (PROTOCOL.md): weight-space cancellation oracle, DiT-XL/2, fp32 throughout (no autocast).
W_plus = heig-8's effective weights (block linears: W_q + δVᵀ with the carry folded in; adaLN: W8 RTN); W_minus = 2W - W_plus
(fp64 compute, fp32 store; unquantized, unclamped: an oracle, not a deployable W4 method). Seeds 4000-4015.
  same-state: e_±,t = f(W_±, x_t) - f(W, x_t) on the fp32 FP trajectory (guided 4 eps channels)
  live: static W_plus, static W_minus, alternating +/- and -/+ (by denoiser call index; same table for both CFG halves)
Run: python -m boa_dit.weights_joint.oracle
"""
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, Run, atomic_write_json
from boa_dit.model import block_linear, labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample
from boa_dit.weights_joint.common import (CK, HEIG, ORACLE, OUT, STAGE, boot_reduction, fp_traj, guard, rel_mse,
                                          step_outputs)


@torch.no_grad()
def build(dev):
    """{param name: (W_fp, W_plus, W_minus)} for every quantized weight (112 block linears + 28 adaLN)."""
    fp = load_dit("cpu")
    q = load_quant_state(load_dit("cpu"), os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    carry = torch.load(os.path.join(CKPT_DIR, "temporal", "rebuild_quant", HEIG, "carry.pt"), map_location="cpu")
    fsd, qsd = fp.state_dict(), q.state_dict()
    out, sym = {}, []
    for name, w in fsd.items():
        if not name.endswith("weight") or torch.equal(w, qsd[name]):
            continue                                                          # unquantized parameter
        wp = qsd[name].double()
        key = None
        parts = name.split(".")
        if parts[0] == "blocks" and ".".join(parts[2:-1]) in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            key = f"b{parts[1]}.{'.'.join(parts[2:-1])}"
            c = carry[key]
            wp = wp + c["delta"].double() @ c["V"].double().T              # carry folded in: y = W_q x + δ(Vᵀx)
        wm = 2 * w.double() - wp
        dp, dm = wp.float() - w, wm.float() - w                               # as stored (fp32)
        sym.append((float((dp + dm).double().norm()), float(dp.double().norm())))
        out[name] = (w.float(), wp.float(), wm.float())
    disc = sum(a for a, _ in sym) / sum(b for _, b in sym)
    return out, {"n_params_tensors": len(out), "symmetry_discrepancy_rel": disc,
                 "max_layer_discrepancy_rel": max(a / b for a, b in sym)}


class Swap:
    """Pre-hooks setting each quantized weight to W_plus / W_minus by denoiser-call parity (pattern) or statically."""

    def __init__(self, model, W, pattern):
        self.hs = []
        mods = dict(model.named_modules())
        for name, (wf, wp, wm) in W.items():
            mod = mods[name.rsplit(".", 1)[0]]
            tabs = {"+": wp.to(next(model.parameters()).device), "-": wm.to(next(model.parameters()).device),
                    "0": wf.to(next(model.parameters()).device)}
            seq = [tabs[c] for c in pattern]

            def pre(m, inp, seq=seq):
                k = 0 if CTX.step is None else CTX.step % len(seq)
                m.weight.data = seq[k]
            self.hs.append(mod.register_forward_pre_hook(pre))

    def remove(self):
        for h in self.hs:
            h.remove()


@torch.no_grad()
def main():
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    guard(0.15, "oracle")
    os.makedirs(OUT, exist_ok=True)
    res = {"seeds": ORACLE, "precision": "fp32 (no autocast)", "source": "heig-8 W4 (+carry) + W8 adaLN, high-precision activations"}
    with Run(STAGE, "oracle", dev, {"seeds": ORACLE}) as run:
        W, res["symmetry"] = build(dev)
        print("symmetry", res["symmetry"], flush=True)
        tr = fp_traj(dev, ORACLE, False, "oracle_fp32")
        m, d = load_dit(dev), make_diffusion(50)
        labels = labels_for_seeds(ORACLE)
        # same-state diagnostic
        ss = {"e_plus": [], "e_minus": [], "cos": [], "mean_energy": []}
        for sign in ("+", "-"):
            sw = Swap(m, W, sign)
            ss["out" + sign] = {i: step_outputs(m, d, tr["x"][i].to(dev), labels, i, amp=False)[:, :4].cpu() for i in range(50)}
            sw.remove()
        for i in range(50):
            ep, em = ss["out+"][i] - tr["ref"][i], ss["out-"][i] - tr["ref"][i]
            a, b = ep.flatten(1).double(), em.flatten(1).double()
            ss["e_plus"].append(float(a.pow(2).sum(1).mean())); ss["e_minus"].append(float(b.pow(2).sum(1).mean()))
            ss["cos"].append(float(((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1))).mean()))
            ss["mean_energy"].append(float(((a + b) / 2).pow(2).sum(1).mean()))
        del ss["out+"], ss["out-"]
        ep, em, mn = map(np.array, (ss["e_plus"], ss["e_minus"], ss["mean_energy"]))
        ss["mean_over_individual_per_step"] = (mn / ((ep + em) / 2)).tolist()
        ss["overall"] = {"energy_plus": float(ep.sum()), "energy_minus": float(em.sum()), "energy_mean": float(mn.sum()),
                         "mean_over_individual": float(mn.sum() / ((ep.sum() + em.sum()) / 2)), "cos_mean_over_steps": float(np.mean(ss["cos"]))}
        res["same_state"] = ss
        # live trajectories
        res["live"] = {}
        for pat, lab in (("+", "static_plus"), ("-", "static_minus"), ("+-", "alt_plus_minus"), ("-+", "alt_minus_plus")):
            sw = Swap(m, W, pat)
            x0 = ddim_sample(m, d, noise_for_seeds(ORACLE, dev), labels, 1.5, amp=False).cpu()
            sw.remove()
            r = rel_mse(x0, tr["x0"])
            res["live"][lab] = {"rel_mse_mean": float(r.mean()), "rel_mse_per_seed": r.tolist(),
                                "abs_err_per_seed": (x0 - tr["x0"]).flatten(1).pow(2).sum(1).double().tolist()}
            print(lab, f"{float(r.mean()):.6f}", flush=True)
        L = res["live"]
        for alt in ("alt_plus_minus", "alt_minus_plus"):
            for st in ("static_plus", "static_minus"):
                L[alt][f"vs_{st}"] = boot_reduction(L[alt]["rel_mse_per_seed"], L[st]["rel_mse_per_seed"])
        sp = np.array(L["static_plus"]["abs_err_per_seed"])
        L["ratio_alt_pm_over_static_plus_abs_err"] = float(np.mean(L["alt_plus_minus"]["abs_err_per_seed"]) / sp.mean())
        res["stage1_reference"] = {"what": "weight-induced prediction-error replay, sign-alternated / recorded (seeds 0-31, fp16 autocast)",
                                   "ratio": 0.002257, "ci": [0.000801, 0.004745], "settings_match": False}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "oracle.json"), res)
    print({k: v for k, v in res["same_state"]["overall"].items()}, flush=True)
    print({k: (v["rel_mse_mean"] if isinstance(v, dict) else v) for k, v in res["live"].items()}, flush=True)


if __name__ == "__main__":
    main()
