"""Part A drivers (rule.md), DiT-XL/2 256, eval protocol of eval_latent_mse.py (DDIM-50, cfg 1.5, fp16 autocast, bs 32).
  setup : FP eval latents 0-511 (eval_latent_mse.fp_reference) + FP trajectory of seeds 0-31 (x_t at all 50 steps)
          + FP guided outputs on it (diagnostics reference)
  a0    : healing check, FP model, seeds 0-31, error injected into the 4 eps channels at every step, patterns i/ii/iii
  arm   : latent rel. MSE of an arm (step-indexed tables) on seeds [0, n) + diagnostics (per-step error, C/I, spectra)
Run: python -m boa_dit.temporal.dit setup | a0 | arm --arm b --n 512
"""
import argparse
import json
import math
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks, band_spectrum
from boa_dit.objective.proxy import step_outputs
from boa_dit.sample import ddim_sample, sample_seeds
from boa_dit.temporal.common import (CK, HEIG, LAYERS_FILE, OUT, PAIR_FILE, STAGE, StepTables, boot_ratio, boot_reduction,
                                     ci_split)

DIAG_SEEDS = list(range(32))
TRAJ = os.path.join(CK, "fp_traj32_all.pt")
SPEC_STEPS = tuple(range(20, 26))
STORED_A = os.path.join(RESULTS, "gate1", "eval", f"latent_{HEIG}.json")


@torch.no_grad()
def setup(dev):
    with Run(STAGE, "setup_dit_fp_refs", dev, {"what": "fp eval latents 0-511, fp traj 0-31 all steps, fp guided outputs"}):
        fp_reference(dev)
        if not os.path.exists(TRAJ):
            m, diffusion = load_dit(dev), make_diffusion(50)
            labels = labels_for_seeds(DIAG_SEEDS)
            x0, states = ddim_sample(m, diffusion, noise_for_seeds(DIAG_SEEDS, dev), labels, 1.5, record_states=True)
            xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:32]
            assert torch.equal(x0.cpu(), xf), "trajectory run does not reproduce the FP eval latents"
            ref = {i: step_outputs(m, diffusion, states[i], labels, i)[0][:, :4].cpu() for i in range(50)}
            atomic_torch_save({"x": {i: s.cpu() for i, s in states.items()}, "x0": x0.cpu(), "ref": ref}, TRAJ)


def energy_profile():
    d1 = json.load(open(os.path.join(RESULTS, "objective", "d1.json")))
    E = {int(k): v for k, v in d1["network"]["dg4_energy_per_sample_by_step"].items()}
    ks = sorted(E)
    out = []
    for i in range(50):
        if i >= ks[-1]:
            out.append(E[ks[-1]])
            continue
        lo = max(k for k in ks if k <= i)
        hi = min(k for k in ks if k > i)
        w = (i - lo) / (hi - lo)
        out.append(math.exp((1 - w) * math.log(E[lo]) + w * math.log(E[hi])))
    return out


@torch.no_grad()
def a0(dev):
    E = energy_profile()
    m, diffusion = load_dit(dev), make_diffusion(50)
    seeds = DIAG_SEEDS
    labels = labels_for_seeds(seeds)
    noise = noise_for_seeds(seeds, dev)
    shape = (len(seeds), 4, 32, 32)

    def unit(gen_seed):
        z = torch.stack([torch.randn(shape[1:], generator=torch.Generator().manual_seed(gen_seed + s)) for s in seeds])
        return z / z.flatten(1).norm(dim=1).view(-1, 1, 1, 1)

    u = unit(7000)
    res = {"energy_per_step": E, "seeds": seeds, "final_err": {}}
    with Run(STAGE, "a0_healing", dev, {"seeds": "0-31"}) as run:
        x_ref = ddim_sample(m, diffusion, noise, labels, 1.5).cpu()
        xf = torch.load(os.path.join(CKPT_DIR, "fp_eval_latents.pt"), map_location="cpu")[:32]
        res["ref_bitexact_vs_fp_eval"] = bool(torch.equal(x_ref, xf))
        for pat in ("i_constant", "ii_alternating", "iii_independent"):
            def hook(i, xx, out, pat=pat):
                if pat == "iii_independent":
                    e = unit(7000 + 100 * i)
                else:
                    e = u * (1 if (pat == "i_constant" or i % 2 == 0) else -1)
                out = out.clone()
                out[:, :4] += (e * math.sqrt(E[i])).to(out.device)
                return out
            x0 = ddim_sample(m, diffusion, noise, labels, 1.5, eps_hook=hook).cpu()
            res["final_err"][pat] = (x0 - x_ref).flatten(1).pow(2).sum(1).double().tolist()
            print(pat, f"{np.mean(res['final_err'][pat]):.4g}", flush=True)
        f = res["final_err"]
        r23 = boot_ratio(f["ii_alternating"], f["iii_independent"])
        r21 = boot_ratio(f["ii_alternating"], f["i_constant"])
        res["ratio_ii_over_iii"], res["ratio_ii_over_i"] = r23, r21
        res["read_healed"] = bool(abs(r23["ratio"] - 1) <= 0.20 and r21["ratio"] <= 0.50)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "a0.json"), res)
    print({k: v for k, v in res.items() if k not in ("final_err", "config", "energy_per_step")}, flush=True)


def arm_tables(arm):
    """-> (list of {key: (Wq, delta)}, V). Arms per rule.md A2/A3."""
    P = torch.load(PAIR_FILE, map_location="cpu")
    if arm in ("k3", "k4", "bu"):                                             # exploratory tables (pair_extra.py)
        X = torch.load(os.path.join(CK, "pair_extra.pt"), map_location="cpu")
        P = {k: {**p, **{n: v for n, v in X[k].items() if n.startswith(("c_", "d_"))}} for k, p in P.items()}
    V = {k: p["V"] for k, p in P.items()}
    w = lambda p, name: p["scale"] * (p["c_" + name].float() - p["zero"])     # BoA fake_quantize arithmetic

    def tab(name):
        return {k: (w(p, name), p["d_" + name]) for k, p in P.items()}
    if arm == "e":
        return [{k: ((w(p, "Wq1") + w(p, "Wq2")) / 2, ((p["d_Wq1"].float() + p["d_Wq2"].float()) / 2).half())
                 for k, p in P.items()}], V
    spec = {"a": ["Wq1"], "b": ["Wq1", "Wq2"], "c": ["Wq1", "Wq3"], "d": ["Wq2"], "f": ["Wq2", "Wq1"],
            "s8": ["Wq1", "Wq2s8"], "s16": ["Wq1", "Wq2s16"],
            "k3": ["Wq1", "Wq3", "Wq4"], "k4": ["Wq1", "Wq3", "Wq4", "Wq5"], "bu": ["Wq1", "Wq2u"]}[arm]
    return [tab(n) for n in spec], V


def load_arm(arm, dev):
    if arm in ("w5", "w6"):                                                   # single-table heig-8 recipe at 5 / 6 bits
        from boa_dit.deflate.carry import install
        name = f"deflate_heig_r8_w{arm[1]}"
        m = load_dit(dev)
        load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{name}", "model.pt"))
        return m, install(m, torch.load(os.path.join(CK, "rebuild_quant", name, "carry.pt"), map_location="cpu"))
    m = load_dit(dev)
    load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))      # W_q1 state incl. W8 adaLN
    tables, V = arm_tables(arm)
    return m, StepTables(m, tables, V)


@torch.no_grad()
def diagnostics(m, diffusion, dev):
    tr = torch.load(TRAJ, map_location="cpu")
    labels = labels_for_seeds(DIAG_SEEDS)
    masks = band_masks()
    errs = []
    for i in range(50):
        q = step_outputs(m, diffusion, tr["x"][i].to(dev), labels, i)[0][:, :4].cpu()
        errs.append(q - tr["ref"][i])
    E = torch.stack(errs, 1)                                                  # [32, 50, 4, 32, 32]
    spec = np.mean([band_spectrum(E[:, i], masks) for i in SPEC_STEPS], axis=0).tolist()
    return {"per_step_err": E.double().pow(2).flatten(2).sum(2).mean(0).tolist(), **ci_split(E),
            "band_spectrum_steps20_25": spec, "band1_share_steps20_25": spec[0]}


@torch.no_grad()
def arm(name, n, dev, check32=False):
    seeds = list(range(n))
    xf, _ = fp_reference(dev)
    xf = xf[:n]
    tag = "check32" if check32 else f"n{n}"
    with Run(STAGE, f"arm_{name}_{tag}", dev, {"arm": name, "n": n}) as run:
        if check32:                                                           # rebuilt heig-8 exactly as eval_latent_mse runs it
            from boa_dit.deflate.carry import install
            m = load_dit(dev)
            load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
            install(m, torch.load(os.path.join(CK, "rebuild_quant", HEIG, "carry.pt"), map_location="cpu"))
        else:
            m, st = load_arm(name, dev)
        diffusion = make_diffusion(50)
        part = os.path.join(CK, f"arm_{name}_{tag}.partial")
        xq = sample_seeds(m, diffusion, seeds, 1.5, 32, partial=part, resume=True)
        rel, mse, psnr = per_image(xq, xf)
        res = {"arm": name, "n": n, "rel_mse_mean": float(rel.mean()), "rel_mse_sem": float(rel.std() / math.sqrt(n)),
               "rel_mse_per_image": rel.tolist()}
        if check32:
            old = json.load(open(STORED_A))["rel_mse_per_image"][:n]
            res["stored_first_n"] = old
            res["bitexact_vs_stored"] = bool(all(a == b for a, b in zip(rel.tolist(), old)))
            res["max_abs_diff_vs_stored"] = float(max(abs(a - b) for a, b in zip(rel.tolist(), old)))
            res["latents_check"] = "rel MSE per image vs the stored 512-seed JSON's first 32"
        else:
            res["diag"] = diagnostics(m, diffusion, dev)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "eval", f"arm_{name}_{tag}.json"), res)
    os.remove(part)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["setup", "a0", "arm", "check32", "diag_only"])
    ap.add_argument("--arm", default="a")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    if a.cmd == "setup":
        setup(a.device)
    elif a.cmd == "a0":
        a0(a.device)
    elif a.cmd == "check32":
        arm("a", 32, a.device, check32=True)
    elif a.cmd == "arm":
        arm(a.arm, a.n, a.device)
    elif a.cmd == "diag_only":                                                # diagnostics for an arm whose latents are stored
        with Run(STAGE, f"diag_{a.arm}", a.device, vars(a)) as run:
            m, _ = load_arm(a.arm, a.device)
            res = {"arm": a.arm, "diag": diagnostics(m, make_diffusion(50), a.device), **run.record()}
        atomic_write_json(os.path.join(OUT, "eval", f"diag_{a.arm}.json"), res)


if __name__ == "__main__":
    main()
