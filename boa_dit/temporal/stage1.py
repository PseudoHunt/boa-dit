"""Stage 1 (rule_feedback.md): replay heig-8's real guided prediction error into the FP sampler under temporal
rearrangements. Recording: δ_t = heig8(x_t^q) - FP(x_t^q) (4 eps channels, CFG-combined) along heig-8's own trajectory,
eval seeds 0-31 x 50 steps. Replay from the same noise: e_t,s = a·sqrt(E_t,s)·unit(pattern_t,s), E_t,s = ||δ_t,s||².
Patterns: i recorded, ii time-reversed, iii sign-alternated, iv shuffled (fixed π), v fixed direction (step mean,
alternating sign), vi fresh white, vii = i with the denoiser frozen to FP predictions on the FP trajectory.
Run: python -m boa_dit.temporal.stage1 --device cuda:0
"""
import argparse
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.model import labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.objective.proxy import step_outputs
from boa_dit.sample import ddim_sample
from boa_dit.temporal.common import CK, HEIG, OUT, STAGE, boot_ratio
from boa_dit.temporal.dit import TRAJ

SEEDS = list(range(32))
T = 50
AMPS = (0.5, 1.0, 2.0)
PATTERNS = ("i_recorded", "ii_reversed", "iii_alternated", "iv_shuffled", "v_fixed_alt", "vi_white", "vii_coef_only")
REC = os.path.join(CK, "stage1_delta_heig8.pt")
OUT1 = os.path.join(OUT, "stage1")


@torch.no_grad()
def record(fp, diffusion, dev, labels, noise):
    if os.path.exists(REC):
        return torch.load(REC, map_location="cpu")
    q = load_dit(dev)
    load_quant_state(q, os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    install(q, torch.load(os.path.join(CK, "rebuild_quant", HEIG, "carry.pt"), map_location="cpu"))
    D = torch.zeros(len(SEEDS), T, 4, 32, 32)

    def hook(i, xx, out):
        f = step_outputs(fp, diffusion, xx, labels, i)[0]
        D[:, i] = (out[:, :4] - f[:, :4]).cpu()
        return out
    x0q = ddim_sample(q, diffusion, noise, labels, 1.5, eps_hook=hook).cpu()
    del q
    torch.cuda.empty_cache()
    atomic_torch_save({"delta": D, "x0_heig8": x0q}, REC)
    return {"delta": D, "x0_heig8": x0q}


def unit(v):
    return v / v.flatten(1).norm(dim=1).clamp_min(1e-20).view(-1, 1, 1, 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    labels = labels_for_seeds(SEEDS)
    noise = noise_for_seeds(SEEDS, dev)
    res = {"seeds": "0-31", "amps": AMPS, "final_err": {}}
    with Run(STAGE, "fb_stage1_replay", dev, vars(a)) as run:
        fp, diffusion = load_dit(dev), make_diffusion(50)
        R = record(fp, diffusion, dev, labels, noise)
        D = R["delta"]
        E = D.flatten(2).pow(2).sum(2)                                         # [32, 50]
        res["recorded_energy_per_step"] = E.mean(0).tolist()
        tr = torch.load(TRAJ, map_location="cpu")
        x_ref = ddim_sample(fp, diffusion, noise, labels, 1.5).cpu()
        res["ref_bitexact_vs_fp_traj"] = bool(torch.equal(x_ref, tr["x0"]))
        res["heig8_endpoint_err"] = (R["x0_heig8"] - x_ref).flatten(1).pow(2).sum(1).double().tolist()
        perm = torch.randperm(T, generator=torch.Generator().manual_seed(0)).tolist()
        mdir = unit(D.mean(1))
        white = {i: unit(torch.stack([torch.randn(4, 32, 32, generator=torch.Generator().manual_seed(9000 + 100 * i + s)) for s in SEEDS]))
                 for i in range(T)}

        def direction(pat, i):
            if pat in ("i_recorded", "vii_coef_only"):
                return unit(D[:, i])
            if pat == "ii_reversed":
                return unit(D[:, T - 1 - i])
            if pat == "iii_alternated":
                return unit(D[:, i]) * (-1) ** i
            if pat == "iv_shuffled":
                return unit(D[:, perm[i]])
            if pat == "v_fixed_alt":
                return mdir * (-1) ** i
            return white[i]
        for amp in AMPS:
            for pat in PATTERNS:
                def hook(i, xx, out, pat=pat, amp=amp):
                    e = (direction(pat, i) * (amp * E[:, i].sqrt()).view(-1, 1, 1, 1)).to(out.device)
                    out = out.clone()
                    if pat == "vii_coef_only":
                        out[:, :4] = tr["ref"][i].to(out.device) + e
                    else:
                        out[:, :4] += e
                    return out
                x0 = ddim_sample(fp, diffusion, noise, labels, 1.5, eps_hook=hook).cpu()
                err = (x0 - x_ref).flatten(1).pow(2).sum(1).double().tolist()
                res["final_err"][f"{pat}@{amp}"] = err
                print(f"{pat} @ {amp}: {np.mean(err):.4g}", flush=True)
        ratios, ok = {}, {}
        for amp in AMPS:
            base = res["final_err"][f"i_recorded@{amp}"]
            ratios[amp] = {p: boot_ratio(res["final_err"][f"{p}@{amp}"], base) for p in PATTERNS if p != "i_recorded"}
            ok[amp] = bool(ratios[amp]["iii_alternated"]["ratio"] < 0.3 and ratios[amp]["vi_white"]["ratio"] < 0.5)
        res["ratios_to_i"] = ratios
        res["read1_by_amp"] = ok
        res["read1_pass"] = bool(all(ok.values()))
        res["read1_fallback_iii_approx_i"] = bool(ratios[1.0]["iii_alternated"]["ratio"] >= 0.7)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT1, "stage1.json"), res)
    for amp in AMPS:
        print(amp, {p: round(r["ratio"], 4) for p, r in ratios[amp].items()}, flush=True)
    print("read1", res["read1_pass"], res["read1_by_amp"], flush=True)


if __name__ == "__main__":
    main()
