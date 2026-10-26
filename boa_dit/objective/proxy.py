"""D4 (rule.md): step-error proxy on DiT-XL/2. FP trajectory of eval seeds 0-63 cached at steps 0,5,..,45; for every
stored W4 arm, rel. MSE of its guided prediction (the sampler's 4 eps channels, through the sampler's own CFG code path)
vs FP on the same x_t, mean over 640 (seed, step). Also saves FP and heig-8 per-branch eps (3 guided channels) for D1.3/D2.
Run: python -m boa_dit.objective.proxy --device cuda:0
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import NULL_CLASS, labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX, ddim_sample

STEPS = tuple(range(0, 50, 5))
SEEDS = list(range(64))
CFG = 1.5
OUT = os.path.join(RESULTS, "objective")
TRAJ = os.path.join(CKPT_DIR, "objective", "fp_traj64.pt")
R = os.path.join(RESULTS)
ARMS = {  # name -> (checkpoint arm or None, carry?, static rank, stored latent json)
    "uniform": ("uniform_w4_K1_lam1_brcfg", False, 0, ("stage2", "eval", "latent_uniform_w4_K1_lam1_brcfg.json")),
    "static-8": ("uniform_w4_K1_lam1_brcfg", False, 8, ("stageA", "eval", "latent_static-8_w4.json")),
    "static-32": ("uniform_w4_K1_lam1_brcfg", False, 32, ("stageA", "eval", "latent_static-32_w4.json")),
    "wsvd-8": ("deflate_wsvd_r8_w4", True, 0, ("gate1", "eval", "latent_deflate_wsvd_r8_w4.json")),
    "dc-8": ("deflate_dc_r8_w4", True, 0, ("gate1", "eval", "latent_deflate_dc_r8_w4.json")),
    "dc-32": ("deflate_dc_r32_w4", True, 0, ("gate1", "eval", "latent_deflate_dc_r32_w4.json")),
    "heig-8": ("deflate_heig_r8_w4", True, 0, ("gate1", "eval", "latent_deflate_heig_r8_w4.json")),
    "E1": ("ts_dc8_ao_w4", True, 0, ("gate1", "eval", "latent_ts_dc8_ao_w4.json")),
    "E2": ("ts_uniform_vv1_w4", True, 0, ("gate1", "eval", "latent_ts_uniform_vv1_w4.json")),
    "E3": ("ts_uniform_fc1v1_w4", True, 0, ("gate1", "eval", "latent_ts_uniform_fc1v1_w4.json")),
    "E4": ("ts_uniform_vv1_fc1v1_w4", True, 0, ("gate1", "eval", "latent_ts_uniform_vv1_fc1v1_w4.json")),
    "E5": ("ts_uniform_vv2_fc1v1_wov2_w4", True, 0, ("gate1", "eval", "latent_ts_uniform_vv2_fc1v1_wov2_w4.json")),
    "E6": ("ts_uniform_vv1_fc1v1_ao_w4", True, 0, ("gate1", "eval", "latent_ts_uniform_vv1_fc1v1_ao_w4.json")),
    "H2": ("ts_heig8_vv1_w4", True, 0, ("gate1", "eval", "latent_ts_heig8_vv1_w4.json")),
    "H4": ("ts_heig8_vv1_fc1v1_w4", True, 0, ("gate1", "eval", "latent_ts_heig8_vv1_fc1v1_w4.json")),
    "H4dsf": ("ts_heig8_vv1_fc1v1_dsf_w4", True, 0, ("gate1", "eval", "latent_ts_heig8_vv1_fc1v1_dsf_w4.json")),
    "Hrg": ("ts_heig8_fc1rg_w4", True, 0, ("gate1", "eval", "latent_ts_heig8_fc1rg_w4.json")),
}


def load_arm(name, dev, extra=None):
    """Returns (model, hooks). extra: {name: (ckpt arm, carry, static rank, latent json)} for new arms."""
    spec = (extra or {}).get(name) or ARMS[name]
    ck, carry, k, _ = spec
    m = load_dit(dev)
    load_quant_state(m, os.path.join(CKPT_DIR, f"ckpt_{ck}", "model.pt"))
    hooks = []
    if carry:
        from boa_dit.deflate.carry import install_from
        hooks = install_from(m, ck)
    if k:
        from boa_dit import lowrank
        c = lowrank.Corrector(m, lowrank.build(4, dev, ["static"]), "static", k, dev)
        hooks = c.hooks
    return m, hooks


@torch.no_grad()
def step_outputs(model, diffusion, x, labels, i, cfg=CFG):
    """One DDIM step at index i from x through the sampler's CFG code path; returns (guided res [n,8,..], raw out [2n,8,..])."""
    cap = {}
    dev = x.device
    n = len(x)
    lab = torch.as_tensor(labels, device=dev, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    ti = list(range(diffusion.num_timesteps))[::-1][i]
    CTX.step, CTX.n = i, n

    def model_fn(xx, ts):
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(torch.cat([xx, xx]), torch.cat([ts, ts]), y).float()
        eps, rest = out[:, :3], out[:, 3:]
        c, u = eps[:n], eps[n:]
        res = torch.cat([u + cfg * (c - u), rest[:n]], dim=1)
        cap["res"], cap["raw"] = res, out
        return res

    t = torch.full((n,), ti, device=dev, dtype=torch.long)
    diffusion.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)
    CTX.step = None
    return cap["res"], cap["raw"]


@torch.no_grad()
def fp_traj(dev):
    if os.path.exists(TRAJ):
        return torch.load(TRAJ, map_location="cpu")
    m, diffusion = load_dit(dev), make_diffusion(50)
    xs = {s: [] for s in STEPS}
    x0 = []
    for b in range(0, len(SEEDS), 32):
        sd = SEEDS[b:b + 32]
        x, st = ddim_sample(m, diffusion, noise_for_seeds(sd, dev), labels_for_seeds(sd), CFG, record_states=True)
        for s in STEPS:
            xs[s].append(st[s].cpu())
        x0.append(x.cpu())
    out = {"steps": STEPS, "x": {s: torch.cat(v) for s, v in xs.items()}, "x0": torch.cat(x0)}
    atomic_torch_save(out, TRAJ)
    return out


@torch.no_grad()
def proxy_for(model, diffusion, traj, dev, keep_branches=False):
    rel = torch.zeros(len(STEPS), len(SEEDS))
    br = {}
    labels = labels_for_seeds(SEEDS)
    for si, s in enumerate(STEPS):
        for b in range(0, len(SEEDS), 32):
            x = traj["x"][s][b:b + 32].to(dev)
            res, raw = step_outputs(model, diffusion, x, labels[b:b + 32], s)
            if keep_branches and b == 0:
                br[s] = {"res": res[:, :4].cpu(), "raw": raw[:, :3].cpu()}
            yield_ = res[:, :4]
            if "ref" in traj:
                ref = traj["ref"][s][b:b + 32].to(dev)
                rel[si, b:b + 32] = ((yield_ - ref).flatten(1).pow(2).sum(1) / ref.flatten(1).pow(2).sum(1)).cpu()
            else:
                traj.setdefault("ref_build", {}).setdefault(s, []).append(yield_.cpu())
    return rel, br


def stored_latent(spec):
    p = os.path.join(R, *spec[3])
    return json.load(open(p))["rel_mse_mean"] if os.path.exists(p) else None


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def run_extra(a):
    dev = a.device
    torch.cuda.set_device(dev)
    extra = {}
    for item in a.extra.split(","):
        lab, ck = item.split("=")
        extra[lab] = (ck, True, 0, ("objective", "eval", f"latent_{ck}.json"))
    p = os.path.join(OUT, "proxy", "d4.json")
    res = json.load(open(p))
    with Run("objective", "d4_proxy_extra", dev, vars(a)) as run:
        diffusion = make_diffusion(50)
        traj = fp_traj(dev)
        fp = load_dit(dev)
        proxy_for(fp, diffusion, traj, dev)
        traj["ref"] = {s: torch.cat(v) for s, v in traj.pop("ref_build").items()}
        del fp
        for lab, spec in extra.items():
            m, _ = load_arm(lab, dev, extra)
            rel, _ = proxy_for(m, diffusion, traj, dev)
            res["arms"][lab] = {"proxy": float(rel.mean()), "proxy_by_step": rel.mean(1).tolist(),
                                "latent_rel_mse": stored_latent(spec), "per_pair": rel.flatten().tolist()}
            print(f"{lab}: proxy {rel.mean():.5f}  latent {res['arms'][lab]['latent_rel_mse']}", flush=True)
            del m
            torch.cuda.empty_cache()
        names = [n for n in res["arms"] if res["arms"][n]["latent_rel_mse"] is not None]
        pr = np.array([res["arms"][n]["proxy"] for n in names])
        lat = np.array([res["arms"][n]["latent_rel_mse"] for n in names])
        res["spearman_all"], res["pearson_all"], res["n_arms_all"] = spearman(pr, lat), float(np.corrcoef(pr, lat)[0, 1]), len(names)
        res["extra_run"] = run.record()
    atomic_write_json(p, res)
    print(f"Spearman (all {res['n_arms_all']} arms) {res['spearman_all']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--extra", default="", help="comma list label=ckpt_arm (carry arms, latent in results/objective/eval); "
                                                "merged into d4.json")
    a = ap.parse_args()
    if a.extra:
        return run_extra(a)
    dev = a.device
    torch.cuda.set_device(dev)
    os.makedirs(os.path.join(OUT, "proxy"), exist_ok=True)
    with Run("objective", "d4_proxy", dev, vars(a)) as run:
        diffusion = make_diffusion(50)
        traj = fp_traj(dev)
        fp = load_dit(dev)
        _, br_fp = proxy_for(fp, diffusion, traj, dev, keep_branches=True)       # builds the FP reference
        traj["ref"] = {s: torch.cat(v) for s, v in traj.pop("ref_build").items()}
        rel_fp, _ = proxy_for(fp, diffusion, traj, dev)
        del fp
        torch.cuda.empty_cache()
        res = {"fp_floor": float(rel_fp.mean()), "arms": {}}
        atomic_torch_save({"fp": br_fp}, os.path.join(CKPT_DIR, "objective", "branches_fp.pt"))
        for name in a.arms.split(","):
            m, hooks = load_arm(name, dev)
            rel, br = proxy_for(m, diffusion, traj, dev, keep_branches=(name == "heig-8"))
            if name == "heig-8":
                atomic_torch_save({"heig-8": br}, os.path.join(CKPT_DIR, "objective", "branches_heig8.pt"))
            res["arms"][name] = {"proxy": float(rel.mean()), "proxy_by_step": rel.mean(1).tolist(),
                                 "latent_rel_mse": stored_latent(ARMS[name]), "per_pair": rel.flatten().tolist()}
            print(f"{name}: proxy {rel.mean():.5f}  latent {res['arms'][name]['latent_rel_mse']}", flush=True)
            del m
            torch.cuda.empty_cache()
        names = [n for n in res["arms"] if res["arms"][n]["latent_rel_mse"] is not None]
        p = np.array([res["arms"][n]["proxy"] for n in names])
        lat = np.array([res["arms"][n]["latent_rel_mse"] for n in names])
        res["spearman"] = spearman(p, lat)
        res["pearson"] = float(np.corrcoef(p, lat)[0, 1])
        res["n_arms"] = len(names)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "proxy", "d4.json"), res)
    print(f"Spearman {res['spearman']:.3f}, Pearson {res['pearson']:.3f} over {res['n_arms']} arms; FP floor {res['fp_floor']:.2e}", flush=True)


if __name__ == "__main__":
    main()
