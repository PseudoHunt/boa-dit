"""Stage 1 / 4.1: sampler error-propagation gain g_t^2 by a perturbation sweep on the FP sampler.

gain2[i] = mean_seeds ||x0' - x0||^2 / mean_seeds ||D||^2, D added to the 4 eps channels after CFG at step i only.
--probe iso : D = rho * ||eps|| * z/||z||, z ~ N(0, I)
--probe rtn : D = (eps_RTN(x_i) - eps_FP(x_i)) rescaled to the same norm (all block linears W4 RTN per-channel)
"""
import argparse
import copy
import os

import numpy as np
import torch

from boa_dit.common import RESULTS, Run, atomic_torch_save, atomic_write_json, atomic_savefig
from boa_dit.model import LINEARS, block_linear, labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.qutils import rtn_
from boa_dit.sample import ddim_sample

STAGE1_SEEDS = list(range(2000, 2016))


def _sqnorm(x):
    return x.flatten(1).pow(2).sum(1)


def cfg_eps(model, x, i, diffusion, labels, cfg, amp=True):
    """CFG-combined 4-channel eps of `model` at state x, step i (one model call, same layout as ddim_sample)."""
    from boa_dit.model import NULL_CLASS
    n = len(x)
    lab = torch.as_tensor(labels, device=x.device)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    t = torch.full((2 * n,), int(diffusion.timestep_map[diffusion.num_timesteps - 1 - i]), device=x.device)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(torch.cat([x, x]), t, y).float()
    eps, rest = out[:, :3], out[:, 3:4]
    c, u = eps[:n], eps[n:]
    return torch.cat([u + cfg * (c - u), rest[:n]], 1)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--probe", default="iso", choices=["iso", "rtn"])
    ap.add_argument("--rtn_bits", type=int, default=4)
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--fp32", action="store_true", help="full-fp32 forward (no fp16 rounding-chaos floor)")
    args = ap.parse_args()
    tag = args.tag or (f"gain_{args.probe}_rho{args.rho:g}" + ("_fp32" if args.fp32 else ""))
    amp = not args.fp32
    out_json = os.path.join(RESULTS, "stage1", f"{tag}.json")
    partial = os.path.join(RESULTS, "stage1", f".{tag}.partial.pt")
    dev = args.device
    seeds = STAGE1_SEEDS
    labels = labels_for_seeds(seeds)
    model, diffusion = load_dit(dev), make_diffusion(args.steps)
    S = args.steps
    rtn_model = None
    if args.probe == "rtn":
        rtn_model = copy.deepcopy(model)
        for blk in rtn_model.blocks:
            for nm in LINEARS:
                rtn_(block_linear(blk, nm), args.rtn_bits)

    with Run("stage1", tag, dev, vars(args)) as run:
        noise = noise_for_seeds(seeds, dev)
        x0, states = ddim_sample(model, diffusion, noise, labels, args.cfg, record_states=True, amp=amp)
        # sanity: restarting from a recorded state reproduces x0 exactly
        x0_restart = ddim_sample(model, diffusion, noise, labels, args.cfg, start_step=S // 2,
                                 x_init=states[S // 2], amp=amp)
        restart_exact = bool(torch.equal(x0, x0_restart))
        done = torch.load(partial, map_location="cpu") if (args.resume and os.path.exists(partial)) else {}
        for i in range(S):
            if i in done:
                continue
            gen = torch.Generator(device=dev).manual_seed(10_000 + i)
            rec = {}

            def hook(step, xx, out):
                if step != i:
                    return out
                eps = out[:, :4]
                if args.probe == "iso":
                    z = torch.randn(eps.shape, generator=gen, device=dev)
                else:
                    z = cfg_eps(rtn_model, xx, step, diffusion, labels, args.cfg, amp) - eps
                z = z / _sqnorm(z).sqrt().view(-1, 1, 1, 1)
                D = args.rho * _sqnorm(eps).sqrt().view(-1, 1, 1, 1) * z
                rec["D2"] = _sqnorm(D)
                out = out.clone()
                out[:, :4] += D
                return out

            x0p = ddim_sample(model, diffusion, noise, labels, args.cfg, eps_hook=hook, start_step=i, x_init=states[i],
                              amp=amp)
            dx2 = _sqnorm(x0p - x0)
            done[i] = {"dx2": dx2.cpu(), "D2": rec["D2"].cpu()}
            atomic_torch_save(done, partial)
            print(f"  step {i:2d}: gain2 = {float(dx2.mean() / rec['D2'].mean()):.4g}", flush=True)
        gain2 = [float(done[i]["dx2"].mean() / done[i]["D2"].mean()) for i in range(S)]
        per_seed = [(done[i]["dx2"] / done[i]["D2"]).tolist() for i in range(S)]
        g = np.array(gain2)
        gmed = np.array([np.median(r) for r in per_seed])
        res = {"tag": tag, "seeds": seeds, "labels": labels.tolist(), "rho": args.rho, "probe": args.probe,
               "step_convention": "i=0 first (t=980) .. i=S-1 last (t=0)",
               "timesteps": [int(diffusion.timestep_map[S - 1 - i]) for i in range(S)],
               "gain2": gain2, "gain2_per_seed": per_seed,
               "max_over_median": float(g.max() / np.median(g)),
               "gain2_median_of_ratios": gmed.tolist(),
               "median_of_ratios_max_over_median": float(gmed.max() / np.median(gmed)),
               "forward": "fp32" if args.fp32 else "fp16-autocast",
               "argmax_step": int(g.argmax()), "argmin_step": int(g.argmin()),
               "monotone_nonincreasing": bool(np.all(np.diff(g) <= 0)),
               "monotone_nondecreasing": bool(np.all(np.diff(g) >= 0)),
               "restart_from_state_exact": restart_exact, "x0_sqnorm_mean": float(_sqnorm(x0).mean())}
        res.update(run.record())
    atomic_write_json(out_json, res)
    os.remove(partial)
    print({k: v for k, v in res.items() if k not in ("gain2", "gain2_per_seed", "timesteps", "labels")})


if __name__ == "__main__":
    main()
