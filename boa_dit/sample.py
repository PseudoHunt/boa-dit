"""Seeded DDIM sampling (cond+uncond batched like DiT.forward_with_cfg); returns x0 latents.

Step index convention used everywhere in this repo: i = 0..S-1 in *sampling order*
(i = 0 is the first, noisiest step; i = S-1 is the last). Hooks read CTX.step.
"""
import argparse
import os
import time

import torch

from boa_dit.common import RESULTS, Run, atomic_torch_save, atomic_write_json, assert_no_grad
from boa_dit.model import NULL_CLASS, decode, labels_for_seeds, load_dit, load_vae, make_diffusion, noise_for_seeds


class _Ctx:
    step = None   # current sampling step index
    n = None      # number of images in the batch (rows [0,n) cond, [n,2n) uncond)


CTX = _Ctx()


@torch.no_grad()
def ddim_sample(model, diffusion, noise, labels, cfg=1.5, eps_hook=None, start_step=0, x_init=None,
                record_states=False, amp=True):
    """eps_hook(i, x_i, out) -> out is called on the CFG-combined [N,8,32,32] model output.
    amp=False runs the forward in full fp32 (used by the gain probe: fp16 rounding chaos sets a floor
    on ||dx0||^2 that swamps small finite-difference perturbations)."""
    dev = noise.device
    x = (noise if x_init is None else x_init).float().clone()
    n = len(x)
    lab = torch.as_tensor(labels, device=dev, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    indices = list(range(diffusion.num_timesteps))[::-1]
    states = {}
    for i, ti in enumerate(indices):
        if i < start_step:
            continue
        if record_states:
            states[i] = x.clone()
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            # ts arrive already mapped to the original 1000-step grid by _WrappedModel.
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                out = model(torch.cat([xx, xx]), torch.cat([ts, ts]), y).float()
            # Official forward_with_cfg: CFG on the first 3 eps channels only; the 4th eps
            # channel and the variance channels come from the cond half.
            eps, rest = out[:, :3], out[:, 3:]
            c, u = eps[:n], eps[n:]
            res = torch.cat([u + cfg * (c - u), rest[:n]], dim=1)
            if eps_hook is not None:
                res = eps_hook(_i, xx, res)
            return res

        t = torch.full((n,), ti, device=dev, dtype=torch.long)
        x = diffusion.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return (x, states) if record_states else x


@torch.no_grad()
def sample_seeds(model, diffusion, seeds, cfg=1.5, bs=32, partial=None, resume=False, labels=None, log=True):
    """Batched sampling with per-batch checkpointing to `partial` (dict of batch_idx -> latents)."""
    dev = next(model.parameters()).device
    seeds = list(seeds)
    labels = labels_for_seeds(seeds) if labels is None else labels
    done = {}
    if partial and resume and os.path.exists(partial):
        done = torch.load(partial)
    for b in range(0, len(seeds), bs):
        if b in done:
            continue
        t0 = time.time()
        x0 = ddim_sample(model, diffusion, noise_for_seeds(seeds[b:b + bs], dev), labels[b:b + bs], cfg)
        assert_no_grad(x0)
        done[b] = x0.cpu()
        if partial:
            atomic_torch_save(done, partial)
        if log:
            print(f"  batch {b // bs + 1}/{(len(seeds) + bs - 1) // bs}  {time.time() - t0:.1f}s", flush=True)
    return torch.cat([done[b] for b in sorted(done)])


def parse_seeds(s):
    a, b = s.split("-") if "-" in s else (s, s)
    return list(range(int(a), int(b) + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-7")
    ap.add_argument("--cfg", type=float, default=1.5)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default=os.path.join(RESULTS, "stage0", "fp_latents.pt"))
    ap.add_argument("--png", action="store_true")
    ap.add_argument("--check_repeat", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--timing_bs", type=int, default=0, help="time one batch of this size")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    model, diffusion = load_dit(args.device), make_diffusion(args.steps)
    rec = {}
    with Run("stage0", "sample", args.device, vars(args)) as run:
        x0 = sample_seeds(model, diffusion, seeds, args.cfg, args.bs, partial=args.out + ".partial",
                          resume=args.resume)
        atomic_torch_save({"seeds": seeds, "labels": labels_for_seeds(seeds), "x0": x0}, args.out)
        if args.check_repeat:
            x0b = sample_seeds(model, diffusion, seeds, args.cfg, args.bs)
            rec["repeat_bitwise_equal"] = bool(torch.equal(x0, x0b))
            rec["repeat_max_abs_diff"] = float((x0 - x0b).abs().max())
        if args.timing_bs:
            ts = list(range(10_000, 10_000 + args.timing_bs))
            torch.cuda.synchronize(args.device)
            t0 = time.time()
            sample_seeds(model, diffusion, ts, args.cfg, args.timing_bs, log=False)
            torch.cuda.synchronize(args.device)
            rec["timing_bs"] = args.timing_bs
            rec["s_per_image"] = (time.time() - t0) / args.timing_bs
        if args.png:
            from torchvision.utils import save_image
            vae = load_vae(args.device)
            imgs = decode(vae, x0[:8].to(args.device))
            p = os.path.join(os.path.dirname(args.out), "fp_samples.png")
            save_image(imgs.float() / 255, p + ".tmp.png", nrow=4)
            os.replace(p + ".tmp.png", p)
            rec["png_labels"] = labels_for_seeds(seeds[:8]).tolist()
        if os.path.exists(args.out + ".partial"):
            os.remove(args.out + ".partial")
        rec.update(run.record())
    atomic_write_json(os.path.join(os.path.dirname(args.out), "sample_run.json"), rec)
    print(rec)


if __name__ == "__main__":
    main()
