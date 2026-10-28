"""Follow-up Experiment 3 (results/followup/PROTOCOL.md §3): trajectory-aware optimization of heig-8's existing rank-8 carry.
Only the carry tensors V [d_in, 8] and delta [d_out, 8] of the 112 block linears are trained (fp32 masters, exported fp16 in
the original shapes); W4 codes/scales and everything else stay frozen.
  check   : differentiable path == heig-8 (5-step rollout vs ddim_sample; untrained fp16 export bitwise through install())
  cache   : heig-8 trajectories (training 8200-8263, design 8300-8331), FP endpoints (design), normalization constants
  profile : seconds per update for C1 and C2/C3 at the chosen batch
  train   : --arm C1|C2|C3 --lam --lr --hours [--tag]   (matched wall-clock training time)
  design  : lr pick (C1 short runs) / λ pick (C2, C3 short runs) on design seeds, then frozen
  eval    : seeds 8400-8527: C0-C3 latent / PSNR / LPIPS vs FP; C, I, C/I on 8400-8431 (FP-trajectory states)
Run: python -m boa_dit.followup.carry_opt check|cache|profile|design|final|eval
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.f4 import STAGE, guard, spent
from boa_dit.model import NULL_CLASS, block_linear, decode, labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
from boa_dit.objective.d2 import band_masks
from boa_dit.sample import ddim_sample

OUT = os.path.join(RESULTS, "followup", "e3")
CK = os.path.join(CKPT_DIR, "followup", "e3")
HEIG = "deflate_heig_r8_w4"
QSTATE = os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt")
CARRY = os.path.join(CKPT_DIR, "temporal", "rebuild_quant", HEIG, "carry.pt")
TRAIN, DESIGN, EVAL = list(range(8200, 8264)), list(range(8300, 8332)), list(range(8400, 8528))
CI_SEEDS = EVAL[:32]
T, K, CFG, B = 50, 5, 1.5, 4
LRS, LAMS = [1e-4, 3e-4], [0.5, 2.0]
LOW = band_masks(32)[0]


# ------------------------------------------------------------------------------------------------- trainable carry
class Carry(torch.nn.Module):
    """fp32 master copies of every V and delta; forward hooks reproduce deflate.carry.attach under fp16 autocast."""

    def __init__(self, model, carry, dev):
        super().__init__()
        self.keys = sorted(carry)
        self.V = torch.nn.ParameterList([torch.nn.Parameter(carry[k]["V"].float().to(dev)) for k in self.keys])
        self.D = torch.nn.ParameterList([torch.nn.Parameter(carry[k]["delta"].float().to(dev)) for k in self.keys])
        self.hs = []
        for i, k in enumerate(self.keys):
            l, ln = k.split(".", 1)
            self.hs.append(block_linear(model.blocks[int(l[1:])], ln).register_forward_hook(self._hook(i)))

    def _hook(self, i):
        def hook(mod, inp, out):
            x, dt = inp[0], out.dtype
            if dt == torch.float16:                                           # same op order and dtypes as attach()
                return out + ((x.half() @ self.V[i].half()) @ self.D[i].half().T).to(dt)
            return out + (x.float() @ self.V[i]) @ self.D[i].T
        return hook

    def export(self):
        return {k: {"V": self.V[i].detach().half().cpu(), "delta": self.D[i].detach().half().cpu()} for i, k in enumerate(self.keys)}

    def remove(self):
        for h in self.hs:
            h.remove()


def heig_base(dev):
    """heig-8 W4 weights (incl. W8 adaLN) without the carry; frozen."""
    m = load_dit(dev)
    load_quant_state(m, QSTATE)
    m.requires_grad_(False)
    return m


def fp_model(dev):
    m = load_dit(dev)
    m.requires_grad_(False)
    return m


# ------------------------------------------------------------------------------------------- differentiable sampler
class Sched:
    def __init__(self, d, dev):
        self.ab = torch.tensor(d.alphas_cumprod, dtype=torch.float64)
        self.abp = torch.tensor(d.alphas_cumprod_prev, dtype=torch.float64)
        self.tmap = torch.tensor(d.timestep_map, dtype=torch.long, device=dev)
        self.idx = list(range(d.num_timesteps))[::-1]                          # step i -> respaced t


def guided_eps(model, x, labels, i, sch, amp=True):
    """the sampler's CFG path, differentiable: CFG on the first 3 eps channels, channel 4 from the cond half."""
    n = len(x)
    y = torch.cat([labels, torch.full_like(labels, NULL_CLASS)])
    t = sch.tmap[torch.full((2 * n,), sch.idx[i], device=x.device, dtype=torch.long)]
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        out = model(torch.cat([x, x]), t, y).float()
    eps3, rest = out[:, :3], out[:, 3:]
    c, u = eps3[:n], eps3[n:]
    return torch.cat([u + CFG * (c - u), rest[:n, :1]], 1)


def ddim_step(x, eps, i, sch):
    """deterministic DDIM (eta=0, no clipping) in float32, as guided_diffusion: x0 from eps, then x_{i+1}."""
    t = sch.idx[i]
    ab, abp = float(sch.ab[t]), float(sch.abp[t])
    x0 = math.sqrt(1.0 / ab) * x - math.sqrt(1.0 / ab - 1) * eps
    e = (math.sqrt(1.0 / ab) * x - x0) / math.sqrt(1.0 / ab - 1)
    return x0 * math.sqrt(abp) + math.sqrt(1 - abp) * e


def rollout(model, x, labels, i0, k, sch, grad, ckpt=True):
    """k DDIM calls from step i0; returns (eps at the start state, end state). grad: keep the graph through every state."""
    eps0 = None
    for i in range(i0, i0 + k):
        if grad and ckpt:
            eps = checkpoint(guided_eps, model, x, labels, i, sch, use_reentrant=False)
        else:
            eps = guided_eps(model, x, labels, i, sch)
        if eps0 is None:
            eps0 = eps
        x = ddim_step(x, eps, i, sch)
    return eps0, x


# ---------------------------------------------------------------------------------------------------- checks, cache
def check(dev):
    with Run(STAGE, "e3_check", dev, {}) as run:
        d = make_diffusion(T)
        sch = Sched(d, dev)
        m = heig_base(dev)
        carry = torch.load(CARRY, map_location="cpu")
        seeds = DESIGN[:4]
        lab = torch.as_tensor(labels_for_seeds(seeds), device=dev)
        nz = noise_for_seeds(seeds, dev)
        hs = install(m, carry)                                                # the deployed path
        ref, st = ddim_sample(m, d, nz, labels_for_seeds(seeds), CFG, record_states=True)
        for h in hs:
            h.remove()
        c = Carry(m, carry, dev)
        res = {}
        with torch.no_grad():
            for i0 in (0, 20, 45):
                _, xe = rollout(m, st[i0], lab, i0, min(K, T - i0), sch, grad=False)
                tgt = st[i0 + min(K, T - i0)] if i0 + K < T else ref
                res[f"rollout_rel_err_from_{i0}"] = float((xe - tgt).norm() / tgt.norm())
        # gradients reach V and delta through a 5-step rollout (no hidden detach)
        x = st[20].clone()
        _, xe = rollout(m, x, lab, 20, K, sch, grad=True)
        xe.pow(2).mean().backward()
        res["grad_norm_V"] = float(sum(p.grad.norm() ** 2 for p in c.V) ** 0.5)
        res["grad_norm_D"] = float(sum(p.grad.norm() ** 2 for p in c.D) ** 0.5)
        res["grad_first_block_nonzero"] = bool(c.V[0].grad.abs().sum() > 0)
        exp = c.export()
        c.remove()
        res["export_bitwise_equal_to_heig8_carry"] = all(torch.equal(exp[k]["V"], carry[k]["V"]) and torch.equal(exp[k]["delta"], carry[k]["delta"]) for k in carry)
        hs = install(m, exp)
        x2 = ddim_sample(m, d, nz, labels_for_seeds(seeds), CFG)
        for h in hs:
            h.remove()
        res["export_sampling_bitwise"] = bool(torch.equal(x2, ref))
        res["pass"] = max(v for k, v in res.items() if k.startswith("rollout")) <= 1e-3 and res["export_sampling_bitwise"] and res["grad_first_block_nonzero"]
        res.update(run.record())
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, "check.json"), res)
    print(res, flush=True)
    if not res["pass"]:
        raise SystemExit(6)


@torch.no_grad()
def cache(dev):
    if not guard(0.08, "e3 cache"):
        raise SystemExit(5)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "e3_cache", dev, {"train": [TRAIN[0], TRAIN[-1]], "design": [DESIGN[0], DESIGN[-1]]}) as run:
        d = make_diffusion(T)
        sch = Sched(d, dev)
        q = heig_base(dev)
        hs = install(q, torch.load(CARRY, map_location="cpu"))
        fp = fp_model(dev)
        out = {}
        for name, seeds in (("train", TRAIN), ("design", DESIGN)):
            X = torch.zeros(len(seeds), T, 4, 32, 32)
            for b in range(0, len(seeds), 32):
                s = seeds[b:b + 32]
                _, st = ddim_sample(q, d, noise_for_seeds(s, dev), labels_for_seeds(s), CFG, record_states=True)
                for i in range(T):
                    X[b:b + 32, i] = st[i][:, :4].cpu()
            out[name] = {"x": X, "labels": torch.as_tensor(labels_for_seeds(seeds)), "seeds": seeds}
        out["design"]["x0fp"] = torch.cat([ddim_sample(fp, d, noise_for_seeds(DESIGN[b:b + 32], dev), labels_for_seeds(DESIGN[b:b + 32]), CFG).cpu()
                                           for b in range(0, len(DESIGN), 32)])
        # normalization constants (C0 on the first 16 training seeds): local, rollout full, rollout band-1, per start step
        n_loc, n_roll, n_low = torch.zeros(T), torch.zeros(T), torch.zeros(T)
        X, L = out["train"]["x"][:16].to(dev), out["train"]["labels"][:16].to(dev)
        lowm = LOW.to(dev)
        for i in range(T):
            k = min(K, T - i)
            es, xs = rollout(q, X[:, i], L, i, k, sch, grad=False)
            et, xt = rollout(fp, X[:, i], L, i, k, sch, grad=False)
            n_loc[i] = (es - et).pow(2).flatten(1).sum(1).mean()
            dd = xs - xt
            n_roll[i] = dd.pow(2).flatten(1).sum(1).mean()
            n_low[i] = band_energy(dd, lowm).mean()
        out["norm"] = {"loc": n_loc, "roll": n_roll, "low": n_low}
        for h in hs:
            h.remove()
        run.record()
    atomic_torch_save(out, os.path.join(CK, "cache.pt"))
    print("  cache done; n_loc", n_loc[[0, 25, 49]].tolist(), "n_roll", n_roll[[0, 25, 49]].tolist(), flush=True)


def band_energy(dd, mask):
    F = torch.fft.fft2(dd.float())
    return ((F.real ** 2 + F.imag ** 2) * mask).flatten(1).sum(1) / (32 * 32)


# ---------------------------------------------------------------------------------------------------------- training
def train(dev, arm, lam, lr, hours, tag, seed=0, max_updates=None):
    """matched wall-clock training of one arm from the heig-8 carry; returns the exported carry path and a log."""
    C = torch.load(os.path.join(CK, "cache.pt"))
    d = make_diffusion(T)
    sch = Sched(d, dev)
    m = heig_base(dev)
    fp = fp_model(dev)
    carry = Carry(m, torch.load(CARRY, map_location="cpu"), dev)
    opt = torch.optim.Adam(carry.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(init_scale=2.0 ** 12)
    X, L = C["train"]["x"], C["train"]["labels"]
    nl, nr, nw = (C["norm"][k].to(dev) for k in ("loc", "roll", "low"))
    lowm = LOW.to(dev)
    g = torch.Generator().manual_seed(seed)
    log, n_up, t_start = [], 0, time.time()
    while time.time() - t_start < hours * 3600 and (max_updates is None or n_up < max_updates):
        si = torch.randint(0, len(X), (B,), generator=g)
        i0 = int(torch.randint(0, T, (1,), generator=g))                      # one start step per update (shared k)
        k = min(K, T - i0)
        x = X[si, i0].to(dev)
        lab = L[si].to(dev)
        with torch.no_grad():
            et, xt = rollout(fp, x, lab, i0, k if arm != "C1" else 1, sch, grad=False)
        if arm == "C1":
            es = guided_eps(m, x, lab, i0, sch)
            loss = ((es - et).pow(2).flatten(1).sum(1) / nl[i0]).mean()
            lr_ = lt_ = None
        else:
            es, xs = rollout(m, x, lab, i0, k, sch, grad=True)
            ll = ((es - et).pow(2).flatten(1).sum(1) / nl[i0]).mean()
            dd = xs - xt
            lroll = (dd.pow(2).flatten(1).sum(1) / nr[i0]).mean() if arm == "C2" else (band_energy(dd, lowm) / nw[i0]).mean()
            loss = ll + lam * lroll
            lr_, lt_ = float(ll), float(lroll)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        n_up += 1
        if n_up % 50 == 0:
            log.append({"update": n_up, "t": time.time() - t_start, "loss": float(loss), "local": lr_, "roll": lt_})
            print(f"    {tag}: update {n_up}, {time.time() - t_start:.0f} s, loss {float(loss):.4f}" + (f" (local {lr_:.4f}, roll {lt_:.4f})" if lr_ is not None else ""), flush=True)
    exp = carry.export()
    carry.remove()
    path = os.path.join(CK, f"carry_{tag}.pt")
    atomic_torch_save(exp, path)
    return path, {"arm": arm, "lam": lam, "lr": lr, "hours": hours, "max_updates": max_updates, "seed": seed, "updates": n_up, "train_seconds": time.time() - t_start, "log": log}


def profile(dev):
    """seconds per update for C1 and C2 (C3 = C2 cost), 20 updates each after 3 warm-up."""
    if not guard(0.05, "e3 profile"):
        raise SystemExit(5)
    with Run(STAGE, "e3_profile", dev, {"B": B, "K": K}) as run:
        res = {}
        for arm in ("C1", "C2"):
            _, lg = train(dev, arm, 1.0, 1e-4, 60 / 3600, f"profile_{arm}")
            res[arm] = {"sec_per_update": lg["train_seconds"] / max(lg["updates"], 1), "updates_in_60s": lg["updates"]}
            torch.cuda.empty_cache()
        res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "profile.json"), res)
    print(res, flush=True)


# ------------------------------------------------------------------------------------------------------- evaluation
@torch.no_grad()
def sample_with(dev, carry_path, seeds, d, m=None):
    m = heig_base(dev) if m is None else m
    hs = install(m, torch.load(carry_path, map_location="cpu"))
    x = torch.cat([ddim_sample(m, d, noise_for_seeds(seeds[b:b + 32], dev), labels_for_seeds(seeds[b:b + 32]), CFG).cpu()
                   for b in range(0, len(seeds), 32)])
    for h in hs:
        h.remove()
    return x, m


def rel(x, f):
    return ((x - f).flatten(1).pow(2).sum(1) / f.flatten(1).pow(2).sum(1)).double()


@torch.no_grad()
def design_local_loss(dev, carry_path):
    """C1 lr pick: mean normalized local loss on design seeds at steps 0, 5, ..., 45."""
    C = torch.load(os.path.join(CK, "cache.pt"))
    d = make_diffusion(T)
    sch = Sched(d, dev)
    m, fp = heig_base(dev), fp_model(dev)
    hs = install(m, torch.load(carry_path, map_location="cpu"))
    X, L, nl = C["design"]["x"], C["design"]["labels"].to(dev), C["norm"]["loc"]
    v = []
    for i in range(0, T, 5):
        x = X[:, i].to(dev)
        v.append(float(((guided_eps(m, x, L, i, sch) - guided_eps(fp, x, L, i, sch)).pow(2).flatten(1).sum(1) / float(nl[i])).mean()))
    for h in hs:
        h.remove()
    return float(np.mean(v))


def design(dev):
    """lr (C1) then λ (C2, C3) on design seeds with short runs (T_train / 4); frozen into design.json."""
    prof = json.load(open(os.path.join(OUT, "profile.json")))
    ttrain = json.load(open(os.path.join(OUT, "plan.json")))["T_train_hours"]
    short = ttrain / 4
    est = short * (len(LRS) + 2 * len(LAMS)) + 0.05
    if not guard(est, "e3 design"):
        raise SystemExit(5)
    res = {"T_train_hours": ttrain, "short_hours": short, "lr": {}, "lam": {"C2": {}, "C3": {}}}
    with Run(STAGE, "e3_design", dev, {"lrs": LRS, "lams": LAMS, "short_h": short}) as run:
        d = make_diffusion(T)
        C = torch.load(os.path.join(CK, "cache.pt"))
        xf = C["design"]["x0fp"]
        base = design_local_loss(dev, CARRY)
        res["C0_design_local"] = base
        for lr in LRS:
            p, lg = train(dev, "C1", 0.0, lr, short, f"design_C1_lr{lr:g}")
            res["lr"][str(lr)] = {"design_local": design_local_loss(dev, p), "updates": lg["updates"]}
            print(f"  lr {lr:g}: design local {res['lr'][str(lr)]['design_local']:.4f} (C0 {base:.4f})", flush=True)
        lr_star = min(LRS, key=lambda l: res["lr"][str(l)]["design_local"])
        res["lr_star"] = lr_star
        x0, m = sample_with(dev, CARRY, DESIGN, d)
        res["C0_design_latent"] = float(rel(x0, xf).mean())
        for arm in ("C2", "C3"):
            for lam in LAMS:
                p, lg = train(dev, arm, lam, lr_star, short, f"design_{arm}_lam{lam:g}")
                x, _ = sample_with(dev, p, DESIGN, d, m)
                res["lam"][arm][str(lam)] = {"design_latent": float(rel(x, xf).mean()), "updates": lg["updates"]}
                print(f"  {arm} λ {lam:g}: design latent {res['lam'][arm][str(lam)]['design_latent']:.5f} (C0 {res['C0_design_latent']:.5f})", flush=True)
        res["lam_star"] = {a: min(LAMS, key=lambda l: res["lam"][a][str(l)]["design_latent"]) for a in ("C2", "C3")}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "design.json"), res)
    print("  frozen: lr", res["lr_star"], "λ", res["lam_star"], flush=True)


def final(dev):
    dz = json.load(open(os.path.join(OUT, "design.json")))
    tt = dz["T_train_hours"]
    res = {}
    for arm in ("C1", "C2", "C3"):
        f = os.path.join(OUT, f"train_{arm}.json")
        if os.path.exists(f):
            continue
        if not guard(tt + 0.02, f"e3 train {arm}"):
            res[arm] = "UNRUN (budget)"
            atomic_write_json(f, {"arm": arm, "status": "unrun due to cost", "estimate_h": tt})
            continue
        lam = 0.0 if arm == "C1" else dz["lam_star"][arm]
        with Run(STAGE, f"e3_train_{arm}", dev, {"lam": lam, "lr": dz["lr_star"], "hours": tt}) as run:
            p, lg = train(dev, arm, lam, dz["lr_star"], tt, f"final_{arm}")
            lg.update({"carry": p, **run.record()})
        atomic_write_json(f, lg)
        torch.cuda.empty_cache()


@torch.no_grad()
def eval_(dev):
    if not guard(0.15, "e3 eval"):
        raise SystemExit(5)
    paths = {"C0": CARRY}
    for arm in ("C1", "C2", "C3"):
        f = os.path.join(OUT, f"train_{arm}.json")
        if os.path.exists(f) and "carry" in json.load(open(f)):
            paths[arm] = json.load(open(f))["carry"]
    with Run(STAGE, "e3_eval", dev, {"seeds": [EVAL[0], EVAL[-1]], "arms": list(paths)}) as run:
        d = make_diffusion(T)
        sch = Sched(d, dev)
        fp = fp_model(dev)
        xf = torch.cat([ddim_sample(fp, d, noise_for_seeds(EVAL[b:b + 32], dev), labels_for_seeds(EVAL[b:b + 32]), CFG).cpu() for b in range(0, len(EVAL), 32)])
        lab = torch.as_tensor(labels_for_seeds(CI_SEEDS), device=dev)
        _, stf = ddim_sample(fp, d, noise_for_seeds(CI_SEEDS, dev), labels_for_seeds(CI_SEEDS), CFG, record_states=True)
        epsf = [guided_eps(fp, stf[i], lab, i, sch) for i in range(T)]
        del fp
        m = heig_base(dev)
        vae = load_vae(dev)
        imf = decode(vae, xf.to(dev))
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        lat, ims, out = {}, {}, {"seeds": [EVAL[0], EVAL[-1]], "arms": {}}
        for arm, p in paths.items():
            x, m = sample_with(dev, p, EVAL, d, m)
            lat[arm] = x
            im = decode(vae, x.to(dev))
            ims[arm] = im
            ps, lp = [], []
            for i in range(len(im)):
                a, b = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
            hs = install(m, torch.load(p, map_location="cpu"))
            E = torch.stack([(guided_eps(m, stf[i], lab, i, sch) - epsf[i]).double() for i in range(T)], 1)   # [S, T, 4, 32, 32]
            for h in hs:
                h.remove()
            Cc = (T * E.mean(1).pow(2).flatten(1).sum(1))
            I = E.pow(2).flatten(2).sum(2).mean(1)
            carry = torch.load(p, map_location="cpu")
            out["arms"][arm] = {"rel_mse": rel(x, xf).tolist(), "psnr": ps, "lpips": lp, "C": Cc.tolist(), "I": I.tolist(),
                                "C_over_I": float(Cc.sum() / I.sum()), "carry_bytes": int(sum(2 * (c["V"].numel() + c["delta"].numel()) for c in carry.values()))}
            print(f"  {arm}: latent {np.mean(out['arms'][arm]['rel_mse']):.5f} PSNR {np.mean(ps):.2f} LPIPS {np.mean(lp):.4f} C {Cc.mean():.4g} I {I.mean():.4g} C/I {out['arms'][arm]['C_over_I']:.3f}", flush=True)
        out.update(run.record())
    atomic_torch_save({"fp": xf, **lat}, os.path.join(CK, "eval_latents.pt"))
    atomic_torch_save({"fp": imf, **ims}, os.path.join(CK, "eval_images.pt"))
    out["paired"] = paired_table(out["arms"])
    atomic_write_json(os.path.join(OUT, "eval.json"), out)
    for k, v in out["paired"].items():
        print(f"  {k}: latent {100 * v['latent']['rel_change']:+.2f}% [{100 * v['latent']['ci'][0]:+.2f}, {100 * v['latent']['ci'][1]:+.2f}], "
              f"PSNR {v['psnr']['diff']:+.3f} [{v['psnr']['ci'][0]:+.3f}, {v['psnr']['ci'][1]:+.3f}], LPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['ci'][0]:+.4f}, {v['lpips']['ci'][1]:+.4f}]", flush=True)


def boot(a, b, ratio, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    g = np.random.default_rng(0)
    idx = [g.integers(0, len(a), len(a)) for _ in range(n)]
    if ratio:
        d = [a[i].mean() / b[i].mean() - 1 for i in idx]
        return {"rel_change": float(a.mean() / b.mean() - 1), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    d = [np.mean(a[i] - b[i]) for i in idx]
    return {"diff": float(np.mean(a - b)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def paired_table(A):
    pairs = [(a, "C0") for a in ("C1", "C2", "C3")] + [("C2", "C1"), ("C3", "C2")]
    out = {}
    for a, b in pairs:
        if a in A and b in A:
            n = min(len(A[a]["C"]), len(A[b]["C"]))
            out[f"{a}_vs_{b}"] = {"latent": boot(A[a]["rel_mse"], A[b]["rel_mse"], True), "psnr": boot(A[a]["psnr"], A[b]["psnr"], False),
                                  "lpips": boot(A[a]["lpips"], A[b]["lpips"], False), "C": boot(A[a]["C"][:n], A[b]["C"][:n], True)}
    return out


def plan(dev):
    """T_train from the profile and the remaining f4 budget (≤ 0.3 h per arm, the protocol value), written before design."""
    prof = json.load(open(os.path.join(OUT, "profile.json")))
    left = 4.0 - spent() - 0.25                                               # keep the diagnostics/reporting reserve
    fixed = 0.05 + 0.15 + 0.03                                                # design sampling + eval + slack
    tt = min(0.3, (left - fixed) / (3 + 1.5))                                 # 3 final runs + 6 short runs of T/4
    res = {"T_train_hours": round(tt, 3), "f4_spent_at_plan": spent(), "left_after_reserve": left,
           "C1_sec_per_update": prof["C1"]["sec_per_update"], "C2_sec_per_update": prof["C2"]["sec_per_update"],
           "expected_updates": {"C1": int(tt * 3600 / prof["C1"]["sec_per_update"]), "C2/C3": int(tt * 3600 / prof["C2"]["sec_per_update"])},
           "meaningful": tt >= 0.1}
    atomic_write_json(os.path.join(OUT, "plan.json"), res)
    print(res, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "cache", "profile", "plan", "design", "final", "eval"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    {"check": check, "cache": cache, "profile": profile, "plan": plan, "design": design, "final": final, "eval": eval_}[a.cmd](dev)


if __name__ == "__main__":
    main()
