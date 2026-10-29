"""E3C Part 2 (results/e3c/PROTOCOL.md): rollout refit of H-SVDQuant's rank-4 L1L2 branch on PixArt-Σ 1024² W4A4.
Only the 224 branch pairs (L1 [c_in, 4], L2 [4, c_out]) are trained (fp32 masters, exported bf16 in the original shapes);
W4 codes/scales (R), smoothing d, activation quantizers and every other weight stay frozen. Dynamic INT4 activation
fake-quant uses a straight-through gradient (forward values unchanged). The sampler is the deployed one: diffusers
DPM-Solver++(2M), 20 steps, guidance 4.5; rollouts restore the scheduler's multistep history from the cached base trajectory.
  check   : own loop == pipeline (bitwise); untrained QLinearT == QLinear; rollout from cache == cached states; gradients reach block 0
  cache   : base trajectories (states + DPM history) for training prompts mjhq1k[600:664]; normalization constants
  profile : seconds per update (local, rollout)
  train   : --arm P1|P2 --lam --lr --hours [--tag]
  design  : lr (P1 short runs, design-local loss) then λ (P2 short runs, endpoint latent on design[24:32]); frozen
  final   : P1, P2 at T_train with the frozen values
  gen     : --name N --q path --lo --hi : batch-1 images + latents (seed = prompt index) through ext_w4a4's generator
  score   : PSNR / LPIPS / latent vs bf16 refs, paired vs base, both sets (ImageReward runs separately in its venv)
Run: python -m boa_dit.e3c.pixart_refit <cmd>
"""
import argparse
import copy
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.f5 import STAGE
from boa_dit.temporal import ext_w4a4 as EXT
from boa_dit.temporal import pixart as PX
from boa_dit.w4a4 import common as WC
from boa_dit.w4a4.fake import QLinear, q_act, q_act_unsigned, set_module

OUT = os.path.join(RESULTS, "e3c", "pixart")
CK = os.path.join(CKPT_DIR, "e3c", "pixart")
QBASE = os.path.join(WC.CK, f"q_{EXT.BASE}.pt")
TRAIN = list(range(600, 664))                    # mjhq1k indices (untouched)
TRAIN_SET = "mjhq1k"
DESIGN = list(range(24, 32))                     # `design` prompt set indices (E2 used design[0:24])
SETS = {"mjhq64": (0, 64), "confirm": (288, 352)}
STEPS, G, K = WC.STEPS, WC.GUIDANCE, 5
LRS, LAMS = [1e-5, 3e-5, 1e-4, 3e-4], [0.5, 2.0]          # 1e-5, 3e-5 added by the 2026-10-29 amendment
ACK = {"resolution": None, "aspect_ratio": None}
TAG = ""


def configure_pilot():
    """Part 3: training states from all 32 design prompts; separate outputs; K from the pilot profile once it exists."""
    global OUT, CK, TRAIN, TRAIN_SET, TAG, K
    OUT, CK = os.path.join(RESULTS, "e3c", "pixart_pilot"), os.path.join(CKPT_DIR, "e3c", "pixart_pilot")
    TRAIN, TRAIN_SET, TAG = list(range(32)), "design", "pilot"
    f = os.path.join(OUT, "profile.json")
    K = json.load(open(f))["K"] if os.path.exists(f) else 3


# --------------------------------------------------------------------------------------------- trainable branch
class _STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xs, shift):
        return q_act(xs) if shift is None else q_act_unsigned(xs, shift)

    @staticmethod
    def backward(ctx, g):
        return g, None


class QLinearT(QLinear):
    """QLinear (fake.py) with fp32 master L1/L2 parameters; forward values identical to QLinear with bf16 L1/L2."""

    def __init__(self, lin, q, dev):
        super().__init__(lin, q["R"], None, None, q.get("d"), act_shift=q.get("act_shift"))
        self.L1p = torch.nn.Parameter(q["L1"].float().to(dev))
        self.L2p = torch.nn.Parameter(q["L2"].float().to(dev))
        self.register_buffer("L10", q["L1"].to(torch.bfloat16).to(dev))     # initial branch (drift penalty reference)
        self.register_buffer("L20", q["L2"].to(torch.bfloat16).to(dev))
        self.track, self.dnum, self.dden = False, None, None

    def forward(self, x):
        dt = x.dtype
        xs = x.float() if self.d is None else x.float() / self.d
        xq = _STE.apply(xs, self.act_shift)
        y = F.linear(xq.to(torch.bfloat16), self.R, self.bias)
        xb = xs.to(torch.bfloat16)
        br = (xb @ self.L1p.to(torch.bfloat16)) @ self.L2p.to(torch.bfloat16)
        y = y + br
        if self.track:                                                        # functional drift of the branch output vs the initial branch
            self.dnum = (br - (xb @ self.L10) @ self.L20).float().pow(2).sum()
            self.dden = y.detach().float().pow(2).sum()
        return y.to(dt)


def install_train(tr, dev):
    qs = torch.load(QBASE, map_location="cpu")
    mods = {}
    for key, q in sorted(qs.items()):
        l, ln = key.split(".", 1)
        blk = tr.transformer_blocks[int(l)]
        m = QLinearT(WC.block_linear(blk, ln), q, dev).to(dev)
        set_module(blk, ln, m)
        mods[key] = m
    return mods


def export(mods):
    qs = torch.load(QBASE, map_location="cpu")
    for key, m in mods.items():
        qs[key] = dict(qs[key], L1=m.L1p.detach().to(torch.bfloat16).cpu(), L2=m.L2p.detach().to(torch.bfloat16).cpu())
    return qs


def install_q(tr, qs):
    """deployment path: ext_w4a4's QLinearFB (mode base) with the given branch tensors."""
    dev = next(tr.parameters()).device
    for i, (key, q) in enumerate(sorted(qs.items())):
        l, ln = key.split(".", 1)
        blk = tr.transformer_blocks[int(l)]
        g = lambda k: None if q.get(k) is None else q[k].to(dev)
        set_module(blk, ln, EXT.QLinearFB(WC.block_linear(blk, ln), g("R"), g("L1"), g("L2"), g("d"), act_shift=g("act_shift"), mode="base", site=i).to(dev))


def teacher(dev):
    from diffusers import PixArtTransformer2DModel
    t = PixArtTransformer2DModel.from_pretrained(os.path.join(WC.MODEL_DIR, "transformer"), torch_dtype=torch.bfloat16).to(dev)
    t.requires_grad_(False)
    return t


# --------------------------------------------------------------------------------------------- sampler pieces
def emb(e, idx, dev):
    k = WC.embeds_for(e, idx, dev)
    ehs = torch.cat([k["negative_prompt_embeds"], k["prompt_embeds"]]).to(torch.bfloat16)
    mask = torch.cat([k["negative_prompt_attention_mask"], k["prompt_attention_mask"]])
    return ehs, mask


def pred(tr, lat, t, ehs, mask):
    """the pipeline's transformer call + CFG + learned-sigma split (rows: uncond, cond)."""
    inp = torch.cat([lat, lat])
    ts = t[None].to(lat.device).expand(inp.shape[0]) if torch.is_tensor(t) else torch.tensor([t], device=lat.device).expand(inp.shape[0])
    out = tr(inp, encoder_hidden_states=ehs, encoder_attention_mask=mask, timestep=ts, added_cond_kwargs=ACK, return_dict=False)[0]
    u, c = out.chunk(2)
    return (u + G * (c - u)).chunk(2, dim=1)[0]


def new_sched(pipe, dev, i=0, hist=None):
    s = copy.deepcopy(pipe.scheduler)
    s.set_timesteps(STEPS, device=dev)
    s._step_index = i
    s.lower_order_nums = min(i, s.config.solver_order)
    s.model_outputs = [None] * (s.config.solver_order - 1) + [hist]
    return s


def init_latents(pipe, idx, dev):
    return pipe.prepare_latents(len(idx), 4, 1024, 1024, torch.bfloat16, dev, WC.generators(idx, dev), None)


def run_loop(pipe, tr, lat, ehs, mask, dev, i0=0, i1=STEPS, hist=None, record=False):
    s = new_sched(pipe, dev, i0, hist)
    states, hists = [], []
    for i in range(i0, i1):
        if record:
            states.append(lat.detach().cpu())
            hists.append(None if s.model_outputs[-1] is None else s.model_outputs[-1].detach().cpu())
        lat = s.step(pred(tr, lat, s.timesteps[i], ehs, mask), s.timesteps[i], lat, return_dict=False)[0]
    return (lat, states, hists) if record else lat


def rollout(pipe, tr, lat, hist, i0, k, ehs, mask, dev):
    """k scheduler steps from cached state i0 with its DPM history; returns (eps at the start state, end state)."""
    s = new_sched(pipe, dev, i0, hist)
    e0 = None
    for i in range(i0, i0 + k):
        e = pred(tr, lat, s.timesteps[i], ehs, mask)
        e0 = e if e0 is None else e0
        lat = s.step(e, s.timesteps[i], lat, return_dict=False)[0]
    return e0, lat


def setup(dev, trainable=True):
    WC.embed_all(dev)
    pipe = WC.load_pipe(dev)
    mods = install_train(pipe.transformer, dev) if trainable else None
    return pipe, mods


# --------------------------------------------------------------------------------------------- check, cache, profile
def check(dev):
    with Run(STAGE, f"e3c_px{TAG}_check", dev, {"K": K}) as run:
        e = WC.load_embeds("mjhq1k")
        pipe = WC.load_pipe(dev)
        idx = [TRAIN[0]]
        with torch.no_grad():
            ref = WC.sample(pipe, e, idx, dev)                               # bf16 pipeline output (FP), latent
            ehs, mask = emb(e, idx, dev)
            own = run_loop(pipe, pipe.transformer, init_latents(pipe, idx, dev), ehs, mask, dev)
        res = {"own_loop_bitwise_fp": bool(torch.equal(own, ref.to(own.dtype)))}
        install_q(pipe.transformer, torch.load(QBASE, map_location="cpu"))
        with torch.no_grad():
            refq = WC.sample(pipe, e, idx, dev)
            lat0 = init_latents(pipe, idx, dev)
            ownq, st, hi = run_loop(pipe, pipe.transformer, lat0, ehs, mask, dev, record=True)
        res["own_loop_bitwise_w4a4"] = bool(torch.equal(ownq, refq.to(ownq.dtype)))
        mods = install_train(pipe.transformer, dev)
        with torch.no_grad():
            ownt = run_loop(pipe, pipe.transformer, lat0, ehs, mask, dev)
            res["untrained_qlineart_bitwise"] = bool(torch.equal(ownt, ownq))
            for i0 in (0, 9, 17):
                k = min(K, STEPS - i0)
                _, xe = rollout(pipe, pipe.transformer, st[i0].to(dev), None if hi[i0] is None else hi[i0].to(dev), i0, k, ehs, mask, dev)
                tgt = st[i0 + k].to(dev) if i0 + k < STEPS else ownq
                res[f"rollout_rel_err_from_{i0}"] = float((xe.float() - tgt.float()).norm() / tgt.float().norm())
        pipe.transformer.enable_gradient_checkpointing()
        _, xe = rollout(pipe, pipe.transformer, st[9].to(dev), hi[9].to(dev), 9, K, ehs, mask, dev)
        xe.float().pow(2).mean().backward()
        res["grad_block0_L1"] = float(mods["0.attn1.to_q"].L1p.grad.norm())
        res["grad_block27_L2"] = float(mods["27.ff.net.2"].L2p.grad.norm())
        res["peak_mem_GiB"] = torch.cuda.max_memory_allocated() / 2 ** 30
        exp = export(mods)
        qb = torch.load(QBASE, map_location="cpu")
        install_q(pipe.transformer, exp)
        with torch.no_grad():
            res["exported_branch_sampling_bitwise"] = bool(torch.equal(run_loop(pipe, pipe.transformer, lat0, ehs, mask, dev), ownq))
            res["exported_branch_pipeline_bitwise"] = bool(torch.equal(WC.sample(pipe, e, idx, dev), refq))
        bf = lambda t: t.to(torch.bfloat16)                                   # the deployed base: QLinear casts the stored fp32 branch to bf16
        res["export_bitwise_untrained"] = all(torch.equal(exp[k]["L1"], bf(qb[k]["L1"])) and torch.equal(exp[k]["L2"], bf(qb[k]["L2"])) for k in qb)
        res["pass"] = (res["own_loop_bitwise_w4a4"] and res["exported_branch_sampling_bitwise"] and res["exported_branch_pipeline_bitwise"] and res["untrained_qlineart_bitwise"] and res["export_bitwise_untrained"]
                       and max(v for k, v in res.items() if k.startswith("rollout")) <= 1e-3 and res["grad_block0_L1"] > 0)
        res.update(run.record())
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, "check.json"), res)
    print(res, flush=True)
    if not res["pass"]:
        raise SystemExit(6)


@torch.no_grad()
def cache(dev):
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, f"e3c_px{TAG}_cache", dev, {"train": [TRAIN[0], TRAIN[-1]], "train_set": TRAIN_SET, "design": DESIGN, "K": K}) as run:
        pipe = WC.load_pipe(dev)
        install_q(pipe.transformer, torch.load(QBASE, map_location="cpu"))
        tch = teacher(dev)
        out = {}
        for name, setn, idx in (("train", TRAIN_SET, TRAIN), ("design", "design", DESIGN)):
            e = WC.load_embeds(setn)
            X, H, X0 = [], [], []
            for j in idx:
                ehs, mask = emb(e, [j], dev)
                lat, st, hi = run_loop(pipe, pipe.transformer, init_latents(pipe, [j], dev), ehs, mask, dev, record=True)
                X.append(torch.cat(st)); H.append(torch.cat([torch.zeros_like(st[0]) if h is None else h for h in hi]))
                if name == "design":
                    X0.append(run_loop(pipe, tch, init_latents(pipe, [j], dev), ehs, mask, dev).cpu())
                print(f"  cache {name} {j}", flush=True)
            out[name] = {"x": torch.stack(X), "hist": torch.stack(H), "idx": idx, "set": setn}
            if X0:
                out[name]["x0fp"] = torch.cat(X0)
        # normalization constants on the first 4 training prompts: local, rollout (full) per start step
        e = WC.load_embeds(TRAIN_SET)
        nl, nr = torch.zeros(STEPS), torch.zeros(STEPS)
        for n, j in enumerate(TRAIN[:4]):
            ehs, mask = emb(e, [j], dev)
            for i in range(STEPS):
                k = min(K, STEPS - i)
                x, h = out["train"]["x"][n, i:i + 1].to(dev), out["train"]["hist"][n, i:i + 1].to(dev)
                h = None if i == 0 else h
                es, xs = rollout(pipe, pipe.transformer, x, h, i, k, ehs, mask, dev)
                et, xt = rollout(pipe, tch, x, h, i, k, ehs, mask, dev)
                nl[i] += float((es.float() - et.float()).pow(2).sum()) / 4
                nr[i] += float((xs.float() - xt.float()).pow(2).sum()) / 4
        out["norm"] = {"loc": nl, "roll": nr}
        run.record()
    atomic_torch_save(out, os.path.join(CK, "cache.pt"))
    print("  norm loc", nl[[0, 10, 19]].tolist(), "roll", nr[[0, 10, 19]].tolist(), flush=True)


# --------------------------------------------------------------------------------------------- training
def train(dev, arm, lam, lr, hours, tag, seed=0, pipe=None, tch=None, ckpt_at=(), max_updates=None, mu=0.0):
    C = torch.load(os.path.join(CK, "cache.pt"))
    if pipe is None:
        pipe, _ = setup(dev, trainable=False)
        tch = teacher(dev)
    mods = install_train(pipe.transformer, dev)
    pipe.transformer.enable_gradient_checkpointing()
    params = [p for m in mods.values() for p in (m.L1p, m.L2p)]
    opt = torch.optim.Adam(params, lr=lr)
    e = WC.load_embeds(C["train"]["set"])
    X, H = C["train"]["x"], C["train"]["hist"]
    nl, nr = C["norm"]["loc"], C["norm"]["roll"]
    g = torch.Generator().manual_seed(seed)
    for m in mods.values():
        m.track = mu > 0
    saved = {}
    log, n_up, t0 = [], 0, time.time()
    while time.time() - t0 < hours * 3600 and (max_updates is None or n_up < max_updates):
        n = int(torch.randint(0, len(X), (1,), generator=g))
        i = int(torch.randint(0, STEPS, (1,), generator=g))
        k = min(K, STEPS - i)
        ehs, mask = emb(e, [C["train"]["idx"][n]], dev)
        x, h = X[n, i:i + 1].to(dev), (None if i == 0 else H[n, i:i + 1].to(dev))
        with torch.no_grad():
            et, xt = rollout(pipe, tch, x, h, i, 1 if arm == "P1" else k, ehs, mask, dev)
        if arm == "P1":
            es = pred(pipe.transformer, x, new_sched(pipe, dev).timesteps[i], ehs, mask)
            loss = (es.float() - et.float()).pow(2).sum() / float(nl[i])
            ll = lroll = None
            if mu > 0:
                drift = sum(m.dnum for m in mods.values()) / sum(m.dden for m in mods.values())
                loss = loss + mu * drift
        else:
            es, xs = rollout(pipe, pipe.transformer, x, h, i, k, ehs, mask, dev)
            ll = (es.float() - et.float()).pow(2).sum() / float(nl[i])
            lroll = (xs.float() - xt.float()).pow(2).sum() / float(nr[i])
            loss = ll + lam * lroll
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        n_up += 1
        if n_up in ckpt_at:
            pth = os.path.join(CK, f"q_{tag}_u{n_up}.pt")
            atomic_torch_save(export(mods), pth)
            saved[n_up] = {"q": pth, "seconds": time.time() - t0}
        if n_up % 20 == 0:
            log.append({"update": n_up, "t": time.time() - t0, "loss": float(loss), "local": None if ll is None else float(ll),
                        "roll": None if lroll is None else float(lroll)})
            print(f"    {tag}: update {n_up}, {time.time() - t0:.0f} s, loss {float(loss):.4f}" + ("" if ll is None else f" (local {float(ll):.4f}, roll {float(lroll):.4f})"), flush=True)
    qs = export(mods)
    path = os.path.join(CK, f"q_{tag}.pt")
    atomic_torch_save(qs, path)
    install_q(pipe.transformer, torch.load(QBASE, map_location="cpu"))      # frozen base modules again for the next run
    pipe.transformer.disable_gradient_checkpointing()
    return path, {"arm": arm, "lam": lam, "lr": lr, "hours": hours, "seed": seed, "mu": mu, "updates": n_up, "train_seconds": time.time() - t0,
                  "checkpoints": saved, "log": log}


def profile(dev):
    with Run(STAGE, f"e3c_px{TAG}_profile", dev, {"K": K}) as run:
        pipe, _ = setup(dev, trainable=False)
        tch = teacher(dev)
        res = {}
        for arm in ("P1", "P2"):
            torch.cuda.reset_peak_memory_stats()
            _, lg = train(dev, arm, 1.0, 1e-4, 120 / 3600, f"profile_{arm}", pipe=pipe, tch=tch)
            res[arm] = {"sec_per_update": lg["train_seconds"] / max(lg["updates"], 1), "updates_in_120s": lg["updates"],
                        "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2 ** 30}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "profile.json"), res)
    print(res, flush=True)


@torch.no_grad()
def design_local(dev, pipe, tch, qpath, C):
    install_q(pipe.transformer, torch.load(qpath, map_location="cpu"))
    e = WC.load_embeds("design")
    nl, v = C["norm"]["loc"], []
    ts = new_sched(pipe, dev).timesteps
    for n, j in enumerate(DESIGN):
        ehs, mask = emb(e, [j], dev)
        for i in range(0, STEPS, 4):
            x = C["design"]["x"][n, i:i + 1].to(dev)
            v.append(float((pred(pipe.transformer, x, ts[i], ehs, mask).float() - pred(tch, x, ts[i], ehs, mask).float()).pow(2).sum() / float(nl[i])))
    return float(np.mean(v))


@torch.no_grad()
def design_latent(dev, pipe, qpath, C):
    install_q(pipe.transformer, torch.load(qpath, map_location="cpu"))
    e = WC.load_embeds("design")
    r = []
    for n, j in enumerate(DESIGN):
        ehs, mask = emb(e, [j], dev)
        x = run_loop(pipe, pipe.transformer, init_latents(pipe, [j], dev), ehs, mask, dev).float().cpu()
        f = C["design"]["x0fp"][n:n + 1].float()
        r.append(float((x - f).pow(2).sum() / f.pow(2).sum()))
    return float(np.mean(r))


def design(dev):
    """lr (P1 short runs; design-local loss, guarded by P1's design endpoint <= P0's), then λ (P2 short runs, endpoint latent); frozen.
    Short runs already present (q file + earlier design json) are reused unchanged."""
    tt = json.load(open(os.path.join(OUT, "plan.json")))["T_train_hours"]
    short = tt / 4
    f = os.path.join(OUT, "design.json")
    prev = json.load(open(f)) if os.path.exists(f) else {}
    res = {"T_train_hours": tt, "short_hours": short, "lr": {}, "lam": {}, "previous_frozen": prev.get("lr_star")}
    with Run(STAGE, "e3c_px_design2", dev, {"lrs": LRS, "lams": LAMS, "short_h": short}) as run:
        C = torch.load(os.path.join(CK, "cache.pt"))
        pipe, _ = setup(dev, trainable=False)
        tch = teacher(dev)
        res["P0_design_local"] = prev.get("P0_design_local") or design_local(dev, pipe, tch, QBASE, C)
        res["P0_design_latent"] = prev.get("P0_design_latent") or design_latent(dev, pipe, QBASE, C)
        for lr in LRS:
            p = os.path.join(CK, f"q_design_P1_lr{lr:g}.pt")
            r = dict(prev.get("lr", {}).get(str(lr), {}))
            if not (os.path.exists(p) and r):
                p, lg = train(dev, "P1", 0.0, lr, short, f"design_P1_lr{lr:g}", pipe=pipe, tch=tch)
                r = {"design_local": design_local(dev, pipe, tch, p, C), "updates": lg["updates"]}
            r.setdefault("design_latent", design_latent(dev, pipe, p, C))
            res["lr"][str(lr)] = r
            print(f"  lr {lr:g}: design local {r['design_local']:.4f} latent {r['design_latent']:.4f} (P0 {res['P0_design_local']:.4f} / {res['P0_design_latent']:.4f})", flush=True)
        order = sorted(LRS, key=lambda l: res["lr"][str(l)]["design_local"])
        ok = [l for l in order if res["lr"][str(l)]["design_latent"] <= res["P0_design_latent"]]
        lr_star = ok[0] if ok else min(LRS)                                  # lr used for the P2 screen only; promotion is decided below
        res["lr_star"], res["lr_guard_satisfied"] = lr_star, bool(ok)
        res["P1_selection"] = f"lr {ok[0]:g}" if ok else "untrained (P0): no lr satisfies the endpoint guard"
        for lam in LAMS:
            p = os.path.join(CK, f"q_design_P2_lam{lam:g}_lr{lr_star:g}.pt")
            r = dict(prev.get("lam", {}).get(str(lam), {})) if lr_star == prev.get("lr_star") else {}
            if not r:
                p, lg = train(dev, "P2", lam, lr_star, short, f"design_P2_lam{lam:g}_lr{lr_star:g}", pipe=pipe, tch=tch)
                r = {"design_latent": design_latent(dev, pipe, p, C), "updates": lg["updates"]}
            res["lam"][str(lam)] = r
            print(f"  P2 λ {lam:g} (lr {lr_star:g}): design latent {r['design_latent']:.5f} (P0 {res['P0_design_latent']:.5f})", flush=True)
        res["lam_star"] = min(LAMS, key=lambda l: res["lam"][str(l)]["design_latent"])
        res["P2_guard_satisfied"] = res["lam"][str(res["lam_star"])]["design_latent"] <= res["P0_design_latent"]   # P2 gated on its own endpoint
        res["previous_design"] = prev
        res.update(run.record())
    atomic_write_json(f, res)
    print("  frozen: lr", res["lr_star"], "λ", res["lam_star"], "guard satisfied", res["lr_guard_satisfied"], flush=True)


def final(dev):
    """2026-10-29 (user review): no automatic promotion after a failed guard. An arm trains only if its own design guard passed, or a dated
    decision file results/e3c/pixart/promote.json names it explicitly."""
    dz = json.load(open(os.path.join(OUT, "design.json")))
    pf = os.path.join(OUT, "promote.json")
    promote = json.load(open(pf)) if os.path.exists(pf) else {}
    allowed = {"P1": dz.get("lr_guard_satisfied", False) or "P1" in promote.get("arms", []),
               "P2": dz.get("P2_guard_satisfied", False) or "P2" in promote.get("arms", [])}
    print(f"  promotion: {allowed} (decision file: {promote or 'none'})", flush=True)
    if not any(allowed.values()):
        raise SystemExit(8)
    pipe, _ = setup(dev, trainable=False)
    tch = teacher(dev)
    for arm in ("P1", "P2"):
        f = os.path.join(OUT, f"train_{arm}.json")
        if os.path.exists(f) or not allowed[arm]:
            continue
        lam = 0.0 if arm == "P1" else dz["lam_star"]
        with Run(STAGE, f"e3c_px_train_{arm}", dev, {"lam": lam, "lr": dz["lr_star"], "hours": dz["T_train_hours"]}) as run:
            p, lg = train(dev, arm, lam, dz["lr_star"], dz["T_train_hours"], f"final_{arm}", pipe=pipe, tch=tch)
            lg.update({"q": p, **run.record()})
        atomic_write_json(f, lg)


def plan(dev):
    """T_train rule (PROTOCOL Part 2 amendment): 1.0 h, raised to give 150 P2 updates if needed, capped at 2.0 h."""
    prof = json.load(open(os.path.join(OUT, "profile.json")))
    sp = prof["P2"]["sec_per_update"]
    tt = 1.0 if 3600 / sp >= 150 else min(2.0, 150 * sp / 3600)
    res = {"T_train_hours": round(tt, 3), "P1_sec_per_update": prof["P1"]["sec_per_update"], "P2_sec_per_update": sp,
           "expected_updates": {"P1": int(tt * 3600 / prof["P1"]["sec_per_update"]), "P2": int(tt * 3600 / sp)}}
    atomic_write_json(os.path.join(OUT, "plan.json"), res)
    print(res, flush=True)


# --------------------------------------------------------------------------------------------- generation and scores
NAMES = {"mjhq64": {"ref": "bf16_b1", "P0": "e1_w4a4_base_screen", "P1": "p5_P1_mjhq64", "P2": "p5_P2_mjhq64"},
         "confirm": {"ref": "p5_bf16_c288", "P0": "p5_P0_c288", "P1": "p5_P1_c288", "P2": "p5_P2_c288"}}


@torch.no_grad()
def score(dev):
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    res = {}
    with Run(STAGE, "e3c_px_score", dev, {}) as run:
        for setn, (lo, hi) in SETS.items():
            ri, rl = EXT.load(NAMES[setn]["ref"], lo, hi)
            res[setn] = {}
            for arm in ("P0", "P1", "P2"):
                qi, ql = EXT.load(NAMES[setn][arm], lo, hi)
                ps, lp = [], []
                for i in range(len(ri)):
                    a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
                    ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                    lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
                rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
                res[setn][arm] = {"psnr": ps, "lpips": lp, "latent": rel}
                print(f"  {setn} {arm}: PSNR {np.mean(ps):.2f} LPIPS {np.mean(lp):.4f} latent {np.mean(rel):.4f}", flush=True)
        res["run"] = run.record()
    atomic_write_json(os.path.join(OUT, "scores.json"), res)


# ---------------------------------------------------------------------------------------------
@torch.no_grad()
def gen(dev, name, qpath, lo, hi):
    """batch-1 images + latents through ext_w4a4.gen (the generator used for every earlier MJHQ-64 arm). qpath 'bf16' = reference."""
    with Run(STAGE, f"e3c_px_gen_{name}", dev, {"q": qpath, "prompts": [lo, hi]}) as run:
        pipe = WC.load_pipe(dev)
        if qpath != "bf16":
            install_q(pipe.transformer, torch.load(qpath, map_location="cpu"))
        EXT.gen(name, pipe, dev, lo, hi, pilot=False)
        run.record()


# --------------------------------------------------------------------------------------------- Part 3 pilot (user spec)
PILOT_SET = (256, 288)


def pilot_profile(dev):
    """one-update timing at K = 3 (then 2 if > 3 s or > 20 GiB); writes pilot K into profile.json."""
    global K
    pipe, _ = setup(dev, trainable=False)
    tch = teacher(dev)
    res = {}
    with Run(STAGE, "e3c_pxpilot_profile", dev, {}) as run:
        for k in (3, 2):
            K = k
            torch.cuda.reset_peak_memory_stats()
            _, lg = train(dev, "P2", 0.5, 1e-4, 90 / 3600, f"pilot_profile_K{k}", pipe=pipe, tch=tch)
            sp = lg["train_seconds"] / max(lg["updates"], 1)
            res[f"K{k}"] = {"sec_per_update": sp, "updates_in_90s": lg["updates"], "peak_mem_GiB": torch.cuda.max_memory_allocated() / 2 ** 30}
            print(f"  K={k}: {sp:.2f} s/update, peak {res[f'K{k}']['peak_mem_GiB']:.1f} GiB", flush=True)
            if sp <= 3.0 and res[f"K{k}"]["peak_mem_GiB"] <= 20.0:
                break
        res["K"] = K
        res["rule"] = "K=3 unless an update > 3 s or memory > 20 GiB, then K=2 (kept even if still over; reported)"
        torch.cuda.reset_peak_memory_stats()
        _, lg = train(dev, "P1", 0.0, 1e-4, 60 / 3600, "pilot_profile_P1", pipe=pipe, tch=tch)
        res["P1_sec_per_update"] = lg["train_seconds"] / max(lg["updates"], 1)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "profile.json"), res)


def pilot_lr():
    """2026-10-29 amendment: the pilot uses Part 2's frozen lr (selected on Part 2's design data only)."""
    return json.load(open(os.path.join(RESULTS, "e3c", "pixart", "design.json")))["lr_star"]


def pilot_final(dev):
    global K
    K = json.load(open(os.path.join(OUT, "profile.json")))["K"]
    lr = pilot_lr()
    pipe, _ = setup(dev, trainable=False)
    tch = teacher(dev)
    for arm in ("P1", "P2"):
        f = os.path.join(OUT, f"train_{arm}.json")
        if os.path.exists(f):
            continue
        lam = 0.0 if arm == "P1" else 0.5
        with Run(STAGE, f"e3c_pxpilot_train_{arm}", dev, {"lam": lam, "lr": lr, "hours": 0.4, "K": K, "seed": 0}) as run:
            p, lg = train(dev, arm, lam, lr, 0.4, f"pilot_{arm}", pipe=pipe, tch=tch)
            lg.update({"q": p, "K": K, **run.record()})
        atomic_write_json(f, lg)


@torch.no_grad()
def pilot_score(dev):
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    lo, hi = PILOT_SET
    names = {"ref": "pilot_bf16_256", "base": "pilot_base_256", "C1": "pilot_C1_256", "C2": "pilot_C2_256"}
    res = {}
    with Run(STAGE, "e3c_pxpilot_score", dev, {}) as run:
        ri, rl = EXT.load(names["ref"], lo, hi)
        for arm in ("base", "C1", "C2"):
            qi, ql = EXT.load(names[arm], lo, hi)
            ps, lp = [], []
            for i in range(len(ri)):
                a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
            res[arm] = {"psnr": ps, "lpips": lp, "latent": ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()}
            print(f"  {arm}: PSNR {np.mean(ps):.2f} LPIPS {np.mean(lp):.4f} latent {np.mean(res[arm]['latent']):.4f}", flush=True)
        res["run"] = run.record()
    atomic_write_json(os.path.join(OUT, "scores.json"), res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "cache", "profile", "plan", "design", "final", "gen", "score", "pilot_profile", "pilot_final", "pilot_score"])
    ap.add_argument("--pilot", action="store_true", help="Part 3: design prompts as training states, outputs under pixart_pilot/")
    ap.add_argument("--name")
    ap.add_argument("--q")
    ap.add_argument("--lo", type=int)
    ap.add_argument("--hi", type=int)
    a = ap.parse_args()
    if a.pilot or a.cmd.startswith("pilot"):
        configure_pilot()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "gen":
        gen(dev, a.name, a.q, a.lo, a.hi)
    else:
        {"check": check, "cache": cache, "profile": profile, "plan": plan, "design": design, "final": final, "score": score,
         "pilot_profile": pilot_profile, "pilot_final": pilot_final, "pilot_score": pilot_score}[a.cmd](dev)


if __name__ == "__main__":
    main()
