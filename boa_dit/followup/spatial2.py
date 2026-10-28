"""Follow-up Experiment 2 (results/followup/PROTOCOL.md §2): PixArt-Σ W4A4 (hsvd_r4_b0.5_kv16_u2_rs) spatial shaping.
  probe   : 2A equal-RMS low/high/white perturbations added to the quantized attn2.to_q input (all 28 blocks) at one step;
            relative change of the guided prediction at that step (design[0:8], steps 2/10/17, 4 directions), and resumed
            endpoints (design[0:2] x 2 directions x low/high x 3 steps)
  bench   : rounding-kernel runtime / peak temporary memory, λ=0 identity check
  screen  : 2B on design[8:24]: nearest, adjacent 2x2 tiles at λ ∈ {0.25, 1, 4}; freezes λ*; random grouping at λ*
  confirm : mjhq1k[256:288] (untouched): bf16 refs, nearest, adjacent λ*, random λ*
Run: python -m boa_dit.followup.spatial2 probe|bench|screen|confirm
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
from boa_dit.f4 import STAGE, guard
from boa_dit.objective.d2 import band_masks
from boa_dit.temporal import pixart as PX
from boa_dit.temporal.ext_w4a4 import BASE, install
from boa_dit.w4a4 import common as WC
from boa_dit.w4a4.fake import GS, QMAX, QMIN, QLinear

OUT = os.path.join(RESULTS, "followup", "e2")
CK = os.path.join(CKPT_DIR, "followup", "e2")
G = 64                                                                        # token grid side
PROBE_STEPS, PROBE_DIRS, PROBE_PROMPTS = (2, 10, 17), 4, list(range(8))
DOWN_PROMPTS, DOWN_DIRS = [0, 1], 2
SCREEN = ("design", list(range(8, 24)))
SPEC_PROMPTS = list(range(8, 12))                                             # prediction-error spectra on design[8:12]
CONFIRM = ("mjhq1k", list(range(256, 288)))
LAMS = [0.25, 1.0, 4.0]
GUID = 4.5


class Ctl:
    """global switches read by the to_q modules."""
    mode = "base"                                                             # base | tile | probe
    lam = 0.0
    group = "adjacent"                                                        # adjacent | random
    perm = None
    probe = None                                                              # (kind, dir) or None
    probe_step = -1
    acc = None                                                                # error-statistics accumulator or None
    ties = 0.0
    clip = 0.0


CTL = Ctl()
MASK64 = None


def masks64(dev):
    global MASK64
    if MASK64 is None or MASK64[0].device != torch.device(dev):
        MASK64 = [m.to(dev) for m in band_masks(G)]
    return MASK64


def tile_round(xs, lam, group="adjacent", perm=None):
    """xs [N, T=4096, C] in X̃ units. Each value keeps its dynamic per-token group scale s and legal codes; candidates are
    floor/ceil of xs/s. Per channel and group of 4 tokens, pick the combination minimizing Σe² + λ(Σe)², e = s·code − xs.
    Returns (dequantized, n_ties, n_clipped)."""
    N, T, C = xs.shape
    g = xs.reshape(N, T, C // GS, GS)
    s = (g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / QMAX).expand_as(g).reshape(N, T, C)
    v = xs / s
    fl, ce = torch.floor(v), torch.ceil(v)
    clipped = ((v < QMIN - 1e-4) | (v > QMAX + 1e-4)).sum()                  # real range violations (none: scale = group max/7)
    fl, ce = fl.clamp(QMIN, QMAX), ce.clamp(QMIN, QMAX)
    ef, ec = fl * s - xs, ce * s - xs                                         # physical (X̃) errors of the two candidates
    if group == "adjacent":                                                   # [N, 32, 2, 32, 2, C] -> [N, 1024, 4, C]
        to_t = lambda a: a.reshape(N, G // 2, 2, G // 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(N, -1, 4, C)
        from_t = lambda a: a.reshape(N, G // 2, G // 2, 2, 2, C).permute(0, 1, 3, 2, 4, 5).reshape(N, T, C)
    else:
        inv = torch.argsort(perm)
        to_t = lambda a: a[:, perm].reshape(N, -1, 4, C)
        from_t = lambda a: a.reshape(N, T, C)[:, inv]
    EF, EC = to_t(ef), to_t(ec)
    bits = torch.tensor([[(k >> i) & 1 for i in range(4)] for k in range(16)], device=xs.device, dtype=torch.bool)  # [16, 4]
    best_cost, best_k = None, None
    for k in range(16):                                                       # 16 combinations, no [.., 16, ..] tensor
        e = torch.where(bits[k].view(1, 1, 4, 1), EC, EF)
        cost = e.pow(2).sum(2) + lam * e.sum(2).pow(2)
        if best_cost is None:
            best_cost, best_k = cost, torch.zeros_like(cost, dtype=torch.long)
        else:
            better = cost < best_cost                                         # strict: first minimum wins (floor on ties)
            best_cost = torch.where(better, cost, best_cost)
            best_k = torch.where(better, torch.full_like(best_k, k), best_k)
    sel = bits[best_k]                                                        # [N, tiles, C, 4]
    use_ce = from_t(sel.permute(0, 1, 3, 2))
    code = torch.where(use_ce, ce, fl)
    ties = ((v - torch.floor(v)) == 0.5).sum()
    return code * s, ties, clipped


class QLinearE2(QLinear):
    """attn2.to_q site: base rounding, tile-balanced rounding, or base rounding + an equal-RMS probe added to Q(x̃)."""

    def forward(self, x):
        if CTL.mode == "base" and CTL.acc is None:
            return super().forward(x)
        dt = x.dtype
        xs = x.float() / self.d if self.d is not None else x.float()
        if CTL.mode == "tile":
            perm = CTL.perm.to(xs.device) if CTL.group == "random" else None
            xq, nt, nc = tile_round(xs, CTL.lam, CTL.group, perm)
            CTL.ties += float(nt); CTL.clip += float(nc)
        else:
            sh = xs.shape
            g = xs.reshape(*sh[:-1], sh[-1] // GS, GS)
            s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / QMAX
            xq = (torch.clamp(torch.round(g / s), QMIN, QMAX) * s).reshape(sh)
            if CTL.mode == "probe" and CTL.probe is not None and PX.STATE.step == CTL.probe_step:
                xq = xq + probe_field(xs, xq, *CTL.probe, self.site_id)
        if CTL.acc is not None and PX.STATE.step % 2 == 0 and PX.STATE.step <= 18:
            CTL.acc.add(xs, xq, self.R)
        y = F.linear(xq.to(torch.bfloat16), self.R, self.bias)
        if self.L1 is not None and self.L1.shape[1] > 0:
            y = y + (xs.to(torch.bfloat16) @ self.L1) @ self.L2
        return y.to(dt)


def probe_field(xs, xq, kind, d, site):
    """band-filtered Gaussian field on the 64x64 grid, scaled per (sample, channel) to the RMS of the base A4 error there."""
    N, T, C = xs.shape
    rms = (xq - xs).pow(2).mean(1, keepdim=True).sqrt()                      # [N, 1, C]
    gen = torch.Generator(device=xs.device).manual_seed(1000 * d + site)
    z = torch.randn(N, C, G, G, generator=gen, device=xs.device)
    if kind != "white":
        m = masks64(xs.device)[0 if kind == "low" else 3]
        z = torch.fft.ifft2(torch.fft.fft2(z) * m).real
    z = z.reshape(N, C, T).transpose(1, 2)
    z = z / z.pow(2).mean(1, keepdim=True).sqrt().clamp_min(1e-12)
    return z * rms


class ErrAcc:
    """activation error e = Q(x̃) − x̃ at the modified sites: total energy, 4-band energies (64x64 grid), output error e·Rᵀ bands."""

    def __init__(self, dev):
        self.s = torch.zeros(9, dtype=torch.float64, device=dev)

    def add(self, xs, xq, R):
        N, T, C = xs.shape
        e = (xq - xs)
        o = F.linear(e, R.float())
        m = masks64(xs.device)
        for j, a in enumerate((e, o)):
            A = torch.fft.fft2(a.transpose(1, 2).reshape(N, a.shape[2], G, G))
            P = A.real ** 2 + A.imag ** 2
            for b in range(4):
                self.s[1 + 4 * j + b] += (P * m[b]).sum(dtype=torch.float64) / (G * G)
        self.s[0] += e.pow(2).sum(dtype=torch.float64)

    def summary(self):
        s = self.s.cpu().tolist()
        return {"E_total": s[0], "E_bands": s[1:5], "band1_abs": s[1], "Eout_bands": s[5:9],
                "Eout_band_frac": [v / max(sum(s[5:9]), 1e-30) for v in s[5:9]], "E_band_frac": [v / max(sum(s[1:5]), 1e-30) for v in s[1:5]]}


def setup(dev, keep_fp=False):
    """pipeline with the W4A4 base installed; attn2.to_q modules become QLinearE2. keep_fp: also return a bf16 copy of the
    transformer (for teacher-forced prediction-error spectra)."""
    pipe = WC.load_pipe(dev)
    fp_tr = copy.deepcopy(pipe.transformer) if keep_fp else None
    mods = install(pipe.transformer, "base")
    keys = sorted(torch.load(os.path.join(WC.CK, f"q_{BASE}.pt"), map_location="cpu").keys())
    sites = 0
    for k, m in zip(keys, mods):
        if k.endswith("attn2.to_q"):
            assert m.act_shift is None                                        # signed path
            m.__class__ = QLinearE2
            m.site_id = sites
            sites += 1
    assert sites == 28, sites
    return pipe, fp_tr


def guided(o, B):
    """transformer output [2B, 8, h, w] (uncond rows first) -> guided eps [B, 4, h, w]."""
    o = o[:, :4].float()
    u, c = o[:B], o[B:]
    return u + GUID * (c - u)


# ----------------------------------------------------------------------------------------------------------------- 2A
@torch.no_grad()
def probe(dev):
    if not guard(0.15, "e2 probe"):
        raise SystemExit(5)
    pipe, _ = setup(dev)
    tr = pipe.transformer
    e = WC.load_embeds("design")
    kinds = ["low", "high", "white"]
    S = {k: {t: [] for t in PROBE_STEPS} for k in kinds}                      # per prompt: mean over directions
    with Run(STAGE, "e2_probe", dev, {"prompts": PROBE_PROMPTS, "steps": PROBE_STEPS, "dirs": PROBE_DIRS}) as run:
        CTL.mode = "probe"
        state = {"busy": False}

        def hook(mod, args, kwargs, out):
            t = PX.STATE.step
            if state["busy"] or t not in PROBE_STEPS:
                return
            state["busy"] = True
            try:
                o0 = out[0] if isinstance(out, tuple) else out.sample
                B = o0.shape[0] // 2
                e0 = guided(o0, B)
                CTL.probe_step = t
                for k in kinds:
                    vals = []
                    for d in range(PROBE_DIRS):
                        CTL.probe = (k, d)
                        PX.STATE.step = t                                     # re-entrant call: keep the step index
                        o1 = mod(*args, **kwargs)
                        o1 = o1[0] if isinstance(o1, tuple) else o1.sample
                        vals.append(float((guided(o1, B) - e0).pow(2).sum() / e0.pow(2).sum()))
                    state["cur"][k][t] = float(np.mean(vals))
                CTL.probe = None
            finally:
                PX.STATE.step = t
                state["busy"] = False

        h = tr.register_forward_hook(hook, with_kwargs=True)
        try:
            for p in PROBE_PROMPTS:
                state["cur"] = {k: {} for k in kinds}
                sc = PX.StepCounter(tr)
                # the StepCounter pre-hook would also count the re-entrant probe calls: suspend it during them
                def guarded_pre(mod, args, _o=sc._pre):
                    if not state["busy"]:
                        _o(mod, args)
                tr._forward_pre_hooks[sc.h.id] = guarded_pre
                try:
                    PX.sample_lat(pipe, e, [p], dev)
                finally:
                    sc.remove()
                for k in kinds:
                    for t in PROBE_STEPS:
                        S[k][t].append(state["cur"][k][t])
                print(f"  probe prompt {p}: " + ", ".join(f"{k}@{t} {state['cur'][k][t]:.3e}" for k in kinds for t in PROBE_STEPS), flush=True)
        finally:
            h.remove()
        CTL.probe = None
        # downstream: resumed endpoints
        down = {k: {t: [] for t in PROBE_STEPS} for k in ("low", "high")}
        for p in DOWN_PROMPTS:
            CTL.probe = None
            sc = PX.StepCounter(tr)
            try:
                x_ref = PX.sample_lat(pipe, e, [p], dev).float()
            finally:
                sc.remove()
            for k in ("low", "high"):
                for t in PROBE_STEPS:
                    v = []
                    for d in range(DOWN_DIRS):
                        CTL.probe, CTL.probe_step = (k, d), t
                        sc = PX.StepCounter(tr)
                        try:
                            x1 = PX.sample_lat(pipe, e, [p], dev).float()
                        finally:
                            sc.remove()
                        v.append(float((x1 - x_ref).pow(2).sum() / x_ref.pow(2).sum()))
                    down[k][t].append(float(np.mean(v)))
            print(f"  downstream prompt {p}: " + ", ".join(f"{k}@{t} {down[k][t][-1]:.3e}" for k in down for t in PROBE_STEPS), flush=True)
        CTL.probe, CTL.mode = None, "base"
        res = {"prompts": PROBE_PROMPTS, "steps": list(PROBE_STEPS), "dirs": PROBE_DIRS, "S": S, "downstream": down,
               "downstream_prompts": DOWN_PROMPTS, "downstream_dirs": DOWN_DIRS}
        res["ratio_low_high"] = {}
        for t in PROBE_STEPS:
            lr = np.log(np.asarray(S["low"][t]) / np.asarray(S["high"][t]))
            g = np.random.default_rng(0)
            bs = [lr[g.integers(0, len(lr), len(lr))].mean() for _ in range(2000)]
            dl = np.log(np.asarray(down["low"][t]) / np.asarray(down["high"][t]))
            res["ratio_low_high"][str(t)] = {"pred": float(np.exp(lr.mean())), "ci": [float(np.exp(np.percentile(bs, 2.5))), float(np.exp(np.percentile(bs, 97.5)))],
                                             "endpoint": float(np.exp(dl.mean())), "ordering_survives": bool(np.sign(dl.mean()) == np.sign(lr.mean()))}
        res.update(run.record())
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, "probe.json"), res)
    for t, r in res["ratio_low_high"].items():
        print(f"  step {t}: S_low/S_high {r['pred']:.3f} [{r['ci'][0]:.3f}, {r['ci'][1]:.3f}], endpoint {r['endpoint']:.3f}, ordering survives {r['ordering_survives']}", flush=True)


# ----------------------------------------------------------------------------------------------------------------- 2B
@torch.no_grad()
def bench(dev):
    """λ=0 identity vs nearest rounding; runtime and peak temporary memory of tile_round on one to_q input size (bs 2)."""
    torch.manual_seed(0)
    xs = torch.randn(2, G * G, 1152, device=dev) * torch.rand(1, 1, 1152, device=dev) * 3
    g = xs.reshape(2, G * G, 18, GS)
    s = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / QMAX
    near = (torch.clamp(torch.round(g / s), QMIN, QMAX) * s).reshape(xs.shape)
    perm = torch.randperm(G * G, generator=torch.Generator().manual_seed(0)).to(dev)
    out = {}
    for grp in ("adjacent", "random"):
        q0, nt, nc = tile_round(xs, 0.0, grp, perm)
        out[f"lam0_{grp}_mismatch"] = int((q0 != near).sum())
        out[f"lam0_{grp}_ties"] = int(nt)
        out["clipped"] = int(nc)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); base_mem = torch.cuda.memory_allocated()
        t0 = time.time()
        for _ in range(10):
            tile_round(xs, 1.0, grp, perm)
        torch.cuda.synchronize()
        out[f"{grp}_ms_per_call"] = 100 * (time.time() - t0)
        out[f"{grp}_peak_temp_MiB"] = (torch.cuda.max_memory_allocated() - base_mem) / 2 ** 20
    t0 = time.time()
    for _ in range(10):
        torch.clamp(torch.round(g / s), QMIN, QMAX) * s
    torch.cuda.synchronize()
    out["nearest_ms_per_call"] = 100 * (time.time() - t0)
    # objective check: tile rounding at λ>0 must not increase Σe² + λ(Σe)² over nearest, per tile (it is the exact minimizer)
    for lam in LAMS:
        q, _, _ = tile_round(xs, lam, "adjacent")
        tl = lambda a: a.reshape(2, 32, 2, 32, 2, 1152).permute(0, 1, 3, 2, 4, 5).reshape(2, -1, 4, 1152)
        cost = lambda qq: (tl(qq - xs).pow(2).sum(2) + lam * tl(qq - xs).sum(2).pow(2))
        out[f"lam{lam}_tiles_worse_than_nearest"] = int((cost(q) > cost(near) * (1 + 1e-6) + 1e-12).sum())
        out[f"lam{lam}_energy_ratio_vs_nearest"] = float((q - xs).pow(2).sum() / (near - xs).pow(2).sum())
    os.makedirs(OUT, exist_ok=True)
    atomic_write_json(os.path.join(OUT, "bench.json"), out)
    print(out, flush=True)


def img_path(arm, setname, p):
    return os.path.join(CK, "img", arm, setname, f"{p:05d}.pt")


@torch.no_grad()
def gen(pipe, arm, setname, prompts, dev, spectra=False, fp_tr=None):
    """batch-1 generation for one arm (resumable per prompt). Returns seconds per image (new images only) and error stats."""
    e = WC.load_embeds(setname)
    os.makedirs(os.path.dirname(img_path(arm, setname, 0)), exist_ok=True)
    acc = ErrAcc(dev) if arm != "bf16" else None
    spec = torch.zeros(4, dtype=torch.float64)
    times = []
    for p in prompts:
        f = img_path(arm, setname, p)
        if os.path.exists(f) and not (spectra and p in SPEC_PROMPTS):
            continue
        CTL.acc = acc
        h = None
        if spectra and p in SPEC_PROMPTS and fp_tr is not None:
            m128 = [m.to(dev) for m in band_masks(128)]

            def hook(mod, args, kwargs, out):
                t = PX.STATE.step
                if t % 2 or t > 18:
                    return
                o = out[0] if isinstance(out, tuple) else out.sample
                B = o.shape[0] // 2
                of = fp_tr(*args, **kwargs)
                of = of[0] if isinstance(of, tuple) else of.sample
                d = guided(o, B) - guided(of, B)
                A = torch.fft.fft2(d.float())
                P = A.real ** 2 + A.imag ** 2
                for b in range(4):
                    spec[b] += float((P * m128[b]).sum())
            h = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        torch.cuda.synchronize(); t0 = time.time()
        sc = PX.StepCounter(pipe.transformer)
        try:
            lat = PX.sample_lat(pipe, e, [p], dev)
        finally:
            sc.remove()
            if h is not None:
                h.remove()
        img = PX.decode(pipe, lat)
        torch.cuda.synchronize(); times.append(time.time() - t0)
        CTL.acc = None
        atomic_torch_save({"img": img, "lat": lat.float().cpu()}, f)
    return (float(np.mean(times)) if times else None), (acc.summary() if acc is not None else None), spec.tolist()


def load_set(arm, setname, prompts):
    d = [torch.load(img_path(arm, setname, p)) for p in prompts]
    return torch.cat([x["img"] for x in d]), torch.cat([x["lat"] for x in d])


def fidelity(arm, setname, prompts, dev, net):
    ri, rl = load_set("bf16", setname, prompts)
    qi, ql = load_set(arm, setname, prompts)
    ps, lp = [], []
    for i in range(len(ri)):
        a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
        ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
        lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
    rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
    return {"psnr": ps, "lpips": lp, "latent": rel}


def boot(a, b, ratio=False, n=2000):
    a, b = np.asarray(a), np.asarray(b)
    g = np.random.default_rng(0)
    idx = [g.integers(0, len(a), len(a)) for _ in range(n)]
    if ratio:
        d = [a[i].mean() / b[i].mean() - 1 for i in idx]
        return {"rel_change": float(a.mean() / b.mean() - 1), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
    d = [np.mean(a[i] - b[i]) for i in idx]
    return {"diff": float(np.mean(a - b)), "ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}


def set_arm(arm):
    if arm in ("bf16", "nearest"):
        CTL.mode = "base"
        return
    kind, lam = arm.split("_")
    CTL.mode, CTL.lam = "tile", float(lam)
    CTL.group = "adjacent" if kind == "adj" else "random"
    CTL.perm = torch.randperm(G * G, generator=torch.Generator().manual_seed(0))


def run_arms(arms, setname, prompts, dev, tag, spectra=False):
    """bf16 references first (plain pipeline), then quantized arms; returns per-arm timing / error stats / spectra."""
    info = {}
    if "bf16" in arms:
        pipe = WC.load_pipe(dev)
        s, _, _ = gen(pipe, "bf16", setname, prompts, dev)
        info["bf16"] = {"sec_per_image": s}
        del pipe
        torch.cuda.empty_cache()
    q_arms = [a for a in arms if a != "bf16"]
    if q_arms:
        pipe, fp_tr = setup(dev, keep_fp=spectra)
        for a in q_arms:
            set_arm(a)
            CTL.ties = CTL.clip = 0.0
            s, st, sp = gen(pipe, a, setname, prompts, dev, spectra=spectra, fp_tr=fp_tr)
            info[a] = {"sec_per_image": s, "err": st, "pred_err_spectrum": sp,
                       "pred_err_band_frac": [v / max(sum(sp), 1e-30) for v in sp] if spectra else None,
                       "ties": CTL.ties, "clipped": CTL.clip}
            print(f"  {tag} {a}: {s} s/image; err {st and {k: st[k] for k in ('E_total', 'band1_abs')}}; ties {CTL.ties:.0f}, clipped {CTL.clip:.0f}", flush=True)
        CTL.mode = "base"
        del pipe, fp_tr
        torch.cuda.empty_cache()
    return info


def score(arms, setname, prompts, dev):
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    fid = {a: fidelity(a, setname, prompts, dev, net) for a in arms if a != "bf16"}
    out = {}
    for a, m in fid.items():
        out[a] = {k + "_mean": float(np.mean(v)) for k, v in m.items()}
        out[a].update({k + "_per_prompt": v for k, v in m.items()})
        if a != "nearest":
            out[a]["vs_nearest"] = {"latent": boot(m["latent"], fid["nearest"]["latent"], ratio=True),
                                    "psnr": boot(m["psnr"], fid["nearest"]["psnr"]), "lpips": boot(m["lpips"], fid["nearest"]["lpips"])}
    return out


def screen(dev):
    setname, prompts = SCREEN
    arms = ["bf16", "nearest"] + [f"adj_{l:g}" for l in LAMS]
    if not guard(0.35, "e2 screen"):
        raise SystemExit(5)
    with Run(STAGE, "e2_screen", dev, {"set": setname, "prompts": [prompts[0], prompts[-1]], "lams": LAMS}) as run:
        info = run_arms(arms, setname, prompts, dev, "screen", spectra=True)
        sc = score(arms, setname, prompts, dev)
        lam_star = min(LAMS, key=lambda l: sc[f"adj_{l:g}"]["latent_mean"])   # pre-registered: lowest mean endpoint latent
        rnd = f"rnd_{lam_star:g}"
        info.update(run_arms([rnd], setname, prompts, dev, "screen", spectra=True))
        sc.update({rnd: score(["nearest", rnd], setname, prompts, dev)[rnd]})
        res = {"set": setname, "prompts": prompts, "arms": {a: {**info.get(a, {}), **sc.get(a, {})} for a in arms + [rnd]},
               "lambda_star": lam_star, "frozen": {"adjacent": f"adj_{lam_star:g}", "random": rnd}}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "screen.json"), res)
    for a, r in res["arms"].items():
        if "latent_mean" in r:
            vs = r.get("vs_nearest")
            print(f"  {a}: latent {r['latent_mean']:.5f} PSNR {r['psnr_mean']:.2f} LPIPS {r['lpips_mean']:.4f}" +
                  (f" | vs nearest latent {100 * vs['latent']['rel_change']:+.2f}% [{100 * vs['latent']['ci'][0]:+.2f}, {100 * vs['latent']['ci'][1]:+.2f}]" if vs else ""), flush=True)
    print("  frozen:", res["frozen"], flush=True)


def confirm(dev):
    fz = json.load(open(os.path.join(OUT, "screen.json")))["frozen"]
    setname, prompts = CONFIRM
    arms = ["bf16", "nearest", fz["adjacent"], fz["random"]]
    if not guard(0.45, "e2 confirm"):
        raise SystemExit(5)
    with Run(STAGE, "e2_confirm", dev, {"set": setname, "prompts": [prompts[0], prompts[-1]], "frozen": fz}) as run:
        info = run_arms(arms, setname, prompts, dev, "confirm")
        sc = score(arms, setname, prompts, dev)
        res = {"set": setname, "prompts": prompts, "frozen": fz, "arms": {a: {**info.get(a, {}), **sc.get(a, {})} for a in arms}}
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "confirm.json"), res)
    for a, r in res["arms"].items():
        if "latent_mean" in r:
            vs = r.get("vs_nearest")
            print(f"  {a}: latent {r['latent_mean']:.5f} PSNR {r['psnr_mean']:.2f} LPIPS {r['lpips_mean']:.4f}" +
                  (f" | vs nearest latent {100 * vs['latent']['rel_change']:+.2f}% [{100 * vs['latent']['ci'][0]:+.2f}, {100 * vs['latent']['ci'][1]:+.2f}], "
                   f"PSNR {vs['psnr']['diff']:+.3f} [{vs['psnr']['ci'][0]:+.3f}, {vs['psnr']['ci'][1]:+.3f}], LPIPS {vs['lpips']['diff']:+.4f} [{vs['lpips']['ci'][0]:+.4f}, {vs['lpips']['ci'][1]:+.4f}]" if vs else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["probe", "bench", "screen", "confirm"])
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    {"probe": probe, "bench": bench, "screen": screen, "confirm": confirm}[a.cmd](dev)


if __name__ == "__main__":
    main()
