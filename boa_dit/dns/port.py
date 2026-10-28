"""Faithful port of DNS (Yang et al., ICML 2026; official code ZephyrYoung-eYuan/DNS_AQE @ 3f02e69) to DiT-XL/2 256, heig-8 W4, DDIM-50,
cfg 1.5 (results/dns/PROTOCOL.md §1). Correspondence to the official code:
  calib_stats   <- dns_slope_calcu_loop (CNC regression, IQR std, kurtosis, DCNC; analytic variance at W_u = 1)
  solve_abar    <- solve_alpha_prod_pai (bounded minimize_scalar, then the root fallback; eta = 1e-20)
  dns_sample    <- aqe_dns_loop (non-var branch) + dns_infer loop; DCNC noise as in model_forward_tea_stu
Port-only differences: the repository's CFG convention (eps channels 0-2 guided, channel 3 from cond), float64 schedule table,
per-image DCNC generators.
"""
import math

import numpy as np
import torch
from scipy import stats
from scipy.optimize import minimize_scalar, root

from boa_dit.model import NULL_CLASS, noise_for_seeds
from boa_dit.model import create_diffusion

T = 50
CFG = 1.5
WU = 0.2
BASE = create_diffusion(timestep_respacing="")                                 # the 1000-step training schedule
AB = np.asarray(BASE.alphas_cumprod, dtype=np.float64)


def tmap(d):
    """step i (0 = noisiest) -> original timestep passed to the denoiser."""
    m = list(d.timestep_map)
    return [int(m[j]) for j in range(len(m))][::-1]


def prev_of(tm, i):
    return tm[i + 1] if i < len(tm) - 1 else -1                                # final: abar_prev = 1 (guided-diffusion / final_alpha_cumprod)


# ------------------------------------------------------------------------------------------------------- calibration
def calib_stats(tea, diff, wu=WU, std_method="iqr"):
    """tea, diff: [N, C, H, W] float arrays (CFG-combined teacher output and student - teacher). Mirrors dns_slope_calcu_loop."""
    C = diff.shape[1]
    slopes, inter, res = [], [], []
    for c in range(C):
        lr = stats.linregress(tea[:, c].ravel(), diff[:, c].ravel())
        slopes.append(lr.slope)
        inter.append(lr.intercept)
        res.append(diff[:, c].ravel() - (lr.slope * tea[:, c].ravel() + lr.intercept))
    r = np.concatenate(res)
    kurt_raw = float(stats.kurtosis(r, fisher=True))
    s_raw = float(np.std(r))
    r = np.clip(r, -6 * 1.29 * s_raw, 6 * 1.29 * s_raw)
    q75, q25 = np.percentile(r, [75, 25])
    s_o = float((q75 - q25) / 1.349) if std_method == "iqr" else float(np.std(r))
    k_o = float(stats.kurtosis(r, fisher=True))
    a, s_u2 = 0.0, 0.0
    if k_o > 0:
        s_u2 = s_o ** 2 * math.sqrt(-k_o / (-6.0 / 5.0))
        a = math.sqrt(3 * s_u2) * wu
        s2 = s_o ** 2 + s_u2                                                 # analytic, at W_u = 1 (as the code)
    else:
        s2 = s_o ** 2
    return {"slope_k": np.array(slopes), "intercept": np.array(inter), "residual_std": math.sqrt(s2), "uniform_scale": a,
            "sigma_o": s_o, "sigma_u2": s_u2, "kurtosis_clipped": k_o, "kurtosis_raw": kurt_raw, "std_raw": s_raw,
            "clip_frac": float(np.mean(np.abs(np.concatenate(res)) > 6 * 1.29 * s_raw))}


# ---------------------------------------------------------------------------------------------------------- schedule
def solve_abar(abar_t, abar_tau, e2, eta=1e-20):
    """official solve_alpha_prod_pai: abar' in [abar_tau, 1] with abar_tau * (1 + C1(abar')^2 e2) = abar'. Returns (abar', used_fallback)."""
    def eq(p):
        C1 = np.sqrt(1 - p) - np.sqrt((p * (1 - abar_t)) / abar_t)
        return abar_tau * (1 + C1 ** 2 * e2) - p

    def obj(x):
        return eq(x) ** 2

    r = minimize_scalar(obj, bounds=(abar_tau, 1), method="bounded")
    if obj(r.x) > eta ** 2:
        best, err = r.x, obj(r.x)
        for g in np.linspace(abar_tau, min(abar_tau * 1.1, 0.99), 50):
            try:
                rr = root(eq, g)
                if rr.success and abar_tau <= rr.x[0] <= 1:
                    e = obj(rr.x[0])
                    if e < err:
                        err, best = e, rr.x[0]
                        if obj(r.x) < eta ** 2:
                            break
            except Exception:
                continue
        return float(best), True
    return float(r.x), False


def schedule(st, tm):
    """per-step DNS coefficients from frozen statistics (no model involvement); also the zero-statistics check."""
    out = []
    for i, t in enumerate(tm):
        p = prev_of(tm, i)
        td = t - p if p >= 0 else t + 1
        s = st[i]
        k = np.asarray(s["slope_k"], dtype=np.float64)
        e2 = float(np.mean((s["residual_std"] ** 2) / (1 + k) ** 2))
        ab_t, ab_prev = AB[t], (AB[p] if p >= 0 else 1.0)
        fb = False
        if p >= td:
            ab_p, fb = solve_abar(ab_t, ab_prev, e2)
        else:
            ab_p = ab_prev
        ws = math.sqrt(ab_p / ab_t)
        we = math.sqrt(1 - ab_p) - math.sqrt(ab_p * (1 - ab_t) / ab_t)
        C2 = 1 + we ** 2 * e2
        idx = np.where(AB - ab_p / C2 >= 0)[0]
        tau = int(idx[-1]) if idx.size else None
        C3 = 1.0 if (i >= len(tm) - 1 or tau is None) else ab_p / AB[tau]
        out.append({"i": i, "t": t, "prev": p, "abar_t": ab_t, "abar_prev": ab_prev, "abar_q": ab_p, "e2": e2, "ws": ws, "we": we,
                    "C2sq": C2, "tau": tau, "C3sq": C3, "solver_fallback": fb, "tau_not_found": tau is None,
                    "rescale_skipped": i >= len(tm) - 1 or tau is None,
                    "a": 0.0 if i >= len(tm) - 1 else float(s["uniform_scale"]),
                    "k": k.tolist(), "b": np.asarray(s["intercept"], dtype=np.float64).tolist()})
    return out


# ----------------------------------------------------------------------------------------------------------- sampler
class CallCounter:
    def __init__(self, model):
        self.n = 0
        self.h = model.register_forward_hook(self._hook)

    def _hook(self, *a):
        self.n += 1

    def remove(self):
        self.h.remove()


def raw_out(model, x, t, y, amp=True):
    n = len(x)
    tt = torch.full((2 * n,), t, device=x.device, dtype=torch.long)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        return model(torch.cat([x, x]), tt, y).float()                        # [2n, 8, H, W], cond half first


def cfg_combine(out, n, cfg=CFG):
    eps, rest = out[:, :3], out[:, 3:]
    c, u = eps[:n], eps[n:]
    return torch.cat([u + cfg * (c - u), rest[:n]], dim=1)


@torch.no_grad()
def dns_sample(model, seeds, labels, sched, dev, trace_x=False, amp=True):
    """DNS inference (official aqe_dns_loop, deterministic branch). sched: output of schedule(). Returns x0 [and per-step traces]."""
    x = noise_for_seeds(seeds, dev)
    n = len(x)
    lab = torch.as_tensor(labels, device=dev, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    gens = [torch.Generator(device=dev).manual_seed(int(s) * 7919 + 7) for s in seeds]
    tr = []
    for s in sched:
        out = raw_out(model, x, s["t"], y, amp)
        if s["a"] != 0:
            a = s["a"]
            noise = torch.stack([(2 * a) * torch.rand((2,) + out.shape[1:], device=dev, generator=g) - a for g in gens])   # [n, 2, 8, H, W]
            out = out + torch.cat([noise[:, 0], noise[:, 1]])                # cond half, uncond half
        e = cfg_combine(out, n)[:, :4]
        k = torch.tensor(s["k"], device=dev, dtype=torch.float32).view(1, 4, 1, 1)
        b = torch.tensor(s["b"], device=dev, dtype=torch.float32).view(1, 4, 1, 1)
        e = (e - b) / (1 + k)
        xh = s["ws"] * x + s["we"] * e
        xn = xh / math.sqrt(s["C3sq"])
        if trace_x:
            ab_t, ab_prev = s["abar_t"], s["abar_prev"]                       # plain DDIM step from the same state and raw prediction
            e0 = cfg_combine(raw_out(model, x, s["t"], y, amp), n)[:, :4] if s["a"] != 0 else (e * (1 + k) + b)
            x_dd = math.sqrt(ab_prev / ab_t) * x + (math.sqrt(1 - ab_prev) - math.sqrt(ab_prev * (1 - ab_t) / ab_t)) * e0
            tr.append({"i": s["i"], "corr_rel": float((xn - x_dd).norm() / x_dd.norm()), "state_rms": float(xn.pow(2).mean().sqrt())})
        x = xn
    return (x, tr) if trace_x else x
