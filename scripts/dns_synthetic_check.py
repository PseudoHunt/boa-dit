"""Check 9 (results/dns/PROTOCOL.md): one DNS update of the port vs the official aqe_dns_loop (DNS_AQE @ 3f02e69, imported by path),
with a mock model (cond == uncond so both CFG conventions agree, no DCNC noise) and a scheduler carrying the bed's float64 table.
Also check 2's schedule part (zero statistics) and the zero-statistics tau behaviour. CPU only. Run: python scripts/dns_synthetic_check.py"""
import importlib.util, json, math, os, sys, time
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from boa_dit.dns import port as P
from boa_dit.model import make_diffusion

spec = importlib.util.spec_from_file_location("dns_func", "/home/dns_src/DNS_AQE/quant/dns_func.py")
F = importlib.util.module_from_spec(spec); spec.loader.exec_module(F)

class Sch:  # what aqe_dns_loop reads
    alphas_cumprod = torch.tensor(P.AB, dtype=torch.float64)
    final_alpha_cumprod = torch.tensor(1.0, dtype=torch.float64)

tm = P.tmap(make_diffusion(50))
g = torch.Generator().manual_seed(0)
x = torch.randn(3, 4, 32, 32, generator=g, dtype=torch.float32)
half = torch.randn(3, 4, 32, 32, generator=g, dtype=torch.float32)
mock = lambda xx, t, mode, uniform_scale=0, is_var=False, **kw: torch.cat([half, half])
res = {"official_commit": "3f02e6932edb30f58c0471456501c34e85517e27", "steps": {}}
rng = np.random.default_rng(1)
for i in (0, 10, 30, 47, 48, 49):
    st = {"slope_k": rng.normal(0, 0.02, 4), "intercept": rng.normal(0, 0.01, 4), "residual_std": 0.05 + 0.1 * rng.random(), "uniform_scale": 0.0}
    t, p = tm[i], P.prev_of(tm, i)
    p_off = p if i < 49 else t - 50                                    # the official loop's final prev
    xo = F.aqe_dns_loop(timestep=t, diff_slope=st["slope_k"].reshape(1, 4, 1, 1), diff_intercept=st["intercept"].reshape(1, 4, 1, 1),
                        epsilon_delta_t_sq=st["residual_std"] ** 2, prev_timestep=p_off, sample_images=x.clone(), cond_kwargs={"c": None},
                        model_forward_tea_stu=mock, noise_scheduler=Sch, device="cpu", uniform_scale=0.0, cfg=P.CFG, step_index=i, ddim_steps=50)
    s = P.schedule([st] * 50, tm)[i]
    e = P.cfg_combine(torch.cat([half, half]), 3)[:, :4]
    k = torch.tensor(s["k"], dtype=torch.float32).view(1, 4, 1, 1); b = torch.tensor(s["b"], dtype=torch.float32).view(1, 4, 1, 1)
    xp = (s["ws"] * x + s["we"] * ((e - b) / (1 + k))) / math.sqrt(s["C3sq"])
    res["steps"][i] = {"t": t, "prev": p, "rel_diff_vs_official": float((xp - xo.float()).norm() / xo.float().norm()), "abar_q": s["abar_q"],
                       "abar_prev": s["abar_prev"], "tau": s["tau"], "C3sq": s["C3sq"], "fallback": s["solver_fallback"]}
zero = {"slope_k": np.zeros(4), "intercept": np.zeros(4), "residual_std": 0.0, "uniform_scale": 0.0}
z = P.schedule([zero] * 50, tm)
res["zero_stats"] = {"max_abs_abar_q_minus_prev": max(abs(s["abar_q"] - s["abar_prev"]) for s in z),
                     "tau_equals_prev_steps": sum(1 for s in z[:-1] if s["tau"] == s["prev"]), "n_steps_checked": 49,
                     "tau_mismatch": [(s["i"], s["prev"], s["tau"], s["C3sq"]) for s in z[:-1] if s["tau"] != s["prev"]],
                     "max_abs_C3_minus_1": max(abs(s["C3sq"] - 1) for s in z), "ws_we_equal_ddim": max(
                         abs(s["ws"] - math.sqrt(s["abar_prev"] / s["abar_t"])) for s in z)}
res["pass"] = all(v["rel_diff_vs_official"] < 1e-6 for v in res["steps"].values())
os.makedirs("results/dns", exist_ok=True)
json.dump(res, open("results/dns/check_synthetic.json", "w"), indent=1, default=float)
print(json.dumps(res, indent=1, default=float))
