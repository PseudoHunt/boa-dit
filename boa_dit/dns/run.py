"""DNS reproduction and the C-DNS extension on DiT-XL/2 256 heig-8 W4 (results/dns/PROTOCOL.md).
  calib    : 3000 FP samples (3 per class, seeds 40000-42999), then paired teacher/student CFG predictions on forward-noised states
             at all 50 sampler timesteps -> per-step DNS statistics (port.calib_stats), stats.json
  checks   : correctness checks 1-8 (check 9 = scripts/dns_synthetic_check.py)
  design   : seeds 43000-43031: baseline vs DNS endpoints, per-step traces, stability
  sample   : --arm dns|qdrift|cdns_full|cdns_ind --set paired|fid : latents (+ Inception features for fid)
  traces   : C-DNS calibration: paired FP / heig-8 free-running trajectories with eps, plus FP at the quantized states (44000-44063, 44100-44163)
Run: python -m boa_dit.dns.run <cmd> ...
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.dns import port as P
from boa_dit.model import decode, labels_for_seeds, load_dit, load_quant_state, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample, sample_seeds

STAGE = "f6"
OUT = os.path.join(RESULTS, "dns")
CK = os.path.join(CKPT_DIR, "dns")
HEIG = "deflate_heig_r8_w4"
QSTATE = os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt")
CARRY = os.path.join(CKPT_DIR, "temporal", "rebuild_quant", HEIG, "carry.pt")
CAL = list(range(40000, 43000))
DESIGN = list(range(43000, 43032))
TFIT, THELD = list(range(44000, 44064)), list(range(44100, 44164))
PAIRED, FIDS = list(range(20000, 20128)), list(range(30000, 32000))
E3C_CK = os.path.join(CKPT_DIR, "e3c")
QDRIFT = "/home/boa-dit-fw/results/temporal/sampler_corr/qdrift_calibration.json"


def heig(dev):
    m = load_dit(dev)
    load_quant_state(m, QSTATE)
    m.requires_grad_(False)
    hs = install(m, torch.load(CARRY, map_location="cpu"))
    return m, hs


def fp_model(dev):
    m = load_dit(dev)
    m.requires_grad_(False)
    return m


def stats_path():
    return os.path.join(OUT, "stats.json")


def load_stats(rho=None):
    st = json.load(open(stats_path()))["steps"]
    if rho is not None:                                                       # C-DNS: sigma^2 -> rho * sigma^2, everything else DNS
        st = [dict(s, residual_std=float(np.sqrt(r) * s["residual_std"])) for s, r in zip(st, rho)]
    return st


# --------------------------------------------------------------------------------------------------------- calibration
@torch.no_grad()
def calib(dev):
    os.makedirs(CK, exist_ok=True)
    d = make_diffusion(P.T)
    tm = P.tmap(d)
    labels = np.array([j // 3 for j in range(len(CAL))], dtype=np.int64)
    with Run(STAGE, "dns_calib", dev, {"seeds": [CAL[0], CAL[-1]], "labels": "3 per class", "wu": P.WU}) as run:
        fp = fp_model(dev)
        x0 = sample_seeds(fp, d, CAL, P.CFG, 50, partial=os.path.join(CK, "cal_x0.partial"), resume=True, labels=labels, log=False)
        atomic_torch_save({"x0": x0, "labels": labels, "seeds": CAL}, os.path.join(CK, "cal_x0.pt"))
        q, hs = heig(dev)
        steps, bs = [], 100
        for i, t in enumerate(tm):
            g = torch.Generator().manual_seed(45000 + i)
            noise = torch.randn(x0.shape, generator=g)
            ab = P.AB[t]
            tea, dif = [], []
            for b in range(0, len(x0), bs):
                xt = (float(np.sqrt(ab)) * x0[b:b + bs] + float(np.sqrt(1 - ab)) * noise[b:b + bs]).to(dev).float()
                lab = torch.as_tensor(labels[b:b + bs], device=dev)
                y = torch.cat([lab, torch.full_like(lab, 1000)])
                et = P.cfg_combine(P.raw_out(fp, xt, t, y), len(xt))[:, :4]
                es = P.cfg_combine(P.raw_out(q, xt, t, y), len(xt))[:, :4]
                tea.append(et.cpu()); dif.append((es - et).cpu())
            s = P.calib_stats(torch.cat(tea).double().numpy(), torch.cat(dif).double().numpy())
            s = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in s.items()}
            s.update({"i": i, "timestep": t})
            steps.append(s)
            print(f"  step {i} t {t}: k {np.round(s['slope_k'], 4).tolist()} sigma_o {s['sigma_o']:.4f} kurt {s['kurtosis_clipped']:.2f} "
                  f"sigma {s['residual_std']:.4f} a {s['uniform_scale']:.4f}", flush=True)
        rec = run.record()
    atomic_write_json(stats_path(), {"steps": steps, "wu": P.WU, "n": len(CAL), "calibration": "forward-noised FP samples, paired teacher/student",
                                     "official_commit": "3f02e6932edb30f58c0471456501c34e85517e27", **rec})
    sched = P.schedule(steps, tm)
    atomic_write_json(os.path.join(OUT, "schedule.json"), {"steps": sched})


# ------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def checks(dev):
    d = make_diffusion(P.T)
    tm = P.tmap(d)
    res = {}
    with Run(STAGE, "dns_checks", dev, {}) as run:
        q, hs = heig(dev)
        seeds = PAIRED[:4]
        lab = labels_for_seeds(seeds)
        base = ddim_sample(q, d, noise_for_seeds(seeds, dev), lab, P.CFG).cpu()
        e3c = os.path.join(E3C_CK, "eval_latents.pt")
        if os.path.exists(e3c):
            res["c1_base_equals_e3c_C0_latents"] = bool(torch.equal(base, torch.load(e3c)["C0"][:4]))
        res["c1_disabled_is_ddim_sample"] = True                              # the baseline arm calls ddim_sample itself
        zero = [{"slope_k": [0.0] * 4, "intercept": [0.0] * 4, "residual_std": 0.0, "uniform_scale": 0.0}] * P.T
        zs = P.schedule(zero, tm)
        cc = P.CallCounter(q)
        xz = P.dns_sample(q, seeds, lab, zs, dev).cpu()
        res["c8_calls_dns"] = cc.n
        cc.remove()
        res["c2_zero_stats_rel_diff_vs_base"] = float((xz - base).norm() / base.norm())
        res["c2_zero_stats_schedule_identity"] = all(s["abar_q"] == s["abar_prev"] and s["C3sq"] == 1.0 for s in zs)
        # c3: one explicit DDIM step vs guided-diffusion's ddim_sample step from the same state (fp16-free: amp off)
        x = noise_for_seeds(seeds, dev)
        y = torch.cat([torch.as_tensor(lab, device=dev), torch.full((4,), 1000, device=dev)])
        errs, ts_ok, ab_ok = [], True, True
        cap = {}
        hook = lambda i, xx, out: (cap.__setitem__("o", out.clone()), out)[1]
        _, states = ddim_sample(q, d, x, lab, P.CFG, record_states=True, eps_hook=hook)
        for i in (0, 25, 48):
            xi = states[i]
            e = P.cfg_combine(P.raw_out(q, xi, tm[i], y), 4)
            ab_t, ab_p = P.AB[tm[i]], (P.AB[P.prev_of(tm, i)] if i < 49 else 1.0)
            ours = np.sqrt(ab_p / ab_t) * xi + (np.sqrt(1 - ab_p) - np.sqrt(ab_p * (1 - ab_t) / ab_t)) * e[:, :4]
            ref = d.ddim_sample(lambda xx, ts: e, xi, torch.full((4,), 49 - i, device=dev, dtype=torch.long), clip_denoised=False, eta=0.0)["sample"]
            errs.append(float((ours - ref).norm() / ref.norm()))
        res["c3_ddim_step_rel_err_max"] = max(errs)
        res["c4_timestep_is_timestep_map"] = tm == [int(v) for v in list(d.timestep_map)[::-1]]
        res["c4_abar_matches_respaced"] = bool(np.allclose([P.AB[t] for t in tm], np.asarray(d.alphas_cumprod)[::-1], rtol=1e-12, atol=0))
        # c5: CFG convention identical to ddim_sample's model_fn at step 0
        cap2 = {}
        ddim_sample(q, d, x, lab, P.CFG, eps_hook=lambda i, xx, out: (cap2.setdefault(i, out.clone()), out)[1])
        res["c5_cfg_identical"] = bool(torch.equal(cap2[0], P.cfg_combine(P.raw_out(q, x, tm[0], y), 4)))
        cc = P.CallCounter(q)
        ddim_sample(q, d, x, lab, P.CFG)
        res["c8_calls_base"] = cc.n
        cc.remove()
        if os.path.exists(stats_path()):
            sch = json.load(open(os.path.join(OUT, "schedule.json")))["steps"]
            res["c6_schedule_valid"] = all(np.isfinite(s["abar_q"]) and s["abar_prev"] <= s["abar_q"] <= 1 and np.isfinite(s["C3sq"]) for s in sch)
            res["c6_boundary"] = {"penultimate_no_solve": sch[48]["abar_q"] == sch[48]["abar_prev"], "final_no_rescale": sch[49]["C3sq"] == 1.0,
                                  "final_no_dcnc": sch[49]["a"] == 0.0}
            res["c7_stats_sha256"] = hashlib.sha256(open(stats_path(), "rb").read()).hexdigest()
        res.update(run.record())
    res["pass"] = (res["c2_zero_stats_rel_diff_vs_base"] < 1e-4 and res["c2_zero_stats_schedule_identity"] and res["c3_ddim_step_rel_err_max"] < 1e-5
                   and res["c4_timestep_is_timestep_map"] and res["c5_cfg_identical"] and res["c8_calls_dns"] == res["c8_calls_base"] == 50)
    atomic_write_json(os.path.join(OUT, "checks.json"), res)
    print(json.dumps(res, indent=1), flush=True)


# ----------------------------------------------------------------------------------------------------------- design
@torch.no_grad()
def design(dev):
    d = make_diffusion(P.T)
    tm = P.tmap(d)
    sch = P.schedule(load_stats(), tm)
    with Run(STAGE, "dns_design", dev, {"seeds": [DESIGN[0], DESIGN[-1]]}) as run:
        q, hs = heig(dev)
        lab = labels_for_seeds(DESIGN)
        base = ddim_sample(q, d, noise_for_seeds(DESIGN, dev), lab, P.CFG).cpu()
        x, tr = P.dns_sample(q, DESIGN, lab, sch, dev, trace_x=True)
        fp = fp_model(dev)
        xf = ddim_sample(fp, d, noise_for_seeds(DESIGN, dev), lab, P.CFG).cpu()
        rel = lambda a: ((a - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1))
        res = {"finite": bool(torch.isfinite(x).all()), "base_rel_mse": float(rel(base).mean()), "dns_rel_mse": float(rel(x.cpu()).mean()),
               "endpoint_rms": {"fp": float(xf.pow(2).mean().sqrt()), "base": float(base.pow(2).mean().sqrt()), "dns": float(x.pow(2).mean().sqrt())},
               "dns_vs_base_rel": float((x.cpu() - base).norm() / base.norm()), "trace": tr,
               "schedule": [{k: s[k] for k in ("i", "t", "abar_prev", "abar_q", "tau", "C3sq", "e2", "a", "solver_fallback", "tau_not_found", "rescale_skipped")} for s in sch],
               "note": "trace mode makes one extra (diagnostic) denoiser call per step with DCNC on; never used for reported samples", **run.record()}
    atomic_write_json(os.path.join(OUT, "design.json"), res)
    print({k: v for k, v in res.items() if k not in ("trace", "schedule")}, flush=True)


# ---------------------------------------------------------------------------------------------------------- sampling
def qdrift_hook(c):
    def hook(i, xx, out):
        out = out.clone()
        out[:, :4] = out[:, :4] * (1 + c[i])
        return out
    return hook


@torch.no_grad()
def inception_feats(imgs_nhwc, dev, bs=200):
    from pytorch_fid.inception import InceptionV3
    net = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(dev).eval()
    fs = []
    for i in range(0, len(imgs_nhwc), bs):
        x = torch.as_tensor(np.asarray(imgs_nhwc[i:i + bs])).to(dev).permute(0, 3, 1, 2).float() / 255
        fs.append(net(x)[0].squeeze(-1).squeeze(-1).double().cpu())
    return torch.cat(fs).numpy()


def rho_for(arm):
    f = json.load(open(os.path.join(OUT, "cov_fit.json")))
    return f["rho_full"] if arm == "cdns_full" else f["rho_ind"]


@torch.no_grad()
def sample(dev, arm, setname):
    seeds = PAIRED if setname == "paired" else FIDS
    d = make_diffusion(P.T)
    tm = P.tmap(d)
    os.makedirs(os.path.join(CK, setname), exist_ok=True)
    lat_f = os.path.join(CK, setname, f"latents_{arm}.pt")
    with Run(STAGE, f"dns_sample_{arm}_{setname}", dev, {"arm": arm, "seeds": [seeds[0], seeds[-1]]}) as run:
        q, hs = heig(dev)
        part = lat_f + ".partial"
        done = torch.load(part) if os.path.exists(part) else {}
        sch = None if arm == "qdrift" else P.schedule(load_stats(None if arm == "dns" else rho_for(arm)), tm)
        c = json.load(open(QDRIFT))["c"] if arm == "qdrift" else None
        cc = P.CallCounter(q)
        import time
        t_corr = 0.0
        for b in range(0, len(seeds), 32):
            if b in done:
                continue
            s = seeds[b:b + 32]
            lab = labels_for_seeds(s)
            if arm == "qdrift":
                done[b] = ddim_sample(q, d, noise_for_seeds(s, dev), lab, P.CFG, eps_hook=qdrift_hook(c)).cpu()
            else:
                done[b] = P.dns_sample(q, s, lab, sch, dev).cpu()
            atomic_torch_save(done, part)
        x = torch.cat([done[b] for b in sorted(done)])
        rec = {"arm": arm, "set": setname, "denoiser_calls_per_batch": cc.n / max(1, len([b for b in range(0, len(seeds), 32)])), **run.record()}
        cc.remove()
        atomic_torch_save(x, lat_f)
        if setname == "fid":
            del q
            torch.cuda.empty_cache()
            vae = load_vae(dev)
            imgs = torch.cat([decode(vae, x[b:b + 100].to(dev)) for b in range(0, len(x), 100)]).permute(0, 2, 3, 1).numpy()
            del vae
            F = inception_feats(imgs, dev)
            np.save(os.path.join(CK, "fid", f"feats_{arm}.npy"), F.astype(np.float32))
            from boa_dit.eval_fid import REF_STATS, fid_from_stats
            ref = np.load(REF_STATS)
            rec["fid"] = fid_from_stats(F.mean(0), np.cov(F, rowvar=False), ref["mu"], ref["sigma"])
            print(f"  FID-2k {arm}: {rec['fid']:.3f}", flush=True)
    if os.path.exists(lat_f + ".partial"):
        os.remove(lat_f + ".partial")
    atomic_write_json(os.path.join(OUT, f"sample_{arm}_{setname}.json"), rec)


# ------------------------------------------------------------------------------------------------------- C-DNS traces
@torch.no_grad()
def traces(dev):
    d = make_diffusion(P.T)
    tm = P.tmap(d)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "cdns_traces", dev, {"fit": [TFIT[0], TFIT[-1]], "held": [THELD[0], THELD[-1]]}) as run:
        fp = fp_model(dev)
        q, hs = heig(dev)
        out = {}
        for name, seeds in (("fit", TFIT), ("held", THELD)):
            XF, XQ, EF, EQ, EFQ = [], [], [], [], []
            for b in range(0, len(seeds), 32):
                s = seeds[b:b + 32]
                lab = labels_for_seeds(s)
                nz = noise_for_seeds(s, dev)
                cf, cq = {}, {}
                xf0, sf = ddim_sample(fp, d, nz, lab, P.CFG, record_states=True, eps_hook=lambda i, xx, o: (cf.__setitem__(i, o[:, :4].cpu()), o)[1])
                xq0, sq = ddim_sample(q, d, nz, lab, P.CFG, record_states=True, eps_hook=lambda i, xx, o: (cq.__setitem__(i, o[:, :4].cpu()), o)[1])
                y = torch.cat([torch.as_tensor(lab, device=dev), torch.full((len(s),), 1000, device=dev)])
                efq = [P.cfg_combine(P.raw_out(fp, sq[i], tm[i], y), len(s))[:, :4].cpu() for i in range(P.T)]
                XF.append(torch.stack([sf[i].cpu() for i in range(P.T)] + [xf0.cpu()], 1))       # [S, 51, ...]: x_0..x_49 inputs + final
                XQ.append(torch.stack([sq[i].cpu() for i in range(P.T)] + [xq0.cpu()], 1))
                EF.append(torch.stack([cf[i] for i in range(P.T)], 1)); EQ.append(torch.stack([cq[i] for i in range(P.T)], 1))
                EFQ.append(torch.stack(efq, 1))
            out[name] = {k: torch.cat(v) for k, v in (("xf", XF), ("xq", XQ), ("ef", EF), ("eq", EQ), ("efq", EFQ))}
            out[name]["seeds"] = seeds
            print(f"  traces {name} done", flush=True)
        run.record()
    atomic_torch_save(out, os.path.join(CK, "cdns_traces.pt"))


@torch.no_grad()
def score(dev):
    """paired diagnostics on 20000-20127 vs FP: baseline (E3C C0, reused) and every DNS-side arm present; latent rel. MSE, PSNR, LPIPS."""
    import lpips
    E = torch.load(os.path.join(E3C_CK, "eval_latents.pt"))
    lat = {"base": E["C0"]}
    for f in sorted(os.listdir(os.path.join(CK, "paired"))):
        if f.startswith("latents_") and f.endswith(".pt"):
            lat[f[8:-3]] = torch.load(os.path.join(CK, "paired", f))
    with Run(STAGE, "dns_score", dev, {"arms": list(lat)}) as run:
        vae = load_vae(dev)
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        imf = decode(vae, E["fp"].to(dev))
        out = {}
        for k, x in lat.items():
            im = decode(vae, x.to(dev))
            ps, lp = [], []
            for i in range(len(im)):
                a, b = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
                ps.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
                lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
            rel = ((x - E["fp"]).flatten(1).pow(2).sum(1) / E["fp"].flatten(1).pow(2).sum(1)).tolist()
            out[k] = {"rel_mse": rel, "psnr": ps, "lpips": lp}
            print(f"  {k}: latent {np.mean(rel):.5f} PSNR {np.mean(ps):.2f} LPIPS {np.mean(lp):.4f}", flush=True)
        out["run"] = run.record()
    atomic_write_json(os.path.join(OUT, "paired_scores.json"), out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calib", "checks", "design", "sample", "traces", "score"])
    ap.add_argument("--arm")
    ap.add_argument("--set", default="paired")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "sample":
        sample(dev, a.arm, a.set)
    else:
        {"calib": calib, "checks": checks, "design": design, "traces": traces, "score": score}[a.cmd](dev)


if __name__ == "__main__":
    main()
