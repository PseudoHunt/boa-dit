"""Weight-anchor controls (results/weight_anchor_controls/PROTOCOL.md). Stage tag wac1.
Bed and frozen K = 4 coefficients from wa1 (boa_dit.weight_anchor.run). Arms: FP50, Q, A, R, S (stale FP hold), R1 (beta = 1), FP13 (+ FP-k only if R
is faster than FP50). Every arm runs on its own trajectory; every FP anchor is evaluated at that arm's current latent.
Stages: dev (correctness checks C1-C4 + exploratory dev metrics) | timing | conf
Run: python -m boa_dit.weight_anchor.controls <stage>
"""
import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import NULL_CLASS, decode, labels_for_seeds, load_vae, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor.run import BS, T, boot, guided, model_bytes, models, rel

STAGE = "wac1"
OUT = os.path.join(RESULTS, "weight_anchor_controls")
CK = os.path.join(CKPT_DIR, "weight_anchor_controls")
CAP_GPU_H = 0.5
K = 4
DEV, CONF = list(range(92000, 92032)), list(range(93000, 93064))
ARMS = ("FP50", "Q", "A", "R", "S", "R1", "FP13")
FPK_GRID = (10, 13, 16, 20, 25, 30, 35, 40, 45)
NREP = 5


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


def guard(label, est_h):
    s = spent()
    ok = s + est_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "log": []}
    b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h, "ok": ok})
    jdump("budget.json", b)
    print(f"[wac1 budget] {label}: spent {s:.3f} GPU-h, estimate {est_h:.2f} h -> {'OK' if ok else 'STOP (cap)'}", flush=True)
    if not ok:
        sys.exit(5)


def frozen_beta():
    co = jload("coefficients_frozen.json")
    assert co["selected_K"] == K and co["s"] == 1.0
    return {(int(k.split("_")[0][1:]), int(k.split("_")[1][1:])): v for k, v in co["beta_K_deployed"].items()}


@torch.no_grad()
def sample_arm(q, fp, d, noise, labels, arm, beta, k=K, zero_aux=False):
    """One trajectory. Op order for Q/A/R matches weight_anchor.run.live_sample (bitwise). Returns x0 and call counts."""
    x = noise.float().clone()
    n = len(x)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    cnt = {"w4": 0, "fp": 0}
    st = {"err": None, "a": None, "hold": None}
    s = 1.0
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            if arm in ("FP50", "FP13", "FPk"):
                res = guided(fp, xx, ts, y, n)
                cnt["fp"] += 1
                if zero_aux:
                    res = res.clone()
                    res[:, 4:] = 0
                return res
            if arm == "Q" or (arm in ("A", "R", "R1") and _i % k):
                res = guided(q, xx, ts, y, n)
                cnt["w4"] += 1
                if arm in ("R", "R1"):
                    b = beta[(st["a"] // 10, _i - st["a"])] if arm == "R" else 1.0
                    res = res.clone()
                    res[:, :4] -= (s * b) * st["err"]
                return res
            if arm == "S" and _i % k:
                return st["hold"].clone()                                         # no denoiser call; stored guided output
            res = guided(fp, xx, ts, y, n)                                         # anchor: FP at this arm's current latent
            cnt["fp"] += 1
            if arm in ("R", "R1"):
                rq = guided(q, xx, ts, y, n)
                cnt["w4"] += 1
                st["err"], st["a"] = rq[:, :4] - res[:, :4], _i
            if arm == "S":
                st["hold"] = res
            return res
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt


EXPECT = {"FP50": {"w4": 0, "fp": 50}, "Q": {"w4": 50, "fp": 0}, "A": {"w4": 37, "fp": 13}, "R": {"w4": 50, "fp": 13}, "S": {"w4": 0, "fp": 13},
          "R1": {"w4": 50, "fp": 13}, "FP13": {"w4": 0, "fp": 13}}


def gen(q, fp, diffs, seeds, arms, beta, fpk=None):
    lat, cnt = {a: [] for a in arms}, {}
    for b in range(0, len(seeds), BS):
        s = seeds[b:b + BS]
        nz, lab = noise_for_seeds(s, nz_dev()), labels_for_seeds(s)
        for a in arms:
            d = diffs[13] if a == "FP13" else (diffs[fpk] if a == "FPk" else diffs[50])
            x, c = sample_arm(q, fp, d, nz, lab, a, beta)
            lat[a].append(x.cpu())
            cnt[a] = c
        print(f"  seeds {s[0]}-{s[-1]}: {', '.join(arms)} done", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt


def nz_dev():
    return torch.device("cuda:0")


def metrics(lat, dev, ref="FP50"):
    vae = load_vae(dev)
    import lpips
    net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    imf = decode(vae, lat[ref].to(dev))
    out = {}
    for a, x in lat.items():
        if a == ref:
            continue
        im = decode(vae, x.to(dev))
        ps, lp = [], []
        for i in range(len(im)):
            u, v = imf[i:i + 1].to(dev).float(), im[i:i + 1].to(dev).float()
            ps.append(float(10 * torch.log10(255.0 ** 2 / (u - v).pow(2).mean().clamp_min(1e-10))))
            lp.append(float(net(u / 127.5 - 1, v / 127.5 - 1).flatten()[0]))
        out[a] = {"rel_mse": rel(x, lat[ref]), "psnr": ps, "lpips": lp}
        print(f"  {a}: latent rel. MSE {np.mean(out[a]['rel_mse']):.5f}  PSNR {np.mean(ps):.2f}  LPIPS {np.mean(lp):.4f}", flush=True)
    return out


def contrasts(M):
    pairs = [("R", "S"), ("R1", "S"), ("R", "A"), ("S", "A"), ("R", "R1"), ("S", "FP13")] + [(a, "Q") for a in M if a != "Q"]
    out = {}
    for x, y in pairs:
        if x in M and y in M:
            r = {}
            for m, ratio in (("rel_mse", True), ("psnr", False), ("lpips", False)):
                a, b = np.asarray(M[x][m]), np.asarray(M[y][m])
                r[m] = boot(a, b, ratio)
                r[m]["median_diff"] = float(np.median(a - b))
                r[m]["wins_x"] = int(np.sum(a < b) if m != "psnr" else np.sum(a > b))
            out[f"{x}_vs_{y}"] = r
    return out


def reading(c):
    """pre-registered reading of R vs S / R1 vs S."""
    v, lo, hi = c["rel_mse"]["rel_change"], c["rel_mse"]["ci"][0], c["rel_mse"]["ci"][1]
    lp_bad = c["lpips"]["ci"][0] > 0
    if v <= -0.05 and hi < 0 and not lp_bad:
        return "W4-increment signal (>= 5% lower latent MSE than S, CI excludes 0, no clear LPIPS regression)"
    if hi < 0:
        return "increments help, below the 5% threshold" + (" (LPIPS regression)" if lp_bad else "")
    if lo > 0:
        return "S better"
    if lo > -0.05 and hi < 0.05:
        return "no useful increment contribution established (CI within +-5%)"
    return "inconclusive"


@torch.no_grad()
def dev_stage(dev):
    guard("dev+checks", 0.10)
    beta = frozen_beta()
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "dev_checks", dev, {"seeds": [DEV[0], DEV[-1]]}) as run:
        fp, q = models(dev)
        diffs = {50: make_diffusion(50), 13: make_diffusion(13)}
        lat, cnt = gen(q, fp, diffs, DEV, ARMS, beta)
        old = torch.load(os.path.join(CKPT_DIR, "weight_anchor", "live_dev_latents.pt"), map_location="cpu")
        nz, lab = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        xs, cs = sample_arm(q, fp, diffs[50], nz, lab, "S", beta, k=1)
        xz, _ = sample_arm(q, fp, diffs[50], nz, lab, "FP50", beta, zero_aux=True)
        ch = {"C1": {f"{a}=={b}": bool(torch.equal(lat[a], old[b])) for a, b in (("FP50", "fp"), ("Q", "w4"), ("A", "anchor_only"), ("R", "reuse_s1"))},
              "C2_S_K1_equals_FP50": bool(torch.equal(xs.cpu(), lat["FP50"])), "C2_S_K1_counts": cs,
              "C3_counts": cnt, "C3_pass": all(cnt[a] == EXPECT[a] for a in ARMS),
              "C4_aux_zero_equals_FP50": bool(torch.equal(xz.cpu(), lat["FP50"]))}
        ch["pass"] = bool(all(ch["C1"].values()) and ch["C2_S_K1_equals_FP50"] and ch["C3_pass"] and ch["C4_aux_zero_equals_FP50"])
        ch.update(run.record())
        jdump("checks.json", ch)
        print(json.dumps({k: v for k, v in ch.items() if k not in ("config",)}, indent=1), flush=True)
        if not ch["pass"]:
            print("[wac1] checks FAILED: stop", flush=True)
            sys.exit(6)
        atomic_torch_save(lat, os.path.join(CK, "dev_latents.pt"))
        M = metrics(lat, dev)
        out = {"seeds": [DEV[0], DEV[-1]], "note": "exploratory (images used in wa1)", "arms": M, "calls": cnt, "contrasts": contrasts(M)}
        out["readings"] = {k: reading(out["contrasts"][k]) for k in ("R_vs_S", "R1_vs_S")}
        out.update(run.record())
    jdump("dev.json", out)
    for k, v in out["contrasts"].items():
        print(f"  {k}: latent {100 * v['rel_mse']['rel_change']:+.1f}% [{100 * v['rel_mse']['ci'][0]:+.1f}, {100 * v['rel_mse']['ci'][1]:+.1f}] "
              f"LPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['ci'][0]:+.4f}, {v['lpips']['ci'][1]:+.4f}] wins {v['rel_mse']['wins_x']}", flush=True)
    print(out["readings"], flush=True)


def time_arm(q, fp, d, nz, lab, arm, beta, nrep):
    torch.cuda.synchronize()
    sample_arm(q, fp, d, nz, lab, arm, beta)                                      # discarded warm-up
    torch.cuda.synchronize()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    wall, gpu = [], []
    for _ in range(nrep):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        e0.record()
        _, c = sample_arm(q, fp, d, nz, lab, arm, beta)
        e1.record()
        torch.cuda.synchronize()
        wall.append(time.perf_counter() - t0)
        gpu.append(e0.elapsed_time(e1) / 1000)
    peak = torch.cuda.max_memory_allocated()
    return {"wall_s": {"median": float(np.median(wall)), "min": float(np.min(wall)), "max": float(np.max(wall)), "all": wall},
            "cuda_event_s": {"median": float(np.median(gpu)), "min": float(np.min(gpu)), "max": float(np.max(gpu))},
            "working_mem_bytes": int(peak - base), "peak_alloc_bytes_both_models_resident": int(peak), "calls": c}


@torch.no_grad()
def timing(dev):
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        sys.exit(6)
    dv = jload("dev.json")
    beta = frozen_beta()
    est = 0.30
    guard("timing", est)
    with Run(STAGE, "timing", dev, {"batch": [BS, 1], "nrep": NREP}) as run:
        fp, q = models(dev)
        diffs = {50: make_diffusion(50), 13: make_diffusion(13)}
        out = {"device": torch.cuda.get_device_name(0), "conditions": "complete trajectories; warm (1 discarded warm-up per arm and batch size); synchronize around each; "
               "perf_counter wall + CUDA events; no recording; loading and decode excluded; both models resident in the process",
               "resident_model_bytes": {"fp": model_bytes(fp), "w4_sim": model_bytes(q), "both": model_bytes(fp) + model_bytes(q)},
               "models_needed": {"FP50": ["fp"], "FP13": ["fp"], "S": ["fp"], "Q": ["w4_sim"], "A": ["fp", "w4_sim"], "R": ["fp", "w4_sim"], "R1": ["fp", "w4_sim"]},
               "batch32": {}, "batch1": {}}
        for bs, key, seeds in ((BS, "batch32", DEV[:BS]), (1, "batch1", DEV[:1])):
            nz, lab = noise_for_seeds(seeds, dev), labels_for_seeds(seeds)
            for a in ("FP50", "R", "S", "Q", "A", "R1", "FP13"):
                out[key][a] = time_arm(q, fp, diffs[13] if a == "FP13" else diffs[50], nz, lab, a, beta, NREP)
                need = sum(out["resident_model_bytes"][m] for m in out["models_needed"][a])
                out[key][a]["est_needed_mem_bytes (needed resident + working)"] = int(need + out[key][a]["working_mem_bytes"])
                print(f"  {key} {a}: wall median {out[key][a]['wall_s']['median']:.3f} s [{out[key][a]['wall_s']['min']:.3f}, {out[key][a]['wall_s']['max']:.3f}]  "
                      f"working mem {out[key][a]['working_mem_bytes'] / 1e9:.2f} GB", flush=True)
        f50, r = out["batch32"]["FP50"]["wall_s"]["median"], out["batch32"]["R"]["wall_s"]["median"]
        out["fp50_faster_than_R_batch32"] = bool(f50 <= r)
        out["fp50_faster_than_R_batch1"] = bool(out["batch1"]["FP50"]["wall_s"]["median"] <= out["batch1"]["R"]["wall_s"]["median"])
        out["fpk"] = None
        if not out["fp50_faster_than_R_batch32"]:
            nz, lab = noise_for_seeds(DEV[:BS], dev), labels_for_seeds(DEV[:BS])
            grid = {}
            for kk in FPK_GRID:
                grid[kk] = time_arm(q, fp, make_diffusion(kk), nz, lab, "FPk", beta, 3)["wall_s"]
                print(f"  FP-{kk}: {grid[kk]['median']:.3f} s", flush=True)
            fit = [kk for kk in FPK_GRID if grid[kk]["median"] <= r]
            out["fpk"] = {"grid": {str(kk): v for kk, v in grid.items()}, "selected": max(fit) if fit else None,
                          "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        out.update(run.record())
    jdump("timing.json", out)
    print(f"[wac1] FP-50 faster than R (batch 32): {out['fp50_faster_than_R_batch32']}; batch 1: {out['fp50_faster_than_R_batch1']}; FP-k: {out['fpk']}", flush=True)


@torch.no_grad()
def conf(dev):
    tm = jload("timing.json")
    if not tm:
        print("[wac1] timing must run (and freeze any FP-k) before confirmation", flush=True)
        sys.exit(6)
    beta = frozen_beta()
    fpk = tm["fpk"]["selected"] if tm.get("fpk") else None
    arms = ARMS + (("FPk",) if fpk else ())
    guard("conf", 0.12)
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "arms": arms, "fpk": fpk}) as run:
        fp, q = models(dev)
        diffs = {50: make_diffusion(50), 13: make_diffusion(13)}
        if fpk:
            diffs[fpk] = make_diffusion(fpk)
        lat, cnt = gen(q, fp, diffs, CONF, arms, beta, fpk)
        assert all(cnt[a] == EXPECT[a] for a in ARMS)
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt"))
        M = metrics(lat, dev)
        out = {"seeds": [CONF[0], CONF[-1]], "arms": M, "calls": cnt, "fpk": fpk, "contrasts": contrasts(M)}
        out["readings"] = {k: reading(out["contrasts"][k]) for k in ("R_vs_S", "R1_vs_S")}
        out.update(run.record())
    jdump("conf.json", out)
    for k, v in out["contrasts"].items():
        print(f"  {k}: latent {100 * v['rel_mse']['rel_change']:+.1f}% [{100 * v['rel_mse']['ci'][0]:+.1f}, {100 * v['rel_mse']['ci'][1]:+.1f}] "
              f"median diff {v['rel_mse']['median_diff']:+.5f} LPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['ci'][0]:+.4f}, {v['lpips']['ci'][1]:+.4f}] wins {v['rel_mse']['wins_x']}", flush=True)
    print(out["readings"], flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["dev", "timing", "conf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"dev": dev_stage, "timing": timing, "conf": conf}[a.cmd](dev)


if __name__ == "__main__":
    main()
