"""Extension runs on DiT (rule_extension.md): E2 compressed feedback, E3 runtime without diagnostics, E5 site subsets.
Stage-3 configuration (single-table W16A8, smoothed fc1/fc2, guarded feedback); quantizer = dit_act.Quantize, log=False.
Every job pilot-times its first batch, projects its cost and aborts (exit 4) if it would cross the ceiling
(env TEMPORAL_CAP); per-batch partials make it resumable.
  e2 --seeds screen|confirm --arms none,full,mean,dct4,rand4
  e3 --arms none,unif,full,<best compressed>
  e5 --seeds screen|confirm --arms all,fc2,mlp,attn
Run: python -m boa_dit.temporal.ext e2 --seeds screen
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import torch

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_latent_mse import fp_reference, per_image
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample
from boa_dit.temporal.common import CK, OUT, STAGE, boot_reduction
from boa_dit.temporal.decide import CAP, spent
from boa_dit.temporal.dit import diagnostics
from boa_dit.temporal.dit_act import Quantize

EXT = os.path.join(OUT, "ext")
SEEDS = {"screen": (3000, 64), "confirm": (3100, 512)}
SITES = {"all": None, "fc2": ("mlp.fc2",), "mlp": ("mlp.fc1", "mlp.fc2"), "attn": ("attn.qkv", "attn.proj")}


def quantizer(m, dev, arm, family):
    if family == "e5":
        return Quantize(m, "single8", "none", dev, feedback=True, log=False, fb_types=SITES[arm])
    if arm == "none":
        return Quantize(m, "single8", "none", dev, log=False)
    if arm == "unif":
        return Quantize(m, "single8", "unif", dev, log=False)
    if arm == "full":
        return Quantize(m, "single8", "none", dev, feedback=True, log=False)
    return Quantize(m, "single8", "none", dev, feedback=True, log=False, compress=arm)


@torch.no_grad()
def sample_arm(m, d, q, seeds, dev, part, budget_left):
    """Batches of 32 (seeds in order), resumable; pilot-times the first new batch and aborts if the projection exceeds budget_left (h)."""
    done = torch.load(part) if os.path.exists(part) else {}
    t_pilot = None
    for b in range(0, len(seeds), 32):
        if b in done:
            continue
        s = seeds[b:b + 32]
        torch.cuda.synchronize(); t0 = time.time()
        done[b] = ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5).cpu()
        torch.cuda.synchronize()
        if t_pilot is None:
            t_pilot = time.time() - t0
            remaining = (len(seeds) - b - 32) / 32 * t_pilot / 3600
            print(f"  pilot: {t_pilot:.1f} s/batch -> projected {remaining:.3f} h more for this arm", flush=True)
            if remaining > budget_left:
                atomic_torch_save(done, part)
                print("  ABORT: projected cost exceeds the remaining ceiling", flush=True)
                sys.exit(4)
        atomic_torch_save(done, part)
    return torch.cat([done[b] for b in sorted(done)]), t_pilot


@torch.no_grad()
def eval_family(family, seedset, arms, dev):
    s0, n = SEEDS[seedset]
    seeds = list(range(s0, s0 + n))
    with Run(STAGE, f"ext_fp_reference_{seedset}", dev, {"seed_start": s0, "n": n}):   # cached after the first call
        xf, _ = fp_reference(dev, seed_start=s0, n=n)
    xf = xf[:n]
    os.makedirs(EXT, exist_ok=True)
    out_path = os.path.join(EXT, f"{family}_{seedset}.json")
    res = __import__("json").load(open(out_path)) if os.path.exists(out_path) else {"family": family, "seeds": [s0, s0 + n - 1], "arms": {}}
    m, d = load_dit(dev), make_diffusion(50)
    for arm in arms:
        if arm in res["arms"]:
            print(f"  {arm}: already done", flush=True)
            continue
        left = CAP - spent()
        with Run(STAGE, f"ext_{family}_{seedset}_{arm}", dev, {"family": family, "seeds": seedset, "arm": arm}) as run:
            torch.cuda.reset_peak_memory_stats()
            q = quantizer(m, dev, arm, family)
            x, tp = sample_arm(m, d, q, seeds, dev, os.path.join(CK, f"ext_{family}_{seedset}_{arm}.partial"), left)
            state = q.state_bytes()
            peak = torch.cuda.max_memory_allocated()
            rel, _, _ = per_image(x, xf)
            dg = diagnostics(m, d, dev) if seedset == "screen" else None
            q.remove()
            rec = run.record()
        res["arms"][arm] = {"rel_mse_mean": float(rel.mean()), "rel_mse_sem": float(rel.std() / math.sqrt(n)), "rel_mse_per_image": rel.tolist(),
                            "persistent_state_bytes_batch32": state, "persistent_state_bytes_per_image": state / 32, "peak_mem_bytes": peak,
                            "pilot_s_per_batch": tp, "pred_C_over_I": None if dg is None else dg["C_over_I"], "gpu_hours": rec["gpu_hours"]}
        os.remove(os.path.join(CK, f"ext_{family}_{seedset}_{arm}.partial"))
        atomic_write_json(out_path, res)
        a = res["arms"][arm]
        print(f"  {arm}: latent {a['rel_mse_mean']:.5f} ± {a['rel_mse_sem']:.5f}, state {a['persistent_state_bytes_per_image'] / 2**20:.2f} MiB/image, "
              f"peak {peak / 2**30:.2f} GiB" + ("" if dg is None else f", pred C/I {dg['C_over_I']:.4f}") + f", gpu_h {rec['gpu_hours']:.3f}", flush=True)
    A = res["arms"]
    if "none" in A and "full" in A:                                           # gain retained R with bootstrap CI
        en, ef = np.array(A["none"]["rel_mse_per_image"]), np.array(A["full"]["rel_mse_per_image"])
        rng = np.random.default_rng(0)
        idx = [rng.integers(0, n, n) for _ in range(2000)]
        for arm in A:
            if arm in ("none", "full"):
                continue
            ec = np.array(A[arm]["rel_mse_per_image"])
            R = (en.mean() - ec.mean()) / (en.mean() - ef.mean())
            Rs = [(en[i].mean() - ec[i].mean()) / (en[i].mean() - ef[i].mean()) for i in idx]
            A[arm]["gain_retained_R"] = float(R)
            A[arm]["gain_retained_R_ci"] = np.percentile(Rs, [2.5, 97.5]).tolist()
            A[arm]["vs_none"] = boot_reduction(ec, en)
            print(f"  {arm}: R = {R:.3f} [{A[arm]['gain_retained_R_ci'][0]:.3f}, {A[arm]['gain_retained_R_ci'][1]:.3f}]", flush=True)
        A["full"]["vs_none"] = boot_reduction(ef, en)
        atomic_write_json(out_path, res)


@torch.no_grad()
def runtime(arms, dev, reps=5):
    """E3: diagnostics off; identical batch (seeds 0-31); warm-up; reps timed repeats; CUDA events per transformer forward."""
    m, d = load_dit(dev), make_diffusion(50)
    seeds = list(range(32))
    noise, labels = noise_for_seeds(seeds, dev), labels_for_seeds(seeds)
    out_path = os.path.join(EXT, "e3_runtime.json")
    res = {"batch": 32, "steps": 50, "reps": reps, "arms": {}}
    ev = []
    pre = m.register_forward_pre_hook(lambda mod, inp: ev.append([torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)]) or ev[-1][0].record())
    post = m.register_forward_hook(lambda mod, inp, out: ev[-1][1].record())
    with Run(STAGE, "ext_e3_runtime", dev, {"arms": arms, "reps": reps}) as run:
        for arm in ["fp"] + arms:
            q = None if arm == "fp" else quantizer(m, dev, arm, "e2")
            ddim_sample(m, d, noise, labels, 1.5)                             # warm-up
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            e2e, fwd = [], []
            for _ in range(reps):
                ev.clear()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ddim_sample(m, d, noise, labels, 1.5)
                torch.cuda.synchronize(); e2e.append(time.perf_counter() - t0)
                fwd.append(np.mean([a.elapsed_time(b) for a, b in ev]))
            res["arms"][arm] = {"e2e_s_mean": float(np.mean(e2e)), "e2e_s_std": float(np.std(e2e)), "denoiser_ms_per_step_mean": float(np.mean(fwd)),
                                "denoiser_ms_per_step_std": float(np.std(fwd)), "persistent_state_bytes_batch32": 0 if q is None else q.state_bytes(),
                                "peak_mem_bytes": torch.cuda.max_memory_allocated()}
            r = res["arms"][arm]
            print(f"  {arm}: e2e {r['e2e_s_mean']:.2f} ± {r['e2e_s_std']:.2f} s/batch, denoiser {r['denoiser_ms_per_step_mean']:.1f} ms/step, "
                  f"state {r['persistent_state_bytes_batch32'] / 2**20:.0f} MiB/batch, peak {r['peak_mem_bytes'] / 2**30:.2f} GiB", flush=True)
            if q is not None:
                q.remove()
            atomic_write_json(out_path, {**res, "partial": True})
        res.update(run.record())
    pre.remove(); post.remove()
    atomic_write_json(out_path, res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["e2", "e3", "e5"])
    ap.add_argument("--seeds", default="screen", choices=list(SEEDS))
    ap.add_argument("--arms", default=None)
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if a.cmd == "e2":
        eval_family("e2", a.seeds, (a.arms or "none,full,mean,dct4,rand4").split(","), dev)
    elif a.cmd == "e5":
        eval_family("e5", a.seeds, (a.arms or "all,fc2,mlp,attn").split(","), dev)
    else:
        runtime((a.arms or "none,unif,full").split(","), dev)


if __name__ == "__main__":
    main()
