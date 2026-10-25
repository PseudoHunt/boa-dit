"""Gate 2b driver: executes the pre-registered plan in results/gate2b.md on two GPUs, enforcing the 1.5 GPU-h cap.

State is re-derived from result files on every decision, so the driver can be restarted after a crash.
"""
import csv
import json
import os
import subprocess
import sys
import threading
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG_UTC = "2026-10-25T07:29:44Z"
CAP, W3_RESERVE = 1.5, 0.21
EST = {"quant1": 0.105, "quant4": 0.13, "eval512": 0.076, "heldout": 0.02}
G = ["--gain_tag", "gain_iso_rho0.003_fp32", "--gain_stat", "median"]
TAUS = [0.25, 0.5, 0.75]
UNIFORM = {4: "uniform_w4_K1_lam1_brcfg", 3: "uniform_w3_K1_lam1_brcfg"}
LOG = os.path.join(ROOT, "logs", "gate2b_driver.log")
lock = threading.RLock()  # log() is called while holding the lock
launched = set()


def log(msg):
    with lock, open(LOG, "a") as f:
        f.write(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {msg}\n")


def name_of(args):
    sys.path.insert(0, ROOT)
    import argparse
    from boa_dit.quantize import run_name
    ap = argparse.ArgumentParser()
    for k, v in [("--bits", 4), ("--K", 4), ("--lam", 1.0), ("--tau", 1.0), ("--clip_ratio", 0.0)]:
        ap.add_argument(k, type=type(v), default=v)
    for k, v in [("--weighting", "uniform"), ("--branch", "cfg"), ("--gain_tag", "gain_iso_rho0.01"),
                 ("--gain_stat", "mean"), ("--suffix", "")]:
        ap.add_argument(k, default=v)
    return run_name(ap.parse_known_args(args)[0])


def arm_args(arm, tau=None, bits=4):
    b = ["--bits", str(bits)]
    return {"a": ["--weighting", "gates", "--K", "4"] + b,
            "b": ["--weighting", "gain", "--K", "1", "--lam", "0.5"] + G + b,
            "c": ["--weighting", "full", "--K", "4", "--lam", "0.5"] + G + b,
            "d": ["--weighting", "gain", "--K", "1", "--tau", str(tau)] + G + b,
            "e": ["--weighting", "full", "--K", "4", "--tau", str(tau)] + G + b,
            "f": ["--weighting", "gain", "--K", "1", "--clip_ratio", "10"] + G + b}[arm]


def lat(name, heldout=False):
    p = os.path.join(ROOT, "results", "stage2", "eval", f"latent_{name}{'_heldout1032n64' if heldout else ''}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def spent(names):
    tot = 0.0
    for r in csv.DictReader(open(os.path.join(ROOT, "results", "gpu_hours.csv"))):
        if r["start_utc"] >= PREREG_UTC and any(n in r["run"] for n in names):
            tot += float(r["gpu_hours"])
    return tot


def run(dev, args, heldout=False):
    env = dict(os.environ)
    if heldout:
        env.update(EVAL_ARGS="--seed_start 1032 --n 64", EVAL_TAG="_heldout1032n64")
    log(f"{dev} START {name_of(args)}{' (held-out)' if heldout else ''}")
    subprocess.run(["bash", os.path.join(ROOT, "scripts", "run_q_eval.sh"), dev, "stage2", *args], env=env,
                   stdout=open(os.path.join(ROOT, "logs", f"gate2b_{dev[-1]}.log"), "a"), stderr=subprocess.STDOUT)
    log(f"{dev} END   {name_of(args)}")


def all_names():
    n = [name_of(arm_args("d", t)) for t in TAUS]
    for t in TAUS:
        n.append(name_of(arm_args("e", t)))
    n += [name_of(arm_args(x)) for x in "abcf"] + ["fpref_heldout_gate2b"]
    n += [name_of(arm_args(x, t, 3)) for x in "abcdef" for t in TAUS] + [name_of(arm_args(x, None, 3)) for x in "abcf"]
    return sorted(set(n))


def cost(arm, k4):
    return (EST["quant4"] if k4 else EST["quant1"]) + EST["eval512"]


class Plan:
    def __init__(self):
        self.q = [("tau", t) for t in TAUS]
        self.w3_done = False

    def tau_star(self):
        res = {t: lat(name_of(arm_args("d", t)), heldout=True) for t in TAUS}
        if any(v is None for v in res.values()):
            return None, res
        return min(TAUS, key=lambda t: res[t]["rel_mse_mean"]), res

    def next(self):
        with lock:
            if self.q:
                return self.q.pop(0)
            ts, _ = self.tau_star()
            if ts is None:
                return "wait"
            if not getattr(self, "w4_queued", False):
                self.w4_queued = True
                self.q = [("arm", x) for x in "deacbf"]
                return self.q.pop(0)
            return None


def worker(dev, plan, done_w4):
    # cuda:0 is finishing the Gate 2 pooled baseline; wall-clock GPU-hours would double under contention
    pooled = os.path.join(ROOT, "results", "stage2", "eval", "latent_uniform_w4_K1_lam1_brequal.json")
    while dev == "cuda:0" and not os.path.exists(pooled):
        time.sleep(15)
    log(f"{dev} worker active")
    while True:
        job = plan.next()
        if job == "wait":
            time.sleep(15)
            continue
        if job is None:
            return
        kind, x = job
        if kind == "tau":
            run(dev, arm_args("d", x), heldout=True)
            continue
        ts, _ = plan.tau_star()
        args = arm_args(x, ts)
        k4 = x in "ace"
        need = 0.0 if (x == "d") else cost(x, k4)
        if x == "d":
            need = EST["eval512"]
        with lock:
            s = spent(all_names())
            if s + need + W3_RESERVE > CAP:
                log(f"{dev} SKIP arm {x} (cap): spent {s:.3f} + {need:.3f} + reserve {W3_RESERVE} > {CAP}")
                done_w4.setdefault("skipped", []).append(x)
                continue
        run(dev, args)
        done_w4.setdefault("ran", []).append(x)


def reduction(a, b):
    x, y = np.array(a["rel_mse_per_image"]), np.array(b["rel_mse_per_image"])
    return 1 - x.mean() / y.mean()


def main():
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    plan, done = Plan(), {}
    # held-out FP reference first, so two GPUs never race to create it
    subprocess.run([sys.executable, "-c", "from boa_dit.common import Run; from boa_dit.eval_latent_mse import fp_reference\n"
                    "with Run('stage2', 'eval_latent_fpref_heldout_gate2b', 'cuda:1', {}):\n"
                    "    fp_reference('cuda:1', seed_start=1032, n=64)"], cwd=ROOT)
    log("held-out FP reference ready")
    devs = sys.argv[1:] or ["cuda:0", "cuda:1"]
    th = [threading.Thread(target=worker, args=(d, plan, done)) for d in devs]
    for t in th:
        t.start()
    for t in th:
        t.join()
    ts, _ = plan.tau_star()
    best, best_red = None, -1e9
    for x in "abcdef":
        r = lat(name_of(arm_args(x, ts)))
        if r is None:
            continue
        red = reduction(r, lat(UNIFORM[4]))
        log(f"W4 arm {x}: reduction vs uniform {100 * red:.2f}%")
        if red > best_red:
            best, best_red = x, red
    if best is not None and best_red >= 0.05:
        log(f"best W4 arm {best} ({100 * best_red:.2f}%) passes 5%: running W3 rerun")
        run(devs[0], arm_args(best, ts, bits=3))
        r3 = lat(name_of(arm_args(best, ts, bits=3)))
        log(f"W3 arm {best}: reduction vs uniform {100 * reduction(r3, lat(UNIFORM[3])):.2f}%")
    else:
        log(f"no W4 arm reaches 5% (best {best}: {100 * best_red:.2f}%): no W3 rerun")
    log(f"DONE. GPU-hours spent on the set: {spent(all_names()):.3f}")


if __name__ == "__main__":
    main()
