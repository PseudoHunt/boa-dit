"""bar1 steps 3-4: two new training-seed BA/BL replications and the frozen-N fresh evaluation (results/ba_replication/PROTOCOL.md §5-6).
The completed dar1 procedure (boa_dit/anchor_refit/run.py, cd9a052) is reused unchanged; only the draw-list generator seed differs (s = 1, 2; the pilot
is s = 0). Ledger tag bar1 (common budget scripts/h1031b_budget.py).
Stages: repchecks | train <s> | dev <s> | eval | analyze
Run: python -m boa_dit.ba_replication.rep <stage> [s]"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import h1031b_budget as HB
from boa_dit.activation_bridge import run as AB
from boa_dit.anchor_refit import run as R
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install
from boa_dit.dit_joint import run as DJ
from boa_dit.followup.carry_opt import Carry
from boa_dit.model import labels_for_seeds, make_diffusion, noise_for_seeds

TAG = "bar1"
OUT = os.path.join(RESULTS, "ba_replication")
CK = os.path.join(CKPT_DIR, "ba_replication")
DCK = os.path.join(CKPT_DIR, "anchor_refit")
DOUT = os.path.join(RESULTS, "anchor_refit")
N_UPD, CKPTS, TM_CAP = 330, [82, 165, 330], 0.35 * 3600                      # frozen from the pilot profile (not re-profiled)
SEEDS_NEW = (1, 2)
DEV = R.DEV
EVAL_N = 256                                                                  # frozen 2026-10-31 14:45 (PROTOCOL amendment, step 1B)
EVAL = list(range(128000, 128000 + EVAL_N))
BS = 32
EST = {"train_BA": 0.42, "train_BL": 0.38, "dev": 0.14}


def jdump(n, o):
    atomic_write_json(os.path.join(OUT, n), o)


def jload(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def guard(name, est):
    ok, info = HB.ok(TAG, est)
    print(f"[bar1 budget] {name}: {info} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        jdump(f"{name}.json", {"status": "skipped (budget)", **info})
        sys.exit(4)


def seeded_draws(s):
    def draws(n):
        """the pilot draw list with torch Generator seed s (s = 0 reproduces dar1 exactly)."""
        g = torch.Generator().manual_seed(s)
        return [(torch.randint(0, len(R.TRAIN), (R.B,), generator=g).tolist(), R.STARTS[int(torch.randint(0, len(R.STARTS), (1,), generator=g))]) for _ in range(n)]
    return draws


def setup(s):
    """point the dar1 trainer at this branch's checkpoint folder (cache / den linked read-only from dar1, hashes verified) and the seed-s draws."""
    os.makedirs(CK, exist_ok=True)
    art = json.load(open(os.path.join(DOUT, "artifacts.json")))["files"]
    for f in ("anchor_cache.pt", "den.pt"):
        dst = os.path.join(CK, f)
        if not os.path.exists(dst):
            os.symlink(os.path.join(DCK, f), dst)
        assert R.DJ.sha(dst) == art[f]["sha256"], f
    R.CK = CK
    R.draws = seeded_draws(s)


def ckpath(obj, s, u):
    return os.path.join(CK, f"carry_{obj}_s{s}_{'tm' if u == 'tm' else 'u' + str(u)}.pt")


# ------------------------------------------------------------------------------------------------------------------- checks
def stage_repchecks(dev):
    guard("repchecks", 0.08)
    with Run(TAG, "repchecks", dev, {}) as run:
        ch = {"provenance": AB.provenance()}
        ch["hashes_ok"] = all(str(v.get("sha256", "")).startswith(str(v.get("expected"))) for v in ch["provenance"].values() if isinstance(v, dict) and "expected" in v)
        pba, pbl = json.load(open(os.path.join(DOUT, "train_ba.json"))), json.load(open(os.path.join(DOUT, "train_bl.json")))
        setup(0)
        ch["draw_sha_s0_equals_pilot"] = hashlib.sha256(json.dumps(R.draws(R.NMAX)).encode()).hexdigest() == pba["draw_sha"] == pbl["draw_sha"]
        rep = {}
        for obj, pilot in (("BA", pba), ("BL", pbl)):
            r = R.train(dev, obj, n_updates=3, save=False)
            rep[obj] = {"new": [(x["loss"], x["grad_norm"]) for x in r["log"]], "pilot": [(x["loss"], x["grad_norm"]) for x in pilot["log"][:3]]}
            rep[obj]["exact"] = rep[obj]["new"] == rep[obj]["pilot"]
        ch["s0_reproduces_pilot_first3_updates"] = rep
        setup(1)
        sha1 = hashlib.sha256(json.dumps(R.draws(R.NMAX)).encode()).hexdigest()
        setup(2)
        sha2 = hashlib.sha256(json.dumps(R.draws(R.NMAX)).encode()).hexdigest()
        ch["draw_sha"] = {"s1": sha1, "s2": sha2}
        ch["draws_distinct"] = len({sha1, sha2, pba["draw_sha"]}) == 3
        setup(1)
        p1 = R.train(dev, "BA", n_updates=4, save=False, return_params=True)
        p2a = R.train(dev, "BA", n_updates=2, save=False, return_params=True, keep_state=True)
        p2 = R.train(dev, "BA", n_updates=4, save=False, return_params=True, resume_state=p2a["state"])
        ch["resume_bitwise_s1"] = bool(all(torch.equal(a, b) for a, b in zip(p1["params"], p2["params"])))
        # export parity with trained (s1, 4 updates) parameters: exported fp16 carry installed == trainable hook output, one call, bitwise
        d, sch, Ac, bc, m, jq, c, fp = R.trainer(dev)
        for p, v in zip(c.parameters(), p1["params"]):
            p.data.copy_(v.to(dev))
        cache = R.load_cache()
        xs_ = cache["train"][:4, 3].to(dev)
        lab = torch.as_tensor(labels_for_seeds(R.TRAIN[:4]), device=dev)
        with torch.no_grad():
            y_tr = R.stu(m, jq, xs_, lab, 12, sch, False)
        ex = c.export()
        c.remove()
        hs = install(m, ex)
        with torch.no_grad():
            y_in = R.stu(m, jq, xs_, lab, 12, sch, False)
        for h in hs:
            h.remove()
        ch["export_install_bitwise_trained_s1"] = bool(torch.equal(y_tr, y_in))
        # teacher receives no parameter gradient; W4 codes / tables unchanged by a training step
        c2 = Carry(m, torch.load(DJ.BASE_CARRY, map_location="cpu"), dev)
        w0 = m.blocks[0].attn.qkv.weight.detach().clone()
        tabs = DJ.sha(DJ.TABLES)
        loss, cnt = R.loss_ba(m, jq, fp, cache["train"][:4, 0].to(dev), lab, 0, sch, Ac, bc, R.load_den(dev))
        loss.backward()
        ch["teacher_params_no_grad"] = all(p.grad is None for p in fp.parameters())
        ch["window_counts_a0"] = cnt
        ch["base_W4_and_tables_unchanged"] = bool(torch.equal(w0, m.blocks[0].attn.qkv.weight) and DJ.sha(DJ.TABLES) == tabs)
        c2.remove(); jq.remove()
        del m, fp, c, c2, jq
        torch.cuda.empty_cache()
        ch["pass"] = bool(ch["hashes_ok"] and ch["draw_sha_s0_equals_pilot"] and all(v["exact"] for v in rep.values()) and ch["draws_distinct"]
                          and ch["resume_bitwise_s1"] and ch["export_install_bitwise_trained_s1"] and ch["teacher_params_no_grad"]
                          and ch["window_counts_a0"] == {"cheap": 9, "fp": 3, "bridges": 2} and ch["base_W4_and_tables_unchanged"])
        ch.update(run.record())
    jdump("repchecks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1, default=str), flush=True)
    return 0 if ch["pass"] else 6


def need(name, key="pass"):
    r = jload(name)
    if not (r and (r.get(key) or r.get("status") == "complete")):
        print(f"[bar1] {name} required first", flush=True)
        sys.exit(6)
    return r


# ------------------------------------------------------------------------------------------------------------------- training
def stage_train(dev, s):
    need("repchecks.json")
    assert s in SEEDS_NEW
    setup(s)
    out = jload(f"train_s{s}.json") or {"seed": s, "N": N_UPD, "ckpts": CKPTS}
    if "BA" not in out:
        guard(f"train_BA_s{s}", EST["train_BA"])
        with Run(TAG, f"train_BA_s{s}", dev, {"seed": s, "N": N_UPD, "ckpts": CKPTS}) as run:
            r = R.train(dev, "BA", N_UPD, CKPTS, tag=f"BA_s{s}")
            r.update(run.record())
        out["BA"] = finish(r, "BA", s)
        jdump(f"train_s{s}.json", out)
    if "BL" not in out:
        cap = min(out["BA"]["ttrain_s"], TM_CAP)
        guard(f"train_BL_s{s}", EST["train_BL"])
        with Run(TAG, f"train_BL_s{s}", dev, {"seed": s, "N": N_UPD, "ckpts": CKPTS, "time_cap_s": cap}) as run:
            r = R.train(dev, "BL", N_UPD, CKPTS, time_cap_s=cap, tag=f"BL_s{s}")
            r.update(run.record())
        r["time_matched_to_BA_train_s"] = cap
        out["BL"] = finish(r, "BL", s)
        jdump(f"train_s{s}.json", out)
    out["status"] = "complete"
    jdump(f"train_s{s}.json", out)
    print({k: {kk: vv for kk, vv in v.items() if kk != "log"} if isinstance(v, dict) else v for k, v in out.items()}, flush=True)


def finish(r, obj, s):
    """record stability (non-finite loss / grad => training-stability result per the predeclared rule) and checkpoint hashes."""
    fin = all(np.isfinite(x["loss"]) and np.isfinite(x["grad_norm"]) for x in r["log"])
    r["finite"] = bool(fin)
    files = {u: ckpath(obj, s, u) for u in CKPTS + (["tm"] if obj == "BL" else [])}
    r["checkpoints"] = {str(u): {"path": p, "sha256": DJ.sha(p)} for u, p in files.items() if os.path.exists(p)}
    return r


# ------------------------------------------------------------------------------------------------------------------- evaluation
def register(names):
    for k, p in names.items():
        if k == "B0":
            AB.ARMS["B0"] = AB.ARMS["CB"]
        else:
            AB.CARRY[k] = p
            AB.ARMS[k] = AB.A("static", "fp", 4, True, k)


@torch.no_grad()
def gen(M, d, seeds, arms, part=None, bs=BS):
    """generation with per-batch partial saves (resumable); counts asserted per trajectory."""
    P = torch.load(part, map_location="cpu") if part and os.path.exists(part) else {"lat": {a: [] for a in arms}, "done": 0, "timing": {}}
    for b0 in range(P["done"], len(seeds), bs):
        s = seeds[b0:b0 + bs]
        nz, lb = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            sp = AB.ARMS[a]
            torch.cuda.synchronize(); t0 = time.perf_counter()
            x, c = AB.sample(M, sp, d, nz, lb)
            dt = time.perf_counter() - t0
            assert torch.isfinite(x).all() and c == AB.expect(sp), (a, c)
            P["lat"][a].append(x.cpu()); P["timing"].setdefault(a, []).append(dt)
            print(f"  seeds {s[0]}-{s[-1]} {a}: {dt:.1f} s", flush=True)
        P["done"] = b0 + len(s)
        if part:
            atomic_torch_save(P, part)
    return {a: torch.cat(v) for a, v in P["lat"].items()}, P["timing"]


def stage_dev(dev, s):
    tr = need(f"train_s{s}.json")
    names = {"FP50": None}
    elig = {}
    for obj in ("BA", "BL"):
        bad = [x["update"] for x in tr[obj]["log"] if not (np.isfinite(x["loss"]) and np.isfinite(x["grad_norm"]))]
        elig[obj] = [u for u in CKPTS if not bad or u < bad[0]]                  # predeclared rule: checkpoints after a non-finite update are not used
        for u in elig[obj]:
            names[f"{obj}_s{s}@{u}"] = ckpath(obj, s, u)
    names[f"BL_s{s}-tm"] = ckpath("BL", s, "tm")
    register({k: v for k, v in names.items() if v})
    arms = list(names)
    guard(f"dev_s{s}", EST["dev"])
    with Run(TAG, f"dev_s{s}", dev, {"seeds": [DEV[0], DEV[-1]], "arms": arms}) as run:
        M = AB.load_models(dev, ("fp", "static"))
        d = make_diffusion(R.T)
        lat, tim = gen(M, d, DEV, arms)
        del M
        torch.cuda.empty_cache()
        prev = torch.load(os.path.join(DCK, "dev_latents.pt"), map_location="cpu")
        fp_equal = bool(torch.equal(lat["FP50"], prev["FP50"]))
        lat.update({"B0": prev["B0"], "BL-arch": prev["BL-arch"]})               # pilot dev outputs of the fixed comparators (same noise / harness)
        atomic_torch_save(lat, os.path.join(CK, f"dev_latents_s{s}.pt"))
        Mx, _ = AB.evaluate(lat, dev, [])
        from boa_dit.weight_anchor.shared import summary
        S = summary(Mx)
        mean = {a: S[a]["rel_mse"]["mean"] for a in S}

        def pick(pre):
            us = elig[pre.split("_")[0]]
            best = min(mean[f"{pre}@{u}"] for u in us)
            return next(u for u in us if mean[f"{pre}@{u}"] <= best * (1 + 1e-6))
        sel = {"BA": f"BA_s{s}@{pick(f'BA_s{s}')}", "BL": f"BL_s{s}@{pick(f'BL_s{s}')}", "BL-tm": f"BL_s{s}-tm"}
        out = {"status": "complete", "seed": s, "dev_seeds": [DEV[0], DEV[-1]], "arms": Mx, "summary": S, "selected": sel, "FP50_equals_pilot_dev_bitwise": fp_equal,
               "rule": "per objective lowest dev mean endpoint rel. MSE among {82,165,330}, ties (1e-6 rel.) to the earlier; no promotion gate",
               "selected_paths": {k: AB.CARRY[v] for k, v in sel.items()}, "selected_sha256": {k: DJ.sha(AB.CARRY[v]) for k, v in sel.items()},
               "timing": tim, **run.record()}
    jdump(f"dev_s{s}.json", out)
    for a in S:
        print(f"  {a}: rel MSE {S[a]['rel_mse']['mean']:.5f} LPIPS {S[a]['lpips']['mean']:.4f}", flush=True)
    print("selected", sel, flush=True)


def eval_arms():
    names = {"B0": None, "BL-arch": R.ARCH}
    roles = {"B0": "B0", "BL-arch": "BL-arch"}
    for s in SEEDS_NEW:
        dv = need(f"dev_s{s}.json")
        for role, arm in dv["selected"].items():
            names[f"{role}_s{s}"] = dv["selected_paths"][role]
            roles[f"{role}_s{s}"] = arm
    pl = jload("eval_plan.json")
    if pl and pl.get("include_pilot"):
        for k, f in (("BA_s0", "carry_BA_u330.pt"), ("BL_s0", "carry_BL_u330.pt"), ("BL-tm_s0", "carry_BL_tm.pt")):
            names[k] = os.path.join(DCK, f)
            roles[k] = f"pilot {f}"
    return names, roles


def stage_plan(dev=None):
    """§3 pilot-arm decision by measured spend only (before any fresh generation)."""
    for s in SEEDS_NEW:
        need(f"dev_s{s}.json")
    rate_arm, rate_fp = (0.3594 - 0.022) / 5 / 128, 0.022 / 128
    req = EVAL_N * (8 * rate_arm + rate_fp)
    pil = EVAL_N * 3 * rate_arm
    sp = HB.spent()
    kernel_left = max(0.0, HB.PLAN["wkf1"] - HB.spent("wkf1")) if not os.path.exists("/home/boa-dit-wkf/results/w8_kernel/bench.json") else 0.0
    inc = sp + kernel_left + req + pil + 0.25 <= HB.TOTAL_CAP
    out = {"spent_total": sp, "kernel_track_remaining_reserved": kernel_left, "required_eval_h": req, "pilot_arms_h": pil, "include_pilot": bool(inc),
           "rule": "include pilot arms iff spent + kernel remaining + required + pilot + 0.25 <= 4.0 (PROTOCOL §3)", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    jdump("eval_plan.json", out)
    print(out, flush=True)


def stage_eval(dev):
    need("eval_plan.json", key="utc")
    names, roles = eval_arms()
    register({k: v for k, v in names.items() if v})
    arms = ["FP50"] + list(names)
    n_anch = len(names)
    est = EVAL_N * (n_anch * (0.3594 - 0.022) / 5 / 128 + 0.022 / 128) * 1.05
    part = os.path.join(CK, "eval_part.pt")
    if os.path.exists(part):
        done = torch.load(part, map_location="cpu")["done"]
        est *= (EVAL_N - done) / EVAL_N
    guard("eval", est)
    with Run(TAG, "eval", dev, {"seeds": [EVAL[0], EVAL[-1]], "arms": arms, "roles": roles}) as run:
        M = AB.load_models(dev, ("fp", "static"))
        d = make_diffusion(R.T)
        lat, tim = gen(M, d, EVAL, arms, part=part)
        del M
        torch.cuda.empty_cache()
        atomic_torch_save(lat, os.path.join(CK, "eval_latents.pt"))
        Mx, _ = AB.evaluate(lat, dev, [])
        out = {"status": "complete", "seeds": [EVAL[0], EVAL[-1]], "n": EVAL_N, "arms": Mx, "roles": roles, "carry_sha256": {k: DJ.sha(v) for k, v in names.items() if v},
               "calls": {a: AB.expect(AB.ARMS[a]) for a in arms}, "timing": tim, **run.record()}
    jdump("eval.json", out)
    stage_analyze()


def stage_analyze(dev=None):
    ev = need("eval.json")
    Mx = {a: {m: np.asarray(v[m], float) for m in ("rel_mse", "psnr", "lpips")} for a, v in ev["arms"].items()}
    n = len(Mx["B0"]["rel_mse"])
    g = np.random.default_rng(0)
    idx = np.stack([g.integers(0, n, n) for _ in range(10000)])               # joint image bootstrap: same image IDs for every arm and pair
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    seeds = [s for s in (1, 2, 0) if f"BA_s{s}" in Mx]

    def contrast(x, y):
        a, b = Mx[x]["rel_mse"], Mx[y]["rel_mse"]
        dd = a - b
        r = {"ratio": float(a.mean() / b.mean()), "ratio_ci": ci(a[idx].mean(1) / b[idx].mean(1)), "mean_diff": float(dd.mean()), "mean_diff_ci": ci(dd[idx].mean(1)),
             "median_diff": float(np.median(dd)), "wins_x": int((a < b).sum()), "n": n}
        for m in ("lpips", "psnr"):
            dm = Mx[x][m] - Mx[y][m]
            r[f"d{m}"] = float(dm.mean()); r[f"d{m}_ci"] = ci(dm[idx].mean(1))
        return r
    C = {}
    for s in seeds:
        for comp in (f"BL_s{s}", f"BL-tm_s{s}", "BL-arch", "B0"):
            C[f"BA_s{s}_vs_{comp}"] = contrast(f"BA_s{s}", comp)
        C[f"BL_s{s}_vs_B0"] = contrast(f"BL_s{s}", "B0")
    C["BL-arch_vs_B0"] = contrast("BL-arch", "B0")
    rb = {s: Mx[f"BA_s{s}"]["rel_mse"][idx].mean(1) / Mx[f"BL_s{s}"]["rel_mse"][idx].mean(1) for s in (1, 2)}
    agg_pt = 0.5 * sum(C[f"BA_s{s}_vs_BL_s{s}"]["ratio"] for s in (1, 2))
    agg = {"ratio_mean_of_two_new_pairs": agg_pt, "ci_joint_image_bootstrap": ci(0.5 * (rb[1] + rb[2])),
           "both_pairs_below_1": bool(all(C[f"BA_s{s}_vs_BL_s{s}"]["ratio"] < 1 for s in (1, 2)))}
    for comp in ("BL-tm", "BL-arch", "B0"):
        rr = [Mx[f"BA_s{s}"]["rel_mse"][idx].mean(1) / Mx[comp if comp in ("BL-arch", "B0") else f"{comp}_s{s}"]["rel_mse"][idx].mean(1) for s in (1, 2)]
        pt = 0.5 * sum(C[f"BA_s{s}_vs_{comp if comp in ('BL-arch', 'B0') else comp + '_s' + str(s)}"]["ratio"] for s in (1, 2))
        agg[f"secondary_vs_{comp}"] = {"ratio": pt, "ci": ci(0.5 * (rr[0] + rr[1]))}
    up = agg["ci_joint_image_bootstrap"][1]
    if up < 1 and agg["both_pairs_below_1"]:
        reading = "confirmed conditional mean-fidelity improvement (conditional on these trained models)"
    elif up < 1:
        reading = "aggregate improved but mixed training stability (one pair >= 1); not robust superiority"
    else:
        reading = "inconclusive (aggregate CI crosses 1)"
    lp_reg = {k: v["dlpips_ci"][0] > 0 for k, v in C.items() if k.startswith("BA_")}
    # exploratory nested bootstrap over training seeds (all available, incl. pilot if evaluated) and images: fragile at 2-3 seeds
    g2 = np.random.default_rng(1)
    nest = []
    for b in range(2000):
        ss = g2.choice(seeds, len(seeds))
        ii = g2.integers(0, n, n)
        nest.append(np.mean([Mx[f"BA_s{s}"]["rel_mse"][ii].mean() / Mx[f"BL_s{s}"]["rel_mse"][ii].mean() for s in ss]))
    out = {"status": "complete", "n": n, "contrasts": C, "aggregate_primary": agg, "reading_primary": reading, "lpips_regression_flags": lp_reg,
           "per_seed_table": {s: {"BA/BL": C[f"BA_s{s}_vs_BL_s{s}"]["ratio"], "BA/BL-tm": C[f"BA_s{s}_vs_BL-tm_s{s}"]["ratio"], "BA/BL-arch": C[f"BA_s{s}_vs_BL-arch"]["ratio"],
                                  "BA/B0": C[f"BA_s{s}_vs_B0"]["ratio"], "dLPIPS BA-BL": C[f"BA_s{s}_vs_BL_s{s}"]["dlpips"]} for s in seeds},
           "means": {a: {"rel_mse_mean": float(v["rel_mse"].mean()), "rel_mse_median": float(np.median(v["rel_mse"])), "psnr": float(v["psnr"].mean()), "lpips": float(v["lpips"].mean())} for a, v in Mx.items()},
           "nested_seed_image_bootstrap_EXPLORATORY_FRAGILE": {"seeds": seeds, "ci": ci(nest)}}
    jdump("analysis.json", out)
    for a, v in out["means"].items():
        print(f"  {a}: mean {v['rel_mse_mean']:.5f} median {v['rel_mse_median']:.5f} PSNR {v['psnr']:.2f} LPIPS {v['lpips']:.4f}")
    for k, v in C.items():
        print(f"  {k}: ratio {v['ratio']:.3f} [{v['ratio_ci'][0]:.3f}, {v['ratio_ci'][1]:.3f}] wins {v['wins_x']}/{n} dLPIPS {v['dlpips']:+.4f} [{v['dlpips_ci'][0]:+.4f}, {v['dlpips_ci'][1]:+.4f}]")
    print("aggregate", agg, "\nreading:", reading)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
    for s, c in zip(seeds, ("C0", "C1", "C2")):
        ax[0].scatter(Mx[f"BL_s{s}"]["rel_mse"], Mx[f"BA_s{s}"]["rel_mse"], s=7, alpha=0.6, c=c, label=f"seed {s}")
    ax[0].plot([1e-5, 0.3], [1e-5, 0.3], "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("BL-sel endpoint rel. MSE"); ax[0].set_ylabel("BA-sel endpoint rel. MSE"); ax[0].legend(fontsize=8)
    ks = [k for k in C if k.startswith("BA_")]
    ax[1].errorbar([C[k]["ratio"] for k in ks], range(len(ks)), xerr=np.abs(np.array([C[k]["ratio_ci"] for k in ks]).T - [C[k]["ratio"] for k in ks]), fmt="o", ms=3)
    ax[1].set_yticks(range(len(ks))); ax[1].set_yticklabels(ks, fontsize=7); ax[1].axvline(1, c="k", lw=0.8); ax[1].set_xlabel("ratio of mean endpoint rel. MSE")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "eval.png"), dpi=110)


def main():
    os.makedirs(OUT, exist_ok=True)
    st = sys.argv[1]
    if st == "analyze":
        return stage_analyze()
    if st == "plan":
        return stage_plan()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    if st == "repchecks":
        return stage_repchecks(dev)
    if st == "train":
        return stage_train(dev, int(sys.argv[2]))
    if st == "dev":
        return stage_dev(dev, int(sys.argv[2]))
    if st == "eval":
        return stage_eval(dev)


if __name__ == "__main__":
    sys.exit(main() or 0)
