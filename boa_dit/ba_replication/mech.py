"""bar1 step 2: mechanism audit on common states (results/ba_replication/PROTOCOL.md §4). Ledger tag bar1 (common budget scripts/h1031b_budget.py).
Stages: mech (GPU: common-state traces, live replay, standalone vs anchored, conf LPIPS spot-check) | analyze (CPU)
Run: python -m boa_dit.ba_replication.mech mech|analyze"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import h1031b_budget as HB
from boa_dit.activation_bridge import run as AB
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.dit_joint import run as DJ
from boa_dit.model import NULL_CLASS, labels_for_seeds, make_diffusion, noise_for_seeds
from boa_dit.sample import CTX
from boa_dit.weight_anchor.run import guided

TAG = "bar1"
OUT = os.path.join(RESULTS, "ba_replication")
CK = os.path.join(CKPT_DIR, "ba_replication")
DCK = os.path.join(CKPT_DIR, "anchor_refit")
T, KA, LAM = 50, 4, 1.0
MECH = list(range(60200, 60216))
CARRIES = {"B0": "0", "BL@330": os.path.join(DCK, "carry_BL_u330.pt"), "BL-tm": os.path.join(DCK, "carry_BL_tm.pt"),
           "BL-arch": os.path.join(DJ.CK, "carry_C_L_u200.pt"), "BA@330": os.path.join(DCK, "carry_BA_u330.pt")}
STARTS = list(range(0, 45, KA))                                              # replay intervals a -> a+4, a = 0..44
GROUPS = {"early": [0, 4, 8, 12], "middle": [16, 20, 24, 28], "late": [32, 36, 40, 44]}


def jdump(n, o):
    atomic_write_json(os.path.join(OUT, n), o)


def guard(name, est):
    ok, info = HB.ok(TAG, est)
    print(f"[bar1 budget] {name}: {info} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        jdump(f"{name}.json", {"status": "skipped (budget)", **info})
        sys.exit(4)


def key(name):
    return "0" if name == "B0" else f"bar1_{name}"


def register():
    for n, p in CARRIES.items():
        if n != "B0":
            AB.CARRY[key(n)] = p
            AB.ARMS[f"anch:{n}"] = AB.A("static", "fp", 4, True, key(n))
        else:
            AB.ARMS["anch:B0"] = AB.ARMS["CB"]
        AB.ARMS[f"plain:{n}"] = AB.A("static", "fp", None, False, key(n))


@torch.no_grad()
def stage_mech(dev):
    guard("mech", 0.25)
    register()
    with Run(TAG, "mech", dev, {"seeds": [MECH[0], MECH[-1]], "arms": list(CARRIES)}) as run:
        prov = AB.provenance()
        M = AB.load_models(dev, ("fp", "static"))
        d = make_diffusion(T)
        Ac, bc = AB.coeffs(d)
        tmap = torch.as_tensor(d.timestep_map, device=dev, dtype=torch.long)
        n = len(MECH)
        nz, lb = noise_for_seeds(MECH, dev), labels_for_seeds(MECH)
        lab = torch.as_tensor(lb, device=dev, dtype=torch.long)
        y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
        cm, fp = M["static"].m, M["fp"]
        out = {"status": "complete", "seeds": [MECH[0], MECH[-1]], "provenance": prov, "checks": {}}
        # --- common B0 corrected trajectory (pre-anchor-operation states at anchors)
        D = {"x": {}, "e": {}, "anchor": set(), "jump": {}, "fp": fp}
        M["static"].use("0")
        xB0, cB0 = AB.sample(M, AB.ARMS["CB"], d, nz, lb, diag=D)
        X = torch.stack([D["x"][i] for i in range(T)])                          # [50, n, 4, 32, 32]

        def tt(i):
            return tmap[torch.full((n,), T - 1 - i, device=dev, dtype=torch.long)]

        def Q(x, i):
            CTX.step, CTX.n = i, n                                               # static tables are indexed by the sampler step
            r = guided(cm, x, tt(i), y, n)
            CTX.step = None
            return r

        def H(x, i):
            return guided(fp, x, tt(i), y, n)

        def step(x, eps, i):
            ti = torch.full((n,), T - 1 - i, device=dev, dtype=torch.long)
            return d.ddim_sample(lambda xx, ts: eps, x, ti, clip_denoised=False, eta=0.0)["sample"]

        Hs = torch.stack([H(X[i], i)[:, :4] for i in range(T)])
        E = {}
        for name in CARRIES:
            M["static"].use(key(name))
            E[name] = torch.stack([Q(X[i], i)[:, :4] for i in range(T)]) - Hs
        # parity: B0 traces equal the sampler's own diagnostic errors at the same states (non-anchor: same calls; anchor: rc - rh)
        out["checks"]["B0_trace_equals_sampler_diag_maxabs"] = max(float((E["B0"][i].double().cpu() - D["e"][i]).abs().max()) for i in range(T))
        out["checks"]["B0_trajectory_calls"] = cB0
        # static quantizer stateless / step-indexed: same state, same step twice -> bitwise
        M["static"].use("0")
        out["checks"]["static_quantizer_repeat_bitwise"] = bool(torch.equal(Q(X[20], 20), Q(X[20], 20)))
        os.makedirs(CK, exist_ok=True)
        atomic_torch_save({"X": X.cpu(), "H": Hs.cpu(), "E": {k: v.cpu() for k, v in E.items()}, "seeds": MECH}, os.path.join(CK, "mech_traces.pt"))
        # --- live controlled replay: one interval (a -> a+4) and two intervals (a -> a+8)
        rep = {}
        for name in CARRIES:
            M["static"].use(key(name))
            R = {}
            for a in STARTS:
                xa = X[a]
                xF = xa
                FPx = {}
                for i in range(a, min(a + 2 * KA, T)):
                    xF = step(xF, H(xF, i), i)
                    FPx[i + 1] = xF
                x = xa
                rc, rh = Q(x, a), H(x, a)
                dprev = rc[:, :4] - rh[:, :4]
                x = step(x, rh, a)
                calls = {"cheap": 1, "fp": 1}
                r = {}
                for span in (1, 2):
                    c0 = a + (span - 1) * KA
                    if span == 2:
                        if a + 2 * KA > 48:
                            break
                        x = step(xp, hh, c0)                                     # the DDIM step at a+4 uses the pre-correction FP prediction
                    for i in range(c0 + 1, c0 + KA):
                        res = Q(x, i).clone()
                        res[:, :4] -= dprev
                        x = step(x, res, i)
                        calls["cheap"] += 1
                    c1 = c0 + KA
                    xm = x
                    qq, hh = Q(xm, c1), H(xm, c1)
                    calls["cheap"] += 1; calls["fp"] += 1
                    dn = qq[:, :4] - hh[:, :4]
                    coef = AB.bridge_coef(Ac, bc, c0, KA)
                    kick = -LAM * coef * (dn - dprev)
                    xp = xm + kick
                    ref = FPx[c1]
                    dev_m, dev_p = (xm - ref).double(), (xp - ref).double()
                    tgt = (ref - xm).double()
                    kd = kick.double()
                    r[span] = {"pre": dev_m.flatten(1).pow(2).sum(1).tolist(), "post": dev_p.flatten(1).pow(2).sum(1).tolist(),
                               "ref_norm2": ref.double().flatten(1).pow(2).sum(1).tolist(),
                               "kick_dot_target": (kd * tgt).flatten(1).sum(1).tolist(), "kick_norm2": kd.flatten(1).pow(2).sum(1).tolist(),
                               "target_norm2": tgt.flatten(1).pow(2).sum(1).tolist(), "calls": dict(calls),
                               "scalar_replay_kick_maxabs": float((kick - (-LAM * coef * (dn - dprev))).abs().max()), "coef": coef}
                    x, dprev = xp, dn
                R[a] = r
            rep[name] = R
        out["replay"] = rep
        # --- standalone (plain 50 cheap calls) vs anchored, batch 16
        lat = {"FP50": AB.sample(M, AB.ARMS["FP50"], d, nz, lb)[0].cpu()}
        cnt = {}
        for name in CARRIES:
            for mode in ("plain", "anch"):
                k_ = f"{mode}:{name}"
                x, c = AB.sample(M, AB.ARMS[k_], d, nz, lb)
                assert torch.isfinite(x).all() and c == AB.expect(AB.ARMS[k_]), (k_, c)
                lat[k_], cnt[k_] = x.cpu(), c
        out["checks"]["anch_B0_equals_trajectory_endpoint_bitwise"] = bool(torch.equal(lat["anch:B0"], xB0.cpu()))
        out["standalone_calls"] = cnt
        atomic_torch_save(lat, os.path.join(CK, "mech_standalone_latents.pt"))
        del M
        torch.cuda.empty_cache()
        Mx, _ = AB.evaluate(lat, dev, [])
        out["standalone"] = Mx
        # --- export spot-check: re-decode 8 dar1 confirmation images per arm and compare PSNR / LPIPS with the stored values
        L = torch.load(os.path.join(DCK, "conf_latents.pt"), map_location="cpu")
        conf = json.load(open(os.path.join(RESULTS, "anchor_refit", "conf.json")))
        sub = {k: v[:8] for k, v in L.items()}
        Ms, _ = AB.evaluate(sub, dev, [])
        out["checks"]["conf_spotcheck_max_abs_diff"] = {a: {m: float(np.max(np.abs(np.asarray(Ms[a][m]) - np.asarray(conf["arms"][a][m][:8])))) for m in ("psnr", "lpips", "rel_mse")} for a in Ms}
        out.update(run.record())
    jdump("mech.json", out)
    print(json.dumps(out["checks"], indent=1), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------- analysis (CPU)
def boot_ratio(a, b, nb=10000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    g = np.random.default_rng(0)
    idx = g.integers(0, len(a), (nb, len(a)))
    r = a[idx].mean(1) / b[idx].mean(1)
    return {"ratio": float(a.mean() / b.mean()), "ci": [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]}


def stage_analyze(dev=None):
    tr = torch.load(os.path.join(CK, "mech_traces.pt"), map_location="cpu")
    mj = json.load(open(os.path.join(OUT, "mech.json")))
    d = make_diffusion(T)
    Ac, bc = AB.coeffs(d)
    Hs = tr["H"].double()
    hn = Hs.pow(2).flatten(2).sum(2).sum(0)                                       # per image, common normaliser
    res = {"normaliser": "sum_t ||H_t||^2 per image (FP prediction at the common states; identical across arms)", "arms": {}}
    per_img = {}
    for name, E in tr["E"].items():
        E = E.double()                                                          # [50, n, 4, 32, 32]
        en = lambda v: v.pow(2).flatten(-3).sum(-1)                              # energy per image
        raw_t = en(E)                                                           # [50, n]
        raw = raw_t.sum(0)
        mean_e, dev_e, held, Hw, vv, rr, Hv = 0, 0, 0, 0, 0, 0, 0
        for a in STARTS:
            blk = E[a:a + KA]
            m = blk.mean(0)
            mean_e = mean_e + KA * en(m)
            dev_e = dev_e + en(blk - m).sum(0)
            r_ = E[a + 1:a + KA] - E[a]
            held = held + en(r_).sum(0)
            w = AB.weights(Ac, bc, a, KA)
            HI = sum(wj * (E[a + j] - E[a]) for j, wj in zip(range(1, KA), w))
            vI = AB.bridge_coef(Ac, bc, a, KA) * (E[a + KA] - E[a])
            Hw, vv, rr, Hv = Hw + en(HI), vv + en(vI), rr + en(HI - vI), Hv + (HI * vI).flatten(1).sum(1)
        gm = E.mean(0)
        lag = {}
        for Lg in (1, 2, 4, 8):
            a_, b_ = E[:-Lg], E[Lg:]
            cross = (a_ * b_).flatten(2).sum(2).sum(0)
            lag[Lg] = {"cross": float(cross.sum()), "corr": float(cross.sum() / math.sqrt(float(en(a_).sum()) * float(en(b_).sum()))),
                       "increment_energy": float(en(b_ - a_).sum()),
                       "corr_after_removing_per_image_time_mean": float(((a_ - gm) * (b_ - gm)).flatten(2).sum(2).sum() / math.sqrt(float(en(a_ - gm).sum()) * float(en(b_ - gm).sum())))}
        per_img[name] = {"raw_norm": (raw / hn).numpy(), "held": held.numpy(), "Hw": Hw.numpy(), "resid": rr.numpy()}
        res["arms"][name] = {"raw_energy": float(raw.sum()), "raw_energy_normalised": float((raw / hn).mean()),
                             "raw_per_step": raw_t.sum(1).tolist(),
                             "interval_mean_energy": float(mean_e.sum()), "interval_deviation_energy": float(dev_e.sum()),
                             "global_time_mean_energy_share": float((T * en(gm)).sum() / raw.sum()),
                             "held_residual_energy": float(held.sum()), "weighted_held_H_energy_APPROX": float(Hw.sum()), "bridge_v_energy_APPROX": float(vv.sum()),
                             "bridge_residual_H_minus_v_APPROX": float(rr.sum()), "cos_H_v_APPROX": float(Hv.sum() / math.sqrt(float(Hw.sum()) * float(vv.sum()))),
                             "lags": lag}
    for ref in ("B0", "BL@330"):
        res[f"ratios_vs_{ref}"] = {n: {k: res["arms"][n][k] / res["arms"][ref][k] for k in ("raw_energy", "held_residual_energy", "weighted_held_H_energy_APPROX",
                                                                                         "bridge_residual_H_minus_v_APPROX", "interval_mean_energy", "interval_deviation_energy")} for n in res["arms"]}
    # replay
    rp = {}
    for name, R in mj["replay"].items():
        rp[name] = {}
        for span in ("1", "2"):
            starts = [a for a in STARTS if span in R[str(a)]]
            pre = np.array([R[str(a)][span]["pre"] for a in starts])               # [n_a, n]
            post = np.array([R[str(a)][span]["post"] for a in starts])
            refn = np.array([R[str(a)][span]["ref_norm2"] for a in starts])
            kdt = np.array([R[str(a)][span]["kick_dot_target"] for a in starts])
            kn = np.array([R[str(a)][span]["kick_norm2"] for a in starts])
            tn = np.array([R[str(a)][span]["target_norm2"] for a in starts])
            g = {}
            for gname, ga in list(GROUPS.items()) + [("all", starts)]:
                sel = [starts.index(a) for a in ga if a in starts]
                if not sel:
                    continue
                g[gname] = {"pre_rel": float((pre[sel] / refn[sel]).mean()), "post_rel": float((post[sel] / refn[sel]).mean()),
                            "post_over_pre": float(post[sel].sum() / pre[sel].sum()),
                            "cos_kick_target": float(kdt[sel].sum() / np.sqrt(kn[sel].sum() * tn[sel].sum())),
                            "kick_over_target_norm": float(np.sqrt(kn[sel].sum() / tn[sel].sum())),
                            "per_image_post_rel": (post[sel] / refn[sel]).mean(0).tolist(), "per_image_pre_rel": (pre[sel] / refn[sel]).mean(0).tolist()}
            g["per_start"] = {a: {"pre_rel": float((pre[i] / refn[i]).mean()), "post_rel": float((post[i] / refn[i]).mean()),
                                  "cos": float(kdt[i].sum() / np.sqrt(kn[i].sum() * tn[i].sum()))} for i, a in enumerate(starts)}
            rp[name][span] = g
    res["replay"] = rp
    cmp = {}
    for ref in ("B0", "BL@330", "BL-tm", "BL-arch"):
        for span in ("1", "2"):
            for gname in ("early", "middle", "late", "all"):
                if gname in rp["BA@330"][span] and gname in rp[ref][span]:
                    cmp[f"BA@330/{ref} span{span} {gname} post"] = boot_ratio(rp["BA@330"][span][gname]["per_image_post_rel"], rp[ref][span][gname]["per_image_post_rel"])
                    cmp[f"BA@330/{ref} span{span} {gname} pre"] = boot_ratio(rp["BA@330"][span][gname]["per_image_pre_rel"], rp[ref][span][gname]["per_image_pre_rel"])
    res["replay_ratios_image_bootstrap_DESCRIPTIVE"] = cmp
    sa = mj["standalone"]
    res["standalone_vs_anchored"] = {k: {"rel_mse_mean": float(np.mean(v["rel_mse"])), "rel_mse_median": float(np.median(v["rel_mse"])),
                                         "psnr": float(np.mean(v["psnr"])), "lpips": float(np.mean(v["lpips"]))} for k, v in sa.items()}
    res["status"] = "complete"
    jdump("mech_analysis.json", res)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for n in res["arms"]:
        ax[0].plot(res["arms"][n]["raw_per_step"], label=n)
        ps = rp[n]["1"]["per_start"]
        ax[1].plot(STARTS, [ps[a]["post_rel"] for a in STARTS], "-o", ms=3, label=f"{n} post")
        ax[2].plot(STARTS, [ps[a]["pre_rel"] for a in STARTS], "-o", ms=3, label=f"{n} pre")
    ax[0].set_yscale("log"); ax[0].set_xlabel("step"); ax[0].set_title("raw ||e_t||^2 on common B0 states (16 images)"); ax[0].legend(fontsize=7)
    for a_, t_ in ((ax[1], "replay a->a+4: post-kick rel. deviation vs FP"), (ax[2], "replay a->a+4: pre-kick rel. deviation vs FP")):
        a_.set_yscale("log"); a_.set_xlabel("interval start a"); a_.set_title(t_, fontsize=9); a_.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "mech.png"), dpi=110)
    for n, v in res["arms"].items():
        print(f"{n}: raw {v['raw_energy']:.4g} held {v['held_residual_energy']:.4g} Hw {v['weighted_held_H_energy_APPROX']:.4g} resid {v['bridge_residual_H_minus_v_APPROX']:.4g} "
              f"cos {v['cos_H_v_APPROX']:.3f} mean/dev {v['interval_mean_energy']:.4g}/{v['interval_deviation_energy']:.4g} gmean share {v['global_time_mean_energy_share']:.3f}")
    for n in rp:
        for s in ("1", "2"):
            a = rp[n][s]["all"]
            print(f"replay {n} span{s}: pre {a['pre_rel']:.3e} post {a['post_rel']:.3e} post/pre {a['post_over_pre']:.3f} cos {a['cos_kick_target']:.3f} |k|/|t| {a['kick_over_target_norm']:.3f}")
    for k, v in cmp.items():
        if "all" in k:
            print(k, round(v["ratio"], 3), np.round(v["ci"], 3))
    for k, v in res["standalone_vs_anchored"].items():
        print(k, v)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    st = sys.argv[1]
    if st == "mech":
        torch.cuda.set_device("cuda:0")
        sys.exit(stage_mech("cuda:0") or 0)
    stage_analyze()
