"""bar1 step 1A/1B (CPU): audit of the dar1 confirmation outputs and frozen-N power planning (results/ba_replication/PROTOCOL.md §2-3).
Run: python -m boa_dit.ba_replication.audit audit|power"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, atomic_write_json

OUT = os.path.join(RESULTS, "ba_replication")
DAR = os.path.join(RESULTS, "anchor_refit")
DCK = os.path.join(CKPT_DIR, "anchor_refit")
ARMS = ["B0", "BA@330", "BL@330", "BL-tm", "BL-arch"]
PAIRS = [("BA@330", "BL@330"), ("BA@330", "B0"), ("BA@330", "BL-tm"), ("BA@330", "BL-arch"), ("BL@330", "B0"), ("BL-tm", "B0"), ("BL-arch", "B0")]
NBOOT = 10000


def boot_idx(n, nb=NBOOT, seed=0):
    g = np.random.default_rng(seed)
    return np.stack([g.integers(0, n, n) for _ in range(nb)])


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def top_share(v, ks=(1, 5, 10)):
    v = np.asarray(v, float)
    tot = v.sum()
    s = np.sort(v)[::-1]
    return {k: float(s[:k].sum() / tot) for k in ks}


def audit():
    conf = json.load(open(os.path.join(DAR, "conf.json")))
    art = json.load(open(os.path.join(DAR, "artifacts.json")))
    L = torch.load(os.path.join(DCK, "conf_latents.pt"), map_location="cpu")
    rep = {"inputs": {"conf_latents.pt": sha(os.path.join(DCK, "conf_latents.pt")), "conf.json_seeds": conf["seeds"], "n": conf["n"]}, "verify": {}}
    V = rep["verify"]
    V["arm_keys"] = sorted(L)
    V["shapes"] = {k: list(v.shape) for k, v in L.items()}
    V["finite"] = {k: bool(torch.isfinite(v).all()) for k, v in L.items()}
    ref = L["FP50"].double()
    rec = {a: ((L[a].double() - ref).flatten(1).pow(2).sum(1) / ref.flatten(1).pow(2).sum(1)).numpy() for a in ARMS}
    V["rel_mse_recomputed_max_abs_rel_diff"] = {a: float(np.max(np.abs(rec[a] - np.asarray(conf["arms"][a]["rel_mse"])) / np.asarray(conf["arms"][a]["rel_mse"]))) for a in ARMS}
    V["rel_mse_reproduced"] = all(v < 1e-5 for v in V["rel_mse_recomputed_max_abs_rel_diff"].values())
    V["calls"] = conf["calls"]
    V["calls_ok"] = all(conf["calls"][a] == {"cheap": 50, "high": 13, "corrections": 12} for a in ARMS) and conf["calls"]["FP50"] == {"cheap": 0, "high": 50, "corrections": 0}
    V["arms_distinct_from_each_other"] = {f"{a}|{b}": not bool(torch.equal(L[a], L[b])) for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
    ck = {"BA@330": "carry_BA_u330.pt", "BL@330": "carry_BL_u330.pt", "BL-tm": "carry_BL_tm.pt"}
    V["checkpoint_sha_matches_manifest"] = {a: sha(os.path.join(DCK, f)) == art["files"][f]["sha256"] for a, f in ck.items()}
    V["seeds_order"] = "124000..124127 in batch order (stage_conf gen over CONF = range(124000, 124128), bs 32)"
    V["note_precision_cfg"] = "fp16 autocast, fp32 sampler, CFG 1.5 (abn1 sample; dar1 checks 2 reproduce abn1 CB / LCB bitwise)"
    diag = json.load(open(os.path.join(DAR, "diag.json")))
    tf = diag["teacher_forced"]
    rep["five_point_five_x"] = {"definition": "diag.json teacher_forced raw = sum over 50 steps and 8 images (60208-60215) of ||Q(x_t)-F(x_t)||^2 (squared L2, 4-channel guided prediction), on FP50 teacher-forced states (common across arms)",
                                "raw": {k: v["raw"] for k, v in tf.items()}, "ratio_BA_over_B0_squared": tf["BA-sel"]["raw"] / tf["B0"]["raw"],
                                "ratio_relative_L2_sqrt": float(np.sqrt(tf["BA-sel"]["raw"] / tf["B0"]["raw"])),
                                "live_raw": {k: v["raw"] for k, v in diag["live"].items()}}
    M = {a: {m: np.asarray(conf["arms"][a][m], float) for m in ("rel_mse", "psnr", "lpips")} for a in ARMS}
    n = len(M["B0"]["rel_mse"])
    idx = boot_idx(n)
    rep["arms"] = {a: {"mean": float(M[a]["rel_mse"].mean()), "median": float(np.median(M[a]["rel_mse"])), "psnr_mean": float(M[a]["psnr"].mean()),
                       "lpips_mean": float(M[a]["lpips"].mean()), "top_share_of_total_error": top_share(M[a]["rel_mse"]),
                       "quantiles": {q: float(np.quantile(M[a]["rel_mse"], q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)}} for a in ARMS}
    C = {}
    for x, y in PAIRS:
        a, b = M[x]["rel_mse"], M[y]["rel_mse"]
        dd = a - b
        r = {"ratio": float(a.mean() / b.mean()), "ratio_ci": ci((a[idx].mean(1) / b[idx].mean(1))), "mean_diff": float(dd.mean()), "mean_diff_ci": ci(dd[idx].mean(1)),
             "median_diff": float(np.median(dd)), "wins_x": int((a < b).sum()), "n": n,
             "dlpips": float((M[x]["lpips"] - M[y]["lpips"]).mean()), "dlpips_ci": ci((M[x]["lpips"] - M[y]["lpips"])[idx].mean(1)),
             "dpsnr": float((M[x]["psnr"] - M[y]["psnr"]).mean()), "dpsnr_ci": ci((M[x]["psnr"] - M[y]["psnr"])[idx].mean(1))}
        # share of total paired difference from the largest |differences| (signed contribution of those images / total signed difference)
        o = np.argsort(-np.abs(dd))
        r["largest_abs_diff_images_share_of_total_diff"] = {k: float(dd[o[:k]].sum() / dd.sum()) for k in (1, 5, 10)}
        r["largest_abs_diff_image_ids"] = [int(124000 + i) for i in o[:10]]
        loo = np.array([np.delete(a, i).mean() / np.delete(b, i).mean() for i in range(n)])
        r["leave_one_out_ratio_range_DIAGNOSTIC"] = [float(loo.min()), float(loo.max())]
        r["leave_one_out_most_influential"] = {"image": int(124000 + np.argmax(np.abs(loo - r["ratio"]))), "ratio_without": float(loo[np.argmax(np.abs(loo - r["ratio"]))])}
        C[f"{x}_vs_{y}"] = r
    rep["contrasts"] = C
    stored = conf["contrasts"]
    rep["verify"]["primary_ratio_ci_matches_stored"] = {k: [C[k]["ratio"], C[k]["ratio_ci"], stored[k]["rel_mse"]["ratio"], stored[k]["rel_mse"]["ratio_ci"]] for k in C if k in stored}
    # plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    for comp, c in (("BL@330", "C0"), ("B0", "C1"), ("BL-arch", "C2")):
        ax[0].scatter(M[comp]["rel_mse"], M["BA@330"]["rel_mse"], s=9, alpha=0.7, c=c, label=f"y=BA@330 vs x={comp}")
    lo, hi = 1e-5, 0.2
    ax[0].plot([lo, hi], [lo, hi], "k--", lw=0.8); ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("comparator endpoint rel. MSE"); ax[0].set_ylabel("BA@330 endpoint rel. MSE"); ax[0].legend(fontsize=7); ax[0].set_title("per-image paired (124000-124127)")
    for a in ARMS:
        s = np.sort(M[a]["rel_mse"])
        ax[1].plot(s, np.arange(1, n + 1) / n, label=a)
    ax[1].set_xscale("log"); ax[1].set_xlabel("endpoint rel. MSE"); ax[1].set_ylabel("ECDF"); ax[1].legend(fontsize=7)
    dd = M["BA@330"]["rel_mse"] - M["BL@330"]["rel_mse"]
    ax[2].hist(dd, bins=60); ax[2].axvline(0, c="k", lw=0.8); ax[2].set_yscale("log"); ax[2].set_xlabel("BA@330 - BL@330 (per image)"); ax[2].set_title("paired differences")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "audit_1a.png"), dpi=110)
    json.dump({a: M[a]["rel_mse"].tolist() for a in ARMS}, open(os.path.join(OUT, "audit_1a_per_image_rel_mse.json"), "w"))
    rep["status"] = "complete"
    atomic_write_json(os.path.join(OUT, "audit_1a.json"), rep)
    print(json.dumps({"verify": {k: v for k, v in V.items() if k not in ("shapes",)}, "5.5x": rep["five_point_five_x"]}, indent=1, default=str)[:4000])
    for k, r in C.items():
        print(f"{k}: ratio {r['ratio']:.3f} {np.round(r['ratio_ci'], 3)} mdiff {r['mean_diff']:.2e} {np.round(r['mean_diff_ci'], 6)} med {r['median_diff']:.2e} wins {r['wins_x']} "
              f"dLPIPS {r['dlpips']:+.4f} dPSNR {r['dpsnr']:+.2f} top1/5/10 {[round(v, 2) for v in r['largest_abs_diff_images_share_of_total_diff'].values()]} LOO {np.round(r['leave_one_out_ratio_range_DIAGNOSTIC'], 3)}")
    for a in ARMS:
        print(a, {k: round(v, 3) for k, v in rep["arms"][a]["top_share_of_total_error"].items()}, f"mean {rep['arms'][a]['mean']:.5f}")



def power():
    """§3: frozen-N planning from old paired data (124000-124127) under a multiplicative shift to a true ratio of 0.90."""
    p = json.load(open(os.path.join(OUT, "audit_1a_per_image_rel_mse.json")))
    ba, bl, tm = (np.asarray(p[k]) for k in ("BA@330", "BL@330", "BL-tm"))
    alt = 0.90
    s1 = alt / (ba.mean() / bl.mean())
    s2 = alt / (ba.mean() / tm.mean())
    A1, A2 = ba * s1, ba * s2
    g = np.random.default_rng(1)
    out = {"alternative_ratio": alt, "shift_factor_vs_BL@330": float(s1), "n_sim": 2000, "n_boot": 1000, "per_N": {}}
    for N in (128, 256, 512):
        hw, up1, hwa, upa, upa_both = [], [], [], [], []
        for _ in range(2000):
            j = g.integers(0, 128, N)
            a1, b1, a2, b2 = A1[j], bl[j], A2[j], tm[j]
            bi = g.integers(0, N, (1000, N))
            r1 = a1[bi].mean(1) / b1[bi].mean(1)
            r2 = a2[bi].mean(1) / b2[bi].mean(1)
            ra = 0.5 * (r1 + r2)
            c1, ca = np.percentile(r1, [2.5, 97.5]), np.percentile(ra, [2.5, 97.5])
            hw.append((c1[1] - c1[0]) / 2); up1.append(c1[1] < 1)
            hwa.append((ca[1] - ca[0]) / 2); upa.append(ca[1] < 1)
            upa_both.append(ca[1] < 1 and a1.mean() / b1.mean() < 1 and a2.mean() / b2.mean() < 1)
        # delta-method SE of a ratio of means at N, from the observed (shifted) 128 pairs
        R = A1.mean() / bl.mean()
        psi = (A1 - R * bl) / bl.mean()
        out["per_N"][N] = {"single_pair_median_ci_halfwidth": float(np.median(hw)), "single_pair_power_upper_lt_1": float(np.mean(up1)),
                           "aggregate_median_ci_halfwidth_APPROX": float(np.median(hwa)), "aggregate_power_upper_lt_1_APPROX": float(np.mean(upa)),
                           "aggregate_power_confirm_rule_APPROX": float(np.mean(upa_both)), "delta_method_se_single": float(psi.std(ddof=1) / np.sqrt(N))}
        print(N, out["per_N"][N], flush=True)
    rate_arm, rate_fp = (0.3594 - 0.022) / 5 / 128, 0.022 / 128       # A30-h per image (dar1 conf ledger, incl. decode + LPIPS)
    proj = {"spent": 0.0, "mechanism": 0.15, "two_pairs_train_dev": 2 * (0.3885 + 0.3559 + 0.12), "kernel_track": 0.3, "reserve": 0.4}
    out["cost"] = {"rate_anchored_arm_per_image_h": rate_arm, "rate_fp50_per_image_h": rate_fp, "projected_fixed": proj, "per_N_total": {}}
    for N in (512, 256, 128):
        ev = N * (8 * rate_arm + rate_fp)
        out["cost"]["per_N_total"][N] = {"eval_h": ev, "total_h": sum(proj.values()) + ev, "fits": sum(proj.values()) + ev <= 4.0}
    fit = [N for N in (512, 256, 128) if out["cost"]["per_N_total"][N]["fits"]]
    out["frozen_N"] = fit[0] if fit else None
    if out["frozen_N"]:
        pw = out["per_N"][out["frozen_N"]]["aggregate_power_confirm_rule_APPROX"]
        out["label"] = "decisive-capable (power >= 0.8)" if pw >= 0.8 else "bounded replication, not a decisive test (projected power < 0.8 for a 10% reduction)"
    out["limits"] = ("resampling from 128 observed pairs cannot produce tails heavier than observed; one image (124092) carries 28% of BA@330's total error; "
                     "aggregate uses BL@330 and BL-tm as two comparator realizations sharing the same BA outputs (over-correlated proxy; approximate)")
    out["status"] = "complete"
    atomic_write_json(os.path.join(OUT, "power_1b.json"), out)
    print(json.dumps({k: out[k] for k in ("cost", "frozen_N", "label")}, indent=1))


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    {"audit": audit, "power": power}[sys.argv[1]]()
