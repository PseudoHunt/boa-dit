"""Append/refresh the results section of results/gateA.md from JSONs. The pre-registration above the marker is never
modified."""
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from boa_dit.common import atomic_write_text  # noqa: E402

MARK = "\n---\n\n## Results (appended after the runs by `scripts/make_gateA.py`)\n"
RES = os.path.join(ROOT, "results")
ORDER = ["static-16", "static-32", "time2-16", "time2-32", "time3-16", "static-48"]
PAIRS = [("time2-16", "static-32"), ("time3-16", "static-48"), ("time2-32", "static-32"), ("static-32", "static-16")]
PIVOT_START = "2026-10-25T08:45"


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def arm(a, b):
    return load(os.path.join(RES, "stageA", "eval", f"latent_{a}_w{b}.json"))


def uniform(b):
    return load(os.path.join(RES, "stage2", "eval", f"latent_uniform_w{b}_K1_lam1_brcfg.json"))


def paired(x, y):
    x, y = np.array(x["rel_mse_per_image"]), np.array(y["rel_mse_per_image"])
    rng = np.random.default_rng(0)
    bs = [1 - x[i].mean() / y[i].mean() for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 1 - x.mean() / y.mean(), lo, hi, float((x - y < 0).mean())


def fmt(r):
    red, lo, hi, imp = r
    return f"**{100 * red:+.1f}%** [{100 * lo:+.1f}, {100 * hi:+.1f}]", f"{100 * imp:.0f}%"


def spent():
    out = {}
    for row in csv.DictReader(open(os.path.join(RES, "gpu_hours.csv"))):
        if row["start_utc"] >= PIVOT_START and (row["stage"] in ("gateA", "item2") or row["run"].startswith("fid2000")):
            k = "item 1 (FID anchors)" if row["run"].startswith("fid2000") else \
                ("item 2" if row["stage"] == "item2" else "item 3 (Gate A)")
            out[k] = out.get(k, 0.0) + float(row["gpu_hours"])
    return out


def main():
    path = os.path.join(RES, "gateA.md")
    pre = open(path).read().split(MARK)[0]
    L = [MARK]
    verdict = {}
    for b in (4, 3):
        u = uniform(b)
        L.append(f"### W{b} (base `uniform_w{b}_K1_lam1_brcfg`, uncorrected rel. latent MSE {u['rel_mse_mean']:.5f})\n")
        L.append("| arm | extra bits/param | rel. latent MSE | vs uniform (95% CI) | images improved |")
        L.append("| --- | ---: | ---: | --- | ---: |")
        for a in ORDER:
            r = arm(a, b)
            if r is None:
                L.append(f"| {a} | — | not run (cap) or pending | — | — |")
                continue
            c, imp = fmt(paired(r, u))
            L.append(f"| {a} | {r['extra_bits_per_param']:.3f} | {r['rel_mse_mean']:.5f} ± {r['rel_mse_sem']:.5f} | {c} | {imp} |")
        L.append("\nEqual-budget and rank comparisons (reduction of the first arm vs the second):\n")
        L.append("| comparison | reduction (95% CI) | images improved |")
        L.append("| --- | --- | ---: |")
        for x, y in PAIRS:
            rx, ry = arm(x, b), arm(y, b)
            if rx is None or ry is None:
                L.append(f"| {x} vs {y} | pending / not run | — |")
                continue
            c, imp = fmt(paired(rx, ry))
            L.append(f"| {x} vs {y} | {c} | {imp} |")
        L.append("")
        t, s = arm("time2-16", b), arm("static-32", b)
        if t is not None and s is not None:
            r_ts, r_tu = paired(t, s), paired(t, u)
            need = 0.05 if b == 4 else 0.10
            ok1 = r_ts[0] >= need and r_ts[1] > 0
            ok2 = r_tu[0] >= 0.15
            verdict[b] = (ok1 and ok2, r_ts, r_tu, need, ok1, ok2)
    L.append("### Captured weighted residual energy (mean over the 112 linears; fraction of ‖E H½‖²_F)\n")
    L.append("| bits | regime | rank 16 | rank 32 | rank 48 |")
    L.append("| --- | --- | ---: | ---: | ---: |")
    for b in (4, 3):
        ce = load(os.path.join(RES, "stageA", f"captured_energy_w{b}.json"))
        if ce is None:
            continue
        ce = ce["per_layer_regime"]
        for rg in sorted({".".join(k.split(".")[-2:]) for k in ce}):
            v = np.array([x for k, x in ce.items() if k.endswith("." + rg)]).mean(0)
            L.append(f"| W{b} | {rg} | {v[0]:.3f} | {v[1]:.3f} | {v[2]:.3f} |")
    L.append("")
    ra = load(os.path.join(RES, "stage2", "row_attribution.json"))
    if ra:
        L.append("### Item 2a — attribution (uniform W4; one block / layer swapped back to FP; seeds 0–63)\n")
        L.append(f"Base rel. latent MSE on these 64 seeds {ra['base_rel_mse']:.4f}. Share = (base − swapped) / base; the "
                 f"28 block shares sum to {sum(v['contribution_share'] for v in ra['blocks'].values()):.3f} "
                 "(errors do not add exactly).\n")
        top = sorted(ra["blocks"], key=lambda b: -ra["blocks"][b]["contribution_share"])[:8]
        L.append("| block | " + " | ".join(top) + " |")
        L.append("| --- |" + " ---: |" * len(top))
        L.append("| share | " + " | ".join(f"{ra['blocks'][b]['contribution_share']:+.3f}" for b in top) + " |\n")
        L.append("| layer (top-3 blocks) | share |")
        L.append("| --- | ---: |")
        for k, v in ra["layers"].items():
            L.append(f"| {k} | {v['contribution_share']:+.3f} |")
        L.append("")
    gr = load(os.path.join(RES, "stage2", "guard_rows.json"))
    if gr:
        L.append("### Item 2b — guard arms (top-100 rows kept at W8 RTN on top of `uniform`; seeds 0–511)\n")
        L.append(f"Row sets: `gate` = top-100 by RTN-W4 row output error × mean gate energy; `err` = top-100 by the "
                 f"error alone; overlap {gr['overlap_gate_vs_err']} rows.\n")
        L.append("| set | bits | extra bits/param | rel. latent MSE | vs uniform (95% CI) | images improved |")
        L.append("| --- | --- | ---: | ---: | --- | ---: |")
        for b in (4, 3):
            for s in ("gate", "err"):
                r = load(os.path.join(RES, "stage2", "eval", f"latent_guard_{s}_w{b}.json"))
                if r is None:
                    L.append(f"| {s} | W{b} | — | pending | — | — |")
                    continue
                c, imp = fmt(paired(r, uniform(b)))
                L.append(f"| {s} | W{b} | {r['extra_bits_per_param']:.4f} | {r['rel_mse_mean']:.5f} | {c} | {imp} |")
        L.append("")
    fa = load(os.path.join(RES, "stage2", "fid_anchors.json"))
    if fa:
        L.append("### FID-2k anchors (item 1, `results/stage2/fid_anchors.json`)\n")
        L.append("| model | FID-2k | rel. latent MSE (512) |")
        L.append("| --- | ---: | ---: |")
        for k, v in fa["anchors"].items():
            L.append(f"| {k} | {v['fid2k']:.2f} | {v['rel_latent_mse_512']:.4f} |")
        L.append("")
    sp = spent()
    L.append("### GPU-hours (items 1–3, since the pivot started)\n")
    for k in sorted(sp):
        L.append(f"- {k}: {sp[k]:.2f}")
    L.append(f"- **total: {sum(sp.values()):.2f}** (cap 3.5)\n")
    L.append("### Decision (pre-registered rule)\n")
    if not verdict:
        L.append("Pending.")
    else:
        for b, (ok, r_ts, r_tu, need, ok1, ok2) in verdict.items():
            L.append(f"- W{b}: time2-16 vs static-32 {100 * r_ts[0]:+.1f}% [{100 * r_ts[1]:+.1f}, {100 * r_ts[2]:+.1f}] "
                     f"(needs ≥ {100 * need:.0f}%, CI > 0): **{ok1}**; time2-16 vs uniform {100 * r_tu[0]:+.1f}% "
                     f"(needs ≥ 15%): **{ok2}** → {'GO' if ok else 'fail'}")
        go = any(v[0] for v in verdict.values())
        done = len(verdict) == 2
        L.append(f"\n**Gate A: {'GO' if go else ('STOP' if done else 'PENDING')}**\n")
    atomic_write_text(path, pre + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
