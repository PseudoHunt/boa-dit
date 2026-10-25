"""Deflate G2: table, verdict, findings and diagnostics below the marker in results/deflate/G2.md (from JSONs)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from boa_dit.common import atomic_savefig, atomic_write_json, atomic_write_text  # noqa: E402
from make_gateA import paired  # noqa: E402

R = os.path.join(ROOT, "results")
OUT = os.path.join(R, "deflate")
MARK = "\n---\n\n## Results (appended by `scripts/make_deflate_g2.py`)\n"
ARMS = [("DC-r*", "dc", "star"), ("DC-1", "dc", "1"), ("C-r*", "c", "star"), ("HEIG-8", "heig", "8"), ("WSVD-r*", "wsvd", "star")]
TYPES = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]


def load(p):
    return json.load(open(p)) if os.path.exists(p) else None


def arm(mode, r, b, damp=""):
    n = f"deflate_{mode}_r{r}_w{b}{damp}"
    return n, load(os.path.join(OUT, "quant", f"{n}.json")), load(os.path.join(OUT, "eval", f"latent_{n}.json")), \
        load(os.path.join(OUT, "eval", f"fid2000_{n}.json"))


def main():
    fa = load(os.path.join(R, "stage2", "fid_anchors.json"))["anchors"]
    fp = fa["FP"]["fid2k"]
    uni = {b: load(os.path.join(R, "stage2", "eval", f"latent_uniform_w{b}_K1_lam1_brcfg.json")) for b in (4, 3)}
    ufid = {4: fa["uniform W4"]["fid2k"], 3: fa["uniform W3"]["fid2k"]}
    s32 = {b: load(os.path.join(R, "stageA", "eval", f"latent_static-32_w{b}.json")) for b in (4, 3)}
    s8 = {b: load(os.path.join(R, "stageA", "eval", f"latent_static-8_w{b}.json")) for b in (4, 3)}
    s8f = {b: load(os.path.join(R, "stageA", "eval", f"fid2000_static-8_w{b}.json")) for b in (4, 3)}
    L = [MARK, f"Reused: uniform W4 / W3 (FID-2k {ufid[4]:.2f} / {ufid[3]:.2f}; FP {fp:.2f}). static-8 = Gate A's static "
         "low-rank residual correction at rank 8 (same code path as static-32; rebuilt Grams), the matched-rank baseline. "
         "Effective bits per weight = weight bits + extra bits (fp16 carry or correction factors) per quantized weight; "
         "per-channel scale/zero are common to every arm and not counted. Reductions are 1 − relMSE(arm) / relMSE(uniform), "
         "with paired bootstrap 95% CIs over the 512 images (positive = better). WSVD-r* was dropped for budget (reviewer "
         "priority order).\n",
         "| arm | bits | extra bits/weight | effective bits/weight | rel latent MSE | latent MSE @ eff. bits | vs uniform | FID-2k | Δ vs FP |",
         "| --- | --- | ---: | ---: | ---: | --- | --- | ---: | ---: |"]
    rows = {}
    for b in (4, 3):
        L.append(f"| uniform | W{b} | 0 | {b:.2f} | {uni[b]['rel_mse_mean']:.5f} | {uni[b]['rel_mse_mean']:.4f} @ {b:.2f} | — | "
                 f"{ufid[b]:.2f} | {ufid[b] - fp:+.2f} |")
        if s8[b]:
            e8 = s8[b]["extra_bits_per_param"]
            red = paired(s8[b], uni[b])
            L.append(f"| static-8 | W{b} | {e8:.4f} | {b + e8:.3f} | {s8[b]['rel_mse_mean']:.5f} | {s8[b]['rel_mse_mean']:.4f} @ {b + e8:.2f} | "
                     f"{100 * red[0]:+.1f}% [{100 * red[1]:+.1f}, {100 * red[2]:+.1f}] | "
                     + (f"{s8f[b]['fid']:.2f} | {s8f[b]['fid'] - fp:+.2f} |" if s8f[b] else "— | — |"))
        else:
            L.append(f"| static-8 | W{b} | — | — | pending | — | — | — | — |")
        for lab, mode, r in ARMS:
            for damp in ("", "_damp0.05"):
                n, q, lat, fid = arm(mode, r, b, damp)
                if q is None and lat is None:
                    continue
                rows[(lab + damp, b)] = (q, lat, fid)
                red = paired(lat, uni[b]) if lat else None
                eb = q["extra_bits_per_param"] if q else float("nan")
                L.append(f"| {lab}{' (damp 0.05)' if damp else ''} | W{b} | {eb:.4f} | {b + eb:.3f} | "
                         + (f"{lat['rel_mse_mean']:.5f} | {lat['rel_mse_mean']:.4f} @ {b + eb:.2f} | "
                            f"{100 * red[0]:+.1f}% [{100 * red[1]:+.1f}, {100 * red[2]:+.1f}]" if lat else "pending | — | —")
                         + (f" | {fid['fid']:.2f} | {fid['fid'] - fp:+.2f} |" if fid else " | — | — |"))
    L.append("")
    g = lambda lab, b: rows.get((lab, b), (None, None, None))
    L.append("### Matched-bits reading\n")
    for b in (4, 3):
        q, lat, fid = g("DC-r*", b)
        if lat and s8[b]:
            red = paired(lat, s8[b])
            L.append(f"- W{b}: DC-8 ({b + q['extra_bits_per_param']:.2f} bits/weight) vs static-8 ({b + s8[b]['extra_bits_per_param']:.2f}): "
                     f"latent MSE {lat['rel_mse_mean']:.5f} vs {s8[b]['rel_mse_mean']:.5f}, DC-8 vs static-8 {100 * red[0]:+.1f}% "
                     f"[{100 * red[1]:+.1f}, {100 * red[2]:+.1f}]" + (f"; FID-2k {fid['fid']:.2f} vs {s8f[b]['fid']:.2f}" if fid and s8f[b] else "") + ".")
        if lat and s32[b]:
            L.append(f"- W{b}: DC-8 at {b + q['extra_bits_per_param']:.2f} bits/weight: latent MSE {lat['rel_mse_mean']:.5f}; Gate A static-32 at its "
                     f"own {b + 0.593:.2f} bits/weight: {s32[b]['rel_mse_mean']:.5f} (not matched; static-32 spends "
                     f"{0.593 / q['extra_bits_per_param']:.1f}× the extra bits).")
    L.append("")
    v = {}
    dc4, dc3 = g("DC-r*", 4), g("DC-r*", 3)
    if dc3[2] is not None and dc4[2] is not None and dc4[1] is not None:
        rA = dc3[2]["fid"] <= 45
        rB = dc4[2]["fid"] <= 28.5 and dc4[1]["rel_mse_mean"] <= 0.90 * uni[4]["rel_mse_mean"]
        v = {"route_A": rA, "route_B": rB, "verdict": "PASS" if (rA or rB) else "STOP",
             "fid_dc_w3": dc3[2]["fid"], "fid_dc_w4": dc4[2]["fid"],
             "relmse_ratio_w4": dc4[1]["rel_mse_mean"] / uni[4]["rel_mse_mean"]}
        L.append("### Verdict\n")
        e3, e4 = 3 + dc3[0]["extra_bits_per_param"], 4 + dc4[0]["extra_bits_per_param"]
        v.update({"eff_bits_w3": e3, "eff_bits_w4": e4})
        L.append(f"- Route A: DC-r* W3 + carry, {e3:.2f} bits/weight: FID-2k {dc3[2]['fid']:.2f} (≤ 45: **{rA}**).")
        L.append(f"- Route B: DC-r* W4 + carry, {e4:.2f} bits/weight: FID-2k {dc4[2]['fid']:.2f} (≤ 28.5: {dc4[2]['fid'] <= 28.5}) and relMSE ratio vs uniform "
                 f"{v['relmse_ratio_w4']:.3f} (≤ 0.90: {v['relmse_ratio_w4'] <= 0.90}) → **{rB}**.")
        L.append(f"\n**G2: {v['verdict']}**" + (f" via Route B (W4 + carry, {e4:.2f} bits/weight, FID-2k {dc4[2]['fid']:.2f})" if rB else "")
                 + (f" via Route A (W3 + carry, {e3:.2f} bits/weight, FID-2k {dc3[2]['fid']:.2f})" if rA else "") + ".\n")
        L.append("Verdict written 2026-10-25 from the two DC-r* FIDs, the only inputs to the rule. The ablation and baseline "
                 "rows were still running and do not enter it.\n")
    L.append("### Findings (not stop rules)\n")
    for b in (4, 3):
        d, c = g("DC-r*", b)[1], g("C-r*", b)[1]
        if d and c:
            L.append(f"- W{b}: DC-r* / C-r* relMSE = {d['rel_mse_mean'] / c['rel_mse_mean']:.3f} (deflation earns its place at ≤ 0.95); "
                     f"DC-r* vs C-r* {100 * paired(d, c)[0]:+.1f}% [{100 * paired(d, c)[1]:+.1f}, {100 * paired(d, c)[2]:+.1f}].")
    d, c = g("DC-r*", 4)[1], g("C-r*", 4)[1]
    d3, c3 = g("DC-r*", 3)[1], g("C-r*", 3)[1]
    if d and c and d3 and c3:
        earn = d["rel_mse_mean"] <= 0.95 * c["rel_mse_mean"] and d3["rel_mse_mean"] <= 0.95 * c3["rel_mse_mean"]
        L.append(f"  → Hessian deflation earns its place: **{earn}**" + ("" if earn else "; the method is the carry alone."))
    w, dw = g("WSVD-r*", 3), g("DC-r*", 3)
    if w[1] and dw[1] and w[2] and dw[2]:
        better = dw[1]["rel_mse_mean"] < w[1]["rel_mse_mean"] and dw[2]["fid"] < w[2]["fid"]
        L.append(f"- Direction choice (W3): DC-r* relMSE {dw[1]['rel_mse_mean']:.5f} / FID {dw[2]['fid']:.2f} vs WSVD-r* "
                 f"{w[1]['rel_mse_mean']:.5f} / {w[2]['fid']:.2f} → activation-mean directions win on both: **{better}**.")
    o, dd = g("DC-1", 3)[1], g("DC-r*", 3)[1]
    if o and dd:
        L.append(f"- r = 1 vs r_l (W3): DC-1 / DC-r* relMSE = {o['rel_mse_mean'] / dd['rel_mse_mean']:.3f} → r = 1 suffices (within 5%): "
                 f"**{o['rel_mse_mean'] <= 1.05 * dd['rel_mse_mean']}**.")
    L.append("\n### Diagnostics\n")
    L.append("| arm | bits | extra bits: qkv / proj / fc1 / fc2 | median drift ratio vs uniform | max drift ratio (layer) | "
             "median clip frac (uniform) |")
    L.append("| --- | --- | --- | ---: | --- | --- |")
    for (lab, b), (q, _, _) in rows.items():
        if q is None:
            continue
        lay = q["layers"]
        mx = max(lay, key=lambda k: lay[k]["drift_ratio"])
        ebt = " / ".join("%.4f" % q["extra_bits_per_type"].get(t, 0) for t in TYPES)
        L.append(f"| {lab} | W{b} | {ebt} | "
                 f"{np.median([x['drift_ratio'] for x in lay.values()]):.3f} | {lay[mx]['drift_ratio']:.2f} ({mx}) | "
                 f"{np.median([x['clip_frac'] for x in lay.values()]):.4f} ({np.median([x['clip_frac_uniform'] for x in lay.values()]):.4f}) |")
    drift_bits = {b: max((q["max_drift_ratio"] for (lab, bb), (q, _, _) in rows.items() if q and bb == b and "damp" not in lab), default=0)
                  for b in (4, 3)}
    L.append("\nDrift rule (> 3× uniform in any layer → extra damping-0.05 arm): " +
             "; ".join(f"W{b} max ratio {drift_bits[b]:.2f} → {'triggered' if drift_bits[b] > 3 else 'not triggered'}" for b in (4, 3)) + ".\n")
    v["drift_max_ratio"] = drift_bits
    if os.path.exists(os.path.join(OUT, "g2_grid_w3.png")):
        L.append("Matched-seed grid, W3: `g2_grid_w3.png` (seeds 0–7; rows FP, uniform, then the W3 arms).\n")
    atomic_write_json(os.path.join(OUT, "g2_verdict.json"), v)
    path = os.path.join(OUT, "G2.md")
    pre = open(path).read().split(MARK)[0]
    atomic_write_text(path, pre + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
