"""Premise check: results below the marker in results/prior/rule.md, verdict.json and SUMMARY.md (all from JSONs)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from boa_dit.common import atomic_write_json, atomic_write_text  # noqa: E402
from make_gateA import paired  # noqa: E402

OUT = os.path.join(ROOT, "results", "prior")
MARK = "\n---\n\n## Results (appended by `scripts/make_prior.py`)\n"
FP_FID = 23.89
DESC = {"A1": "Q,K FP, all blocks, all steps", "A2": "Q,K FP, all blocks, steps < 14", "A3": "Q,K FP, all blocks, steps ≥ 14",
        "A4": "V FP, all blocks, all steps", "A5": "Q,K FP, block 2, all steps", "A6": "Q,K FP, blocks 0–4, all steps",
        "A7": "Q,K FP, blocks 0–4, steps < 14", "B2": "W4 + deployed D̄, blocks 0–4, steps < 14",
        "B2_fpin": "W4 + FP-input D̄", "B2_r16": "W4 + deployed D̄ at rank 16"}


def load(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    base = load("eval/latent_base.json")
    anchors = json.load(open(os.path.join(ROOT, "results", "stage2", "fid_anchors.json")))["anchors"]
    fid_base = anchors["uniform W4"]["fid2k"]
    fb = load("eval/fid_base.json")
    L = [MARK, f"Integrity: base reproduces the stored W4 per-image errors on seeds 0–127 (max abs diff "
         f"{base['integrity_max_abs_diff_vs_stored']:.1e}). Base rel. latent MSE (0–127) {base['rel_mse_mean']:.5f}; base FID-2k "
         f"{fid_base:.2f} (anchor, reused); FP {FP_FID}.\n",
         "### Part A (share of base error = 1 − relMSE(arm)/relMSE(base); paired bootstrap 95% CI, seeds 0–127)\n",
         "| arm | what is FP | rel latent MSE | share of base error removed (95% CI) | FID-2k | label acc | changed class |",
         "| --- | --- | ---: | --- | ---: | ---: | ---: |",
         f"| base | nothing | {base['rel_mse_mean']:.5f} | — | {fid_base:.2f} | "
         + (f"{100 * fb['label_acc_512']:.1f}% | {100 * fb['class_changed_frac_512']:.1f}% |" if fb else "pending | pending |")]
    sh = {}
    for a in ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "B2", "B2_fpin", "B2_r16"):
        r = load(f"eval/latent_{a}.json")
        if r is None:
            continue
        red, lo, hi, _ = paired(r, base)
        sh[a] = (red, lo, hi, r["rel_mse_mean"])
        f = load(f"eval/fid_{a}.json")
        L.append(f"| {a} | {DESC[a]} | {r['rel_mse_mean']:.5f} | {100 * red:+.1f}% [{100 * lo:+.1f}, {100 * hi:+.1f}] | "
                 + (f"{f['fid2k']:.2f} | {100 * f['label_acc_512']:.1f}% | {100 * f['class_changed_frac_512']:.1f}% |" if f else "— | — | — |"))
        if a == "A7":
            L.append("| | *Part B arms below; same base* | | | | | |")
    L.append("\nShares need not add up: routing (Q,K) and value errors interact.")
    if "A1" in sh and "A4" in sh:
        L.append(f"- Q,K vs V: A1 removes {100 * sh['A1'][0]:.1f}%, A4 removes {100 * sh['A4'][0]:.1f}% → "
                 f"**{'Q,K (routing)' if sh['A1'][0] > sh['A4'][0] else 'V (values)'} dominates**.")
    if "A2" in sh and "A3" in sh:
        L.append(f"- Early vs late routing: A2 (steps < 14) {100 * sh['A2'][0]:.1f}%, A3 (steps ≥ 14) {100 * sh['A3'][0]:.1f}%.")
    L.append("")
    b1 = load("b1.json")
    if b1:
        P = b1["per"]
        L.append("### Part B1 (median over the 16 heads; cond branch; shared = ‖D̄‖² / mean ‖D‖² over the bucket's 160 "
                 "(image, step) samples; per-step version in brackets)\n")
        L.append("| block | def | bucket 0 | bucket 1 | bucket 2 | bucket 7 | bucket 8 | bucket 9 | branch cos (b0) | rank-16 energy (b0) | kernel cos (b0) |")
        L.append("| ---: | --- |" + " ---: |" * 9)
        for l in b1["blocks"]:
            for d in ("deployed", "fpin"):
                c = [P[f"{d}.b{l}.k{k}.br0"] for k in b1["buckets"]]
                L.append(f"| {l} | {d} | " + " | ".join(f"{x['shared_median']:.2f} [{x['shared_per_step_median']:.2f}]" for x in c)
                         + f" | {c[0]['branch_cos_median']:.2f} | {c[0]['energy_rank']['16']:.2f} | {c[0]['kernel_cos_median']:.2f} |")
        L.append("\nRank energy of D̄ (bucket 0, deployed, cond; median over heads), rank 4 / 8 / 16 / 32: " + "; ".join(
            f"b{l} " + " / ".join(f"{P[f'deployed.b{l}.k0.br0']['energy_rank'][r]:.2f}" for r in ("4", "8", "16", "32")) for l in b1["blocks"]) + ".\n")
    v = {}
    if "A1" in sh:
        fa1 = load("eval/fid_A1.json")
        c1_lat = sh["A1"][0] >= 0.30 and sh["A1"][1] > 0.20
        c1_fid = (fid_base - fa1["fid2k"]) >= 0.5 * (fid_base - FP_FID) if fa1 else None
        v["criterion1"] = {"A1_share": sh["A1"][0], "A1_ci": [sh["A1"][1], sh["A1"][2]], "latent_ok": c1_lat,
                           "A1_fid2k": fa1["fid2k"] if fa1 else None, "fid_gain": (fid_base - fa1["fid2k"]) if fa1 else None,
                           "fid_needed": 0.5 * (fid_base - FP_FID), "fid_ok": c1_fid,
                           "pass": bool(c1_lat and c1_fid) if c1_fid is not None else (False if not c1_lat else None)}
    if b1 and "B2" in sh and "A7" in sh:
        qual = [l for l in range(5) if all(b1["per"][f"deployed.b{l}.k{k}.br0"]["shared_median"] >= 0.5 for k in (0, 1, 2))]
        rec = (base["rel_mse_mean"] - sh["B2"][3]) / (base["rel_mse_mean"] - sh["A7"][3])
        v["criterion2"] = {"blocks_qualifying": qual, "shared_ok": len(qual) >= 3, "recovered_fraction": rec, "recovered_ok": rec >= 0.5,
                           "recovered_fpin": (base["rel_mse_mean"] - sh["B2_fpin"][3]) / (base["rel_mse_mean"] - sh["A7"][3]) if "B2_fpin" in sh else None,
                           "recovered_r16": (base["rel_mse_mean"] - sh["B2_r16"][3]) / (base["rel_mse_mean"] - sh["A7"][3]) if "B2_r16" in sh else None,
                           "pass": len(qual) >= 3 and rec >= 0.5}
    L.append("### Verdict\n")
    if "criterion1" in v:
        c = v["criterion1"]
        L.append(f"1. Routing error large: A1 removes {100 * c['A1_share']:.1f}% [{100 * c['A1_ci'][0]:.1f}, {100 * c['A1_ci'][1]:.1f}] "
                 f"(needs ≥ 30%, CI > 20%): **{c['latent_ok']}**; FID-2k gain " +
                 (f"{c['fid_gain']:+.2f} (needs ≥ {c['fid_needed']:.2f}): **{c['fid_ok']}**" if c["fid_gain"] is not None else "pending")
                 + f" → **criterion 1 {c['pass']}**.")
    if "criterion2" in v:
        c = v["criterion2"]
        L.append(f"2. Shared and repairable: blocks 0–4 with median shared ≥ 0.5 in each of buckets 0–2: {c['blocks_qualifying']} "
                 f"(needs ≥ 3): **{c['shared_ok']}**; B2 recovered fraction {c['recovered_fraction']:.2f} of A7's gain (needs ≥ 0.50): "
                 f"**{c['recovered_ok']}**" + (f" (FP-input D̄: {c['recovered_fpin']:.2f}" if c["recovered_fpin"] is not None else "")
                 + (f"; rank 16: {c['recovered_r16']:.2f})" if c["recovered_r16"] is not None else ")") + f" → **criterion 2 {c['pass']}**.")
    stop = (v.get("criterion1", {}).get("pass") is False) or (v.get("criterion2", {}).get("pass") is False)
    v["verdict"] = "STOP before Part C" if stop else "pending"
    L.append(f"\n**{v['verdict']}**" + (": the shared-prior quantization mechanism is not supported on DiT-XL/2." if stop else "") + "\n")
    atomic_write_json(os.path.join(OUT, "verdict.json"), v)
    path = os.path.join(OUT, "rule.md")
    pre = open(path).read().split(MARK)[0]
    atomic_write_text(path, pre + "\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
