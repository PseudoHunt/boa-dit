"""rfb1 report: results/reference_feedback_bridge/{REPORT.md, STATUS.md, artifacts.json} from the stage JSONs (no GPU)."""
import csv
import hashlib
import json
import os

OUT = "results/reference_feedback_bridge"
CK = "ckpt/reference_feedback_bridge"


def load(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def spent(tag="rfb1"):
    return sum(float(r[-1]) for r in csv.reader(open("results/gpu_hours.csv")) if r and r[0] == tag)


def ctab(C, keys):
    L = ["| contrast | ratio of means [95% CI] | wins | ΔLPIPS [95% CI] | ΔPSNR |", "|---|---|---|---|---|"]
    for k in keys:
        if k not in C:
            continue
        m, lp, ps = C[k]["rel_mse"], C[k]["lpips"], C[k]["psnr"]
        L.append(f"| {k.replace('_vs_', ' / ')} | {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] | {m['wins_x']}/{m['n']} | "
                 f"{lp['diff']:+.4f} [{lp['diff_ci'][0]:+.4f}, {lp['diff_ci'][1]:+.4f}] | {ps['diff']:+.2f} |")
    return L


def stab(S):
    L = ["| arm | latent rel. MSE mean | median | PSNR | LPIPS |", "|---|---|---|---|---|"]
    for a, v in S.items():
        L.append(f"| {a} | {v['rel_mse']['mean']:.5f} | {v['rel_mse']['median']:.5f} | {v['psnr']['mean']:.2f} | {v['lpips']['mean']:.4f} |")
    return L


def main():
    pr, ch, dv, cf, dg, bud = (load(n) for n in ("profile.json", "checks.json", "dev.json", "conf.json", "diag.json", "budget.json"))
    art = {}
    for f in sorted(os.listdir(CK)) if os.path.isdir(CK) else []:
        if f.endswith("_latents.pt"):
            art[f] = {"path": os.path.join(CK, f), "sha256": sha(os.path.join(CK, f)), "bytes": os.path.getsize(os.path.join(CK, f))}
    json.dump({"note": "local only (standing instruction: nothing generated/trained is uploaded); regenerate with the stage commands in RUN.md",
               "files": art}, open(os.path.join(OUT, "artifacts.json"), "w"), indent=1)
    st = lambda r: "not run" if r is None else r.get("status", "complete" if r.get("pass") is not None else "?")
    L = ["# rfb1 — anchor-referenced feedback + anchor reuse + bridge (pilot R): report", "",
         f"Protocol: `PROTOCOL.md` (amendments dated). Spend: **{spent():.3f} of 3.0 A30-h** (ledger tag rfb1).", "",
         "## Status", "| stage | status |", "|---|---|"]
    for n, r in (("profile", pr), ("checks", ch), ("dev", dv), ("conf", cf), ("diag", dg)):
        s = ("pass" if r and r.get("pass") else "FAIL") if n == "checks" and r else st(r)
        L.append(f"| {n} | {s} |")
    if pr:
        L += ["", f"Frozen batch size **{pr['batch_size']}** (R-full peak reserved at 16: {pr['rfull']['16']['peak_reserved'] / 2**30:.1f} GiB; 32: "
              f"{'OOM' if pr['rfull']['32'].get('oom') else 'fits'}). R-full state + reference: {pr['rfull']['16']['state_bytes_per_image_incl_cfg'] / 1e6:.0f} MB per image (both CFG rows)."]
    if ch:
        L += ["", "## Correctness gate", f"Overall: **{'pass' if ch['pass'] else 'FAIL'}**.",
              f"- S1 overnight reproduction at batch 32: {ch['S1_reproduce_overnight_bs32']}",
              f"- S2 λ=0: {ch['S2_lambda0_equals_no_bridge']}; S3 K1=FP50: {ch['S3_K1_equals_FP50']}; S4 counts/inputs: {ch['S4_pass']}",
              f"- S5 independence/reset: {ch['S5_independence_reset']}; unit row independence: {ch['row_independence_unit']}",
              f"- S6 resume: {ch['S6_resume_bitwise']}; S7 instrumentation on/off: {ch['S7_instrumentation_on_off_bitwise']}",
              f"- S8 no-anchor R-DCT4 = feedback: {ch['S8_no_anchor_Rdct_equals_feedback']}; R anchor call = G: {ch['S8_R_anchor_call_equals_G']}",
              f"- S9 accounting identity rel. residual energy: RFB {ch['S9_accounting_RFB']['identity_residual_rel_energy']:.2e}, RDB {ch['S9_accounting_RDB']['identity_residual_rel_energy']:.2e}",
              f"- S10 bytes per image (incl. CFG): " + "; ".join(f"{a}: state {v['rolling_state_bytes_per_image'] / 1e6:.2f} MB + ref {v['reference_bytes_per_image'] / 1e6:.2f} MB" for a, v in ch["S10_state_bytes"].items()),
              f"- DDIM coefficients: max rel {ch['ddim_coefficients']['max_rel_l2_vs_ddim_fp32']:.1e}; synthetic: {ch['synthetic_constant_linear']['pass']}"]
    keys = ["RDB_vs_GB", "RFB_vs_GB", "RDB_vs_RFB", "RDB_vs_ACB", "RFB_vs_ACB", "RD_vs_G", "RDB_vs_RD", "ACB_vs_GB", "GB_vs_G"]
    if dv and dv.get("status") == "complete":
        L += ["", "## Development (60200–60231, inspected, exploratory)", *stab(dv["summary"]), "", *ctab(dv["contrasts"], keys), "",
              f"Gate ({dv['gate_rule']}): " + "; ".join(f"{a}: ratio {g['ratio_vs_GB']:.3f}, ΔLPIPS {g['dlpips_vs_GB']:+.4f} → {'pass' if g['pass'] else 'fail'}" for a, g in dv["gate"].items())
              + f" → **{'confirmation run' if dv['gate_pass'] else 'confirmation not run (cohort 122000–122127 left unopened)'}**."]
    if cf and cf.get("status") == "complete":
        L += ["", "## Confirmation (122000–122127, fresh, 128 images)", *stab(cf["summary"]), "", *ctab(cf["contrasts"], keys), "",
              "Readings: " + "; ".join(f"{k.replace('_vs_', '/')} ({cf['labels'][k]}): {v}" for k, v in cf["readings"].items()),
              "", "Secondary intervals exploratory/unadjusted; nonsignificance is not equivalence."]
    elif cf:
        L += ["", f"## Confirmation\n{cf.get('status')}"]
    if dg and dg.get("status") == "complete":
        L += ["", "## Diagnostics (60208–60215)", "| arm | endpoint rel. MSE | raw | held | jump | cos(H,v) | λ_LS frozen window | replay cos(dx,v) 0/20/40 |", "|---|---|---|---|---|---|---|---|"]
        for a, r in dg["arms"].items():
            rp = " / ".join("%.2f" % v["cos_dx_v"] for v in r["replay"].values())
            L.append(f"| {a} | {r['endpoint_rel_mse_vs_FP50_mean']:.5f} | {r['raw']:.4g} | {r['held']:.4g} | {r['jump']:.4g} | {r['cos_H_v']:.3f} | "
                     f"{r['lambda_ls_frozen_window']:.3f} | {rp} |")
        L += ["", "R accounting (all intervals, all sites; energy sums):"]
        for a in ("RFB", "RDB", "RD"):
            r = dg["arms"].get(a, {})
            if "R_accounting_per_interval" in r:
                tot = {}
                for v in r["R_accounting_per_interval"].values():
                    for k, x in v.items():
                        tot[k] = tot.get(k, 0.0) + x
                L.append(f"- {a}: " + ", ".join(f"{k} {v:.4g}" for k, v in sorted(tot.items()) if k.startswith("E[")))
        L += ["", "Timing / memory (warm synchronized trajectory at the frozen batch; fake-quant simulation, not packed-kernel evidence):"]
        for a, v in dg["timing_memory"].items():
            L.append(f"- {a}: {v['warm_s_per_trajectory']:.1f} s per {v['batch']}-image trajectory; state + reference {v['state_plus_reference_bytes_per_image_incl_cfg'] / 1e6:.2f} MB/image")
    ans = os.path.join(OUT, "ANSWERS.md")
    if os.path.exists(ans):
        L += ["", open(ans).read()]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L) + "\n")
    S = ["# rfb1 status", "", f"Spend {spent():.3f} of 3.0 A30-h. Stages: " + ", ".join(f"{n}: {st(r) if n != 'checks' else ('pass' if r and r.get('pass') else st(r))}"
                                                                              for n, r in (("profile", pr), ("checks", ch), ("dev", dv), ("conf", cf), ("diag", dg))),
         "", "Running: nothing (see REPORT.md)." if dg else "Queue in progress or stopped; see REPORT.md and RUN.md."]
    open(os.path.join(OUT, "STATUS.md"), "w").write("\n".join(S) + "\n")


if __name__ == "__main__":
    main()
