"""cfb1 report: results/coupled_feedback_bridge/{REPORT.md, STATUS.md, artifacts.json} from the stage JSONs (no GPU)."""
import csv
import hashlib
import json
import os

OUT = "results/coupled_feedback_bridge"
CK = "ckpt/coupled_feedback_bridge"


def load(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def spent(tag="cfb1"):
    return sum(float(r[-1]) for r in csv.reader(open("results/gpu_hours.csv")) if r and r[0] == tag)


def ctab(C):
    L = ["| contrast | ratio of means [95% CI] | wins | ΔLPIPS [95% CI] | ΔPSNR |", "|---|---|---|---|---|"]
    for k, c in C.items():
        m, lp, ps = c["rel_mse"], c["lpips"], c["psnr"]
        L.append(f"| {k.replace('_vs_', ' / ')} | {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] | {m['wins_x']}/{m['n']} | "
                 f"{lp['diff']:+.4f} [{lp['diff_ci'][0]:+.4f}, {lp['diff_ci'][1]:+.4f}] | {ps['diff']:+.2f} |")
    return L


def stab(S, roles):
    L = ["| role | configuration | latent rel. MSE mean | median | PSNR | LPIPS |", "|---|---|---|---|---|---|"]
    for a, v in S.items():
        L.append(f"| {a} | {roles.get(a, '')} | {v['rel_mse']['mean']:.5f} | {v['rel_mse']['median']:.5f} | {v['psnr']['mean']:.2f} | {v['lpips']['mean']:.4f} |")
    return L


def main():
    pr, ch, ft, vl, cf, dg = (load(n) for n in ("profile.json", "checks.json", "fit.json", "val.json", "conf.json", "diag.json"))
    art = {f: {"path": os.path.join(CK, f), "sha256": sha(os.path.join(CK, f)), "bytes": os.path.getsize(os.path.join(CK, f))}
           for f in (sorted(os.listdir(CK)) if os.path.isdir(CK) else []) if f.endswith("_latents.pt")}
    json.dump({"note": "local only (standing instruction: nothing generated/trained is uploaded); regenerate with RUN.md", "files": art},
              open(os.path.join(OUT, "artifacts.json"), "w"), indent=1)
    st = lambda n, r: ("not run" if r is None else ("pass" if r.get("pass") else "FAIL") if n == "checks" else r.get("status", "complete" if "batch_size" in r else "?"))
    stages = (("profile", pr), ("checks", ch), ("fit", ft), ("val", vl), ("conf", cf), ("diag", dg))
    L = ["# cfb1 — zero-reference feedback with joint gains and bridge strength (pilot J): report", "",
         f"Protocol: `PROTOCOL.md` (amendments dated). Spend: **{spent():.3f} of 3.0 A30-h** (ledger tag cfb1).", "",
         "## Status", "| stage | status |", "|---|---|", *[f"| {n} | {st(n, r)} |" for n, r in stages]]
    if pr and "batch_size" in pr:
        L += ["", f"Frozen batch {pr['batch_size']}; profiled per-trajectory: J {pr['s_per_traj']['J']:.1f} s, FP50 {pr['s_per_traj']['FP50']:.1f} s; "
              f"planned total {pr['plan_total_h']:.2f} h."]
    if ch:
        L += ["", "## Correctness gate", f"Overall **{'pass' if ch['pass'] else 'FAIL'}**.",
              f"- overnight reproduction (batch 32): {ch['reproduce_overnight_bs32']}",
              f"- J(1,1)=AC incl. state: {ch['J(1,1)_nobridge=AC_incl_state']}; λ0=no bridge: {ch['lambda0_equals_no_bridge_incl_state']}; K1=FP50: {ch['K1_equals_FP50']}",
              f"- counts: {ch['counts_pass']}; independence {ch['independence']}; reset {ch['reset_between_batches']}; resume {ch['resume_bitwise']}; instrumentation {ch['instrumentation_on_off_bitwise']}",
              f"- gain placement: anchor g=0 = G call {ch['gain_anchor0_equals_G_call']}; between g=1 = AC call incl. state {ch['gain_between1_equals_AC_call_incl_state']}"]
    if ft and ft.get("status") == "complete":
        L += ["", "## Fit (120000–120031): complete 27-point surface (mean latent rel. MSE vs FP50)", "",
              "| g_anchor, g_between | λ=0.5 | λ=1 | λ=1.5 |", "|---|---|---|---|"]
        for ga in (0, 0.5, 1):
            for gb in (0, 0.5, 1):
                L.append(f"| {ga:g}, {gb:g} | " + " | ".join(f"{ft['surface_mean'][f'J({ga:g},{gb:g},{lam:g})']:.5f}" for lam in (0.5, 1, 1.5)) + " |")
        L += ["", "Selected (frozen before validation): " + ", ".join(f"{k} = {v} (fit {ft['selected_fit_error'][k]:.5f})" for k, v in ft["selected_names"].items())
              + (" — **the J family chose feedback off**" if ft["J_chose_feedback_off"] else "")]
    elif ft:
        L += ["", f"## Fit\n{ft.get('status')}" + (f" (required {ft['required_h']:.2f} h)" if "required_h" in ft else "")]
    if vl and vl.get("status") == "complete":
        L += ["", "## Validation (120100–120131)", *stab(vl["summary"], vl["roles"]), "", *ctab(vl["contrasts"]), "",
              f"Gate ({vl['gate']['rule']}): J nonzero feedback {vl['gate']['J_nonzero_feedback']}, ratio J/B {vl['gate']['ratio_J_over_B']:.3f}, "
              f"ΔLPIPS {vl['gate']['dlpips_J_minus_B']:+.4f} → **{'pass' if vl['gate']['pass'] else 'fail'}**" + (f" ({vl['gate']['note']})" if "note" in vl["gate"] else "")]
    if cf and cf.get("status") == "complete":
        L += ["", "## Confirmation (121000–121255, 256 fresh)", *stab(cf["summary"], cf["roles"]), "", *ctab(cf["contrasts"]), "",
              f"Primary J/B: **{cf['reading_primary']}**. J/U: {cf['reading_J_vs_U']}.", cf["note"]]
    elif cf:
        L += ["", f"## Confirmation\n{cf.get('status')}"]
    if dg and dg.get("status") == "complete":
        L += ["", "## Diagnostics (120000–120007)", "| live configuration | endpoint | E‖H‖² | E⟨H,v⟩ | E‖v‖² | cos | λ_LS (frozen window, not used) | anchor raw | between raw |",
              "|---|---|---|---|---|---|---|---|---|"]
        for k, r in dg["live"].items():
            L.append(f"| {k} | {r['endpoint_rel_mse_mean']:.5f} | {r['HH']:.4g} | {r['Hv']:.4g} | {r['vv']:.4g} | {r['cos_H_v']:.3f} | {r['lambda_ls_frozen_window']:.3f} | {r['raw_anchor']:.4g} | {r['raw_between']:.4g} |")
        L += ["", "| forced trajectory (FP50 latents) | E‖H‖² | E⟨H,v⟩ | E‖v‖² | cos | anchor raw | between raw |", "|---|---|---|---|---|---|---|"]
        for k, r in dg["forced"].items():
            L.append(f"| {k} | {r['HH']:.4g} | {r['Hv']:.4g} | {r['vv']:.4g} | {r['cos_H_v']:.3f} | {r['raw_anchor']:.4g} | {r['raw_between']:.4g} |")
    ans = os.path.join(OUT, "ANSWERS.md")
    if os.path.exists(ans):
        L += ["", open(ans).read()]
    open(os.path.join(OUT, "REPORT.md"), "w").write("\n".join(L) + "\n")
    open(os.path.join(OUT, "STATUS.md"), "w").write("# cfb1 status\n\n" + f"Spend {spent():.3f} of 3.0 A30-h. " +
                                                    ", ".join(f"{n}: {st(n, r)}" for n, r in stages) + "\n")


if __name__ == "__main__":
    main()
