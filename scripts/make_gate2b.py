"""Append/refresh the results section of results/gate2b.md from JSONs. The pre-registration text above the marker is
never modified."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from boa_dit.common import atomic_write_text  # noqa: E402
import arm_weights as AW  # noqa: E402
import run_gate2b as R  # noqa: E402

MARK = "\n---\n\n## Results (appended after the runs by `scripts/make_gate2b.py`)\n"
DESC = {"a": "gates-only, K=4", "b": "gain-only, λ=0.5, K=1", "c": "full, λ=0.5, K=4",
        "d": "gain-only, tempered τ*, K=1", "e": "full, tempered τ*, K=4", "f": "gain-only, clipped max/min ≤ 10, K=1"}
EFF = {"a": dict(mode="gates", K=4), "b": dict(mode="gain", lam=0.5), "c": dict(mode="full", K=4, lam=0.5),
       "d": None, "e": None, "f": dict(mode="gain", clip=10)}


def paired(x, y):
    x, y = np.array(x["rel_mse_per_image"]), np.array(y["rel_mse_per_image"])
    rng = np.random.default_rng(0)
    bs = [1 - x[i].mean() / y[i].mean() for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return 1 - x.mean() / y.mean(), lo, hi, float((x - y < 0).mean())


def main():
    path = os.path.join(ROOT, "results", "gate2b.md")
    pre = open(path).read().split(MARK)[0]
    L = [MARK]
    ts, held = R.tau_star_static() if hasattr(R, "tau_star_static") else R.Plan().tau_star()
    gain_eff = AW.effective("gain")
    L.append(f"**Headline diagnostic (peakedness):** the fp32 median gain alone puts {100 * gain_eff['share_steps_0_14']:.1f}% "
             f"of its weight on steps 0–14 and {100 * gain_eff['share_steps_25_49']:.1f}% on steps 25–49 "
             f"(max/min over buckets {gain_eff['max_over_min']:.0f}).\n")
    L.append("### τ selection on held-out calibration seeds 1032–1095 (64 images, W4, gain-only K=1)\n")
    L.append("| τ | held-out rel. latent MSE | steps 0–14 | steps 25–49 |")
    L.append("| ---: | ---: | ---: | ---: |")
    for t in R.TAUS:
        h = held.get(t) if held else None
        e = AW.effective("gain", tau=t)
        L.append(f"| {t:g} | " + (f"{h['rel_mse_mean']:.5f} ± {h['rel_mse_sem']:.5f}" if h else "—") +
                 f" | {e['share_steps_0_14']:.3f} | {e['share_steps_25_49']:.3f} |")
    L.append(f"\nτ* = **{ts if ts is not None else 'pending'}**\n")
    u4 = R.lat(R.UNIFORM[4])
    L.append(f"### W4 arms vs `uniform` (`{R.UNIFORM[4]}`, rel. latent MSE {u4['rel_mse_mean']:.5f}), eval seeds 0–511\n")
    L.append("| arm | definition | name | steps 0–14 | steps 25–49 | rel. latent MSE | reduction vs uniform | 95% CI | images improved |")
    L.append("| --- | --- | --- | ---: | ---: | ---: | ---: | --- | ---: |")
    best, best_red = None, -1e9
    for x in "abcdef":
        if x in "de" and ts is None:
            L.append(f"| {x} | {DESC[x]} | — | — | — | — | — | — | — |")
            continue
        nm = R.name_of(R.arm_args(x, ts))
        eff = AW.effective(**(EFF[x] or dict(mode="gain" if x == "d" else "full", K=1 if x == "d" else 4, tau=ts)))
        r = R.lat(nm)
        if r is None:
            L.append(f"| {x} | {DESC[x]} | `{nm}` | {eff['share_steps_0_14']:.3f} | {eff['share_steps_25_49']:.3f} | "
                     "not run (cap) or pending | — | — | — |")
            continue
        red, lo, hi, imp = paired(r, u4)
        if red > best_red:
            best, best_red = x, red
        L.append(f"| {x} | {DESC[x]} | `{nm}` | {eff['share_steps_0_14']:.3f} | {eff['share_steps_25_49']:.3f} | "
                 f"{r['rel_mse_mean']:.5f} ± {r['rel_mse_sem']:.5f} | **{100 * red:+.1f}%** | "
                 f"[{100 * lo:+.1f}%, {100 * hi:+.1f}%] | {100 * imp:.0f}% |")
    L.append("")
    w3 = None
    if best is not None:
        r3 = R.lat(R.name_of(R.arm_args(best, ts, bits=3)))
        if r3 is not None:
            red3, lo3, hi3, imp3 = paired(r3, R.lat(R.UNIFORM[3]))
            w3 = red3
            L.append(f"### W3 rerun of the best W4 arm ({best})\n")
            L.append(f"rel. latent MSE {r3['rel_mse_mean']:.5f} vs uniform {R.lat(R.UNIFORM[3])['rel_mse_mean']:.5f}: "
                     f"reduction **{100 * red3:+.1f}%**, 95% CI [{100 * lo3:+.1f}%, {100 * hi3:+.1f}%], "
                     f"{100 * imp3:.0f}% of images improved.\n")
    L.append("### Execution notes\n")
    L.append("- **Arm d was evaluated twice; the first result is invalid.** It was quantized on cuda:1 and evaluated on "
             "cuda:0. GPU peer-to-peer copies on this host are silently wrong (a cuda:1 → cuda:0 `copy_` returns bad "
             "data with no error, although peer access is reported available), and `torch.load` without "
             "`map_location` restores tensors onto the saving GPU, so `load_state_dict` left the FP weights in place "
             "and the eval scored FP against FP (rel. MSE exactly 0). The invalid JSON is kept in "
             "`results/stage2/eval_invalid/`. All checkpoint loads now go through CPU and are verified tensor by tensor "
             "(`boa_dit.model.load_quant_state`). Every other arm was quantized and evaluated on the same GPU; no other "
             "result crossed GPUs. The wasted eval (≈ 0.08 GPU-h) counts toward the cap.")
    L.append("- **Arm f: not run (cap).** When cuda:1 finished arm c, spent + f's projected cost + the W3 reserve exceeded "
             "1.5 GPU-h, so the driver took the skip branch. Its SKIP log line is missing because the driver called "
             "`log()` while holding the same non-reentrant lock and deadlocked there; the driver was then stopped by hand.")
    L.append("- No W3 rerun: no W4 arm reached 5%.\n")
    spent = R.spent(R.all_names())
    L.append(f"GPU-hours spent on the set: **{spent:.2f}** (cap 1.5).\n")
    L.append("### Decision (pre-registered rule)\n")
    if best is None:
        L.append("Pending.")
    else:
        w4_ok = best_red >= 0.05
        L.append(f"- Best W4 arm: **{best}** ({DESC[best]}), reduction {100 * best_red:+.1f}% (needs ≥ 5%): **{w4_ok}**.")
        if w4_ok:
            L.append(f"- W3 rerun reduction: " + ("pending" if w3 is None else f"{100 * w3:+.1f}% (needs ≥ 5%): **{w3 >= 0.05}**."))
        passed = w4_ok and w3 is not None and w3 >= 0.05
        L.append(f"\n**Gate 2b: {'PASS → Stage 3 with arm ' + best + ' as full' if passed else ('PENDING' if (w4_ok and w3 is None) else 'STOP')}**\n")
    atomic_write_text(path, pre + "\n".join(L) + "\n")
    print("written")


if __name__ == "__main__":
    main()
