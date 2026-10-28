"""Parts 1-2 on the real model (PROTOCOL.md): one pass over the 112 heig-8 layers (frozen grid, saved PHP, BoA preprocessing).
Per layer:
  audit: static GPTQ with per-column incremental cost; checked to reproduce heig-8's W_q exactly
  constructions (beta = 0.1, selection per layer): B final column, C final 16, D highest-cost 16 (static audit cost),
         E 16 random (generator seed 1000 + layer index), F all columns
  recon (damped PHP metric): tr(E_A H E_A^T), tr(E_B H E_B^T), shared ||(E_A+E_B)/2||^2_H, alternating ||(E_A-E_B)/2||^2_H,
         differing-code fraction; each table's carry delta_X = (W - Q_X) V (fp16, V shared)
  --hp SEL: static extra precision on SEL's columns (half-step grid there, W4 elsewhere), after the screen
Outputs ckpt/weights_joint/<construction>.pt (uint8 codes + carries), results/temporal/weights_joint/audit.json, build.json.
Run: python -m boa_dit.weights_joint.build [--hp D]
"""
import argparse
import os
import time

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import make_carry
from boa_dit.model import block_linear, load_dit, load_quant_state
from boa_dit.weights_joint.common import CK, HEIG, HEIG_LAYERS, OUT, STAGE, guard
from boa_dit.weights_joint.joint import gptq_static, joint, prep, quad

BETA = 0.1
K = 16
CONS = ("B_final1", "C_final16", "D_top16", "E_rand16", "F_all")


def selections(d, cost, li):
    s = {c: torch.zeros(d, dtype=torch.bool) for c in CONS}
    s["B_final1"][-1:] = True
    s["C_final16"][-K:] = True
    s["D_top16"][torch.topk(cost.cpu(), K).indices] = True
    s["E_rand16"][torch.randperm(d, generator=torch.Generator().manual_seed(1000 + li))[:K]] = True
    s["F_all"][:] = True
    return s


def frac(cost, idx):
    return float(cost[idx].sum() / cost.sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hp", default=None, help="construction whose selected columns get static half-step precision")
    ap.add_argument("--audit_only", action="store_true", help="re-run only the static audit (after the rounding / H_eff fix)")
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    S = torch.load(HEIG_LAYERS, map_location="cpu")
    W_fp = load_dit("cpu")
    W_q = load_quant_state(load_dit("cpu"), os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt"))
    os.makedirs(CK, exist_ok=True)
    keys = list(S.keys())
    if a.hp:
        sel = torch.load(os.path.join(CK, f"{a.hp}.pt"), map_location="cpu")
        guard(0.05, f"build hp_{a.hp}")
    elif a.audit_only:
        guard(0.08, "audit re-run + recon from stored codes")
        stored = {c: torch.load(os.path.join(CK, f"{c}.pt"), map_location="cpu") for c in CONS}
    else:
        guard(0.25, "audit + constructions B-F")
    out = {}
    audit, build = {"layers": {}}, {"beta": BETA, "k": K, "layers": {}}
    with Run(STAGE, "build" if not (a.hp or a.audit_only) else (f"build_hp_{a.hp}" if a.hp else "audit_rerun"), dev, vars(a)) as run:
        t0 = time.time()
        for li, key in enumerate(keys):
            s = S[key]
            l, ln = key.split(".", 1)
            Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            Wq1 = block_linear(W_q.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            scale, zero, V = s["scale"].to(dev), s["zero"].to(dev), s["V"].to(dev)
            Wp, Hc, U = prep(Wf, s["H"].to(dev), s["bits"], s["replace"])
            d = Wp.shape[1]
            if a.hp:
                hc = sel["sel"][key].to(dev)
                Qh, ch, _ = gptq_static(Wp, U, scale, zero, 15, half_cols=hc)
                out[key] = {"codes": ch.to(torch.uint8).cpu(), "half_cols": hc.cpu(), "delta": make_carry(Wf, Qh, V)["delta"]}
                build["layers"][key] = {"recon": quad(Qh - Wp, Hc), "half_cols": int(hc.sum())}
                continue
            Q, codes, cost = gptq_static(Wp, U, scale, zero, 15)
            order = torch.argsort(cost, descending=True)
            audit["layers"][key] = {"d_in": d, "reproduces_heig8": bool(torch.equal(Q, Wq1)), "mismatch_frac": float((Q != Wq1).float().mean()),
                                    "total_cost": float(cost.sum()), "damped_quadratic": quad(Q - Wp, Hc),
                                    "final1": frac(cost, slice(d - 1, d)), "final16": frac(cost, slice(d - 16, d)), "final64": frac(cost, slice(d - 64, d)),
                                    "top16": frac(cost, order[:16]), "top64": frac(cost, order[:64]), "cost": cost.float().cpu().tolist()}
            if a.audit_only:                                                  # recompute recon components from stored codes with H_eff
                rec = {"static": {"recon": quad(Wq1 - Wp, Hc)}}
                for c in CONS:
                    lay = stored[c]["layers"][key]
                    QA = scale * (lay["cA"].to(dev).float() - zero); QB = scale * (lay["cB"].to(dev).float() - zero)
                    EA, EB = QA - Wp, QB - Wp
                    rec[c] = {"recon_A": quad(EA, Hc), "recon_B": quad(EB, Hc), "shared": quad((EA + EB) / 2, Hc), "alternating": quad((EA - EB) / 2, Hc),
                              "frac_codes_differ": float((lay["cA"] != lay["cB"]).float().mean())}
                rec["D_top16_selection_overlap_with_rerun"] = float((stored["D_top16"]["sel"][key] & torch.zeros(d, dtype=torch.bool).index_fill_(
                    0, torch.topk(cost.cpu(), K).indices, True)).sum() / K)
                build["layers"][key] = rec
                continue
            sels = selections(d, cost, li)
            E1 = Wq1 - Wp
            rec = {"static": {"recon": quad(E1, Hc)}}
            for c in CONS:
                QA, QB, cA, cB, jc = joint(Wp, U, scale, zero, 15, BETA, sels[c].to(dev))
                EA, EB = QA - Wp, QB - Wp
                rec[c] = {"recon_A": quad(EA, Hc), "recon_B": quad(EB, Hc), "shared": quad((EA + EB) / 2, Hc), "alternating": quad((EA - EB) / 2, Hc),
                          "surrogate": float(jc.sum()), "frac_codes_differ": float((cA != cB).float().mean()), "n_vary": int(sels[c].sum())}
                out.setdefault(c, {})[key] = {"cA": cA.to(torch.uint8).cpu(), "cB": cB.to(torch.uint8).cpu(), "dA": make_carry(Wf, QA, V)["delta"],
                                              "dB": make_carry(Wf, QB, V)["delta"], "sel": sels[c]}
            build["layers"][key] = rec
            if li == 0:
                el = time.time() - t0
                print(f"  pilot: layer 0 took {el:.1f} s -> projected {el * len(keys) / 3600 * 1.5:.3f} h for 112 layers", flush=True)
            if li % 16 == 0:
                print(f"  {key}: reproduces heig-8 {audit['layers'][key]['reproduces_heig8']}, final16 {audit['layers'][key]['final16']:.3f}, "
                      f"top16 {audit['layers'][key]['top16']:.3f}", flush=True)
        if a.hp:
            atomic_torch_save({"scale": {k: S[k]["scale"] for k in keys}, "zero": {k: S[k]["zero"] for k in keys},
                               "V": {k: S[k]["V"].half() for k in keys}, "layers": out}, os.path.join(CK, f"hp_{a.hp}.pt"))
        elif not a.audit_only:
            for c in CONS:
                atomic_torch_save({"scale": {k: S[k]["scale"] for k in keys}, "zero": {k: S[k]["zero"] for k in keys},
                                   "V": {k: S[k]["V"].half() for k in keys}, "sel": {k: out[c][k]["sel"] for k in keys},
                                   "layers": {k: {x: v for x, v in out[c][k].items() if x != "sel"} for k in keys}}, os.path.join(CK, f"{c}.pt"))
        rec = run.record()
    if a.hp:
        atomic_write_json(os.path.join(OUT, f"build_hp_{a.hp}.json"), {**build, **rec})
        return
    L = audit["layers"]
    w = np.array([v["total_cost"] for v in L.values()])
    agg = {f: float((np.array([v[f] for v in L.values()]) * w).sum() / w.sum()) for f in ("final1", "final16", "final64", "top16", "top64")}
    audit["aggregate_cost_weighted"] = agg
    audit["all_layers_reproduce_heig8"] = all(v["reproduces_heig8"] for v in L.values())
    audit["telescoping_max_rel"] = max(abs(v["total_cost"] - v["damped_quadratic"]) / v["damped_quadratic"] for v in L.values())
    audit["processing_order"] = "natural column order (heig-8 GPTQ has act_order_col = False; no permutation)"
    atomic_write_json(os.path.join(OUT, "audit.json" if not a.audit_only else "audit_rerun.json"), {**audit, **rec})
    if a.audit_only:
        tot = {c: {m: sum(v[c][m] for v in build["layers"].values()) for m in build["layers"][keys[0]][c]} for c in CONS}
        tot["static"] = {"recon": sum(v["static"]["recon"] for v in build["layers"].values())}
        build["totals"] = tot
        build["D_top16_mean_selection_overlap"] = float(np.mean([v["D_top16_selection_overlap_with_rerun"] for v in build["layers"].values()]))
        atomic_write_json(os.path.join(OUT, "recon_rerun.json"), {**build, **rec})
        print("audit (re-run):", agg, "| reproduce", audit["all_layers_reproduce_heig8"], "| telescoping", audit["telescoping_max_rel"],
              "| D overlap", build["D_top16_mean_selection_overlap"], flush=True)
        return
    tot = {c: {m: sum(v[c][m] for v in build["layers"].values()) for m in build["layers"][keys[0]][c]} for c in CONS}
    tot["static"] = {"recon": sum(v["static"]["recon"] for v in build["layers"].values())}
    build["totals"] = tot
    atomic_write_json(os.path.join(OUT, "build.json"), {**build, **rec})
    print("audit aggregate:", agg, "| reproduce", audit["all_layers_reproduce_heig8"], "| telescoping", audit["telescoping_max_rel"], flush=True)
    for c in CONS:
        t = tot[c]
        print(f"  {c}: recon A {t['recon_A']:.4g} B {t['recon_B']:.4g} (static {tot['static']['recon']:.4g}); shared {t['shared']:.4g}, "
              f"alternating {t['alternating']:.4g}", flush=True)


if __name__ == "__main__":
    main()
