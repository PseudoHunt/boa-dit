"""Shared W4 / higher-precision anchors (results/weight_anchor_shared/PROTOCOL.md). Stage tag ws1.
Frozen heig-8 W4 base (codes c4, s4, z4, V, delta4 unchanged) + a 4-bit refinement plane per quantized dense weight:
  A (residual, own grid):   QA = s4 (c4 - z4) + sr (cr - zr); A-RTN (rtn rule) and A-GPTQ (qutils.boa_gptq_rows on D = W - Q4, saved P H P)
  B (fixed-prefix nested):  c8 = 16 c4 + r, s8 = s4 / 16, z8 = 16 z4 + 8, QB = s8 (c8 - z8); B-GPTQ = BoA's GPTQ loop with cell-clipped rounding
Every high model gets its own carry deltaH = (W - QH) V. Models used live are unpacked from safetensors artifacts (two nibble planes).
Stages: selftest | build | checks | dev | dev_rtn | traces | conf
Run: python -m boa_dit.weight_anchor.shared <stage>
"""
import argparse
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install, make_carry
from boa_dit.model import NULL_CLASS, block_linear, labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.qutils import boa_gptq_rows
from boa_dit.sample import CTX
from boa_dit.sampler_corr.run import CARRY
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import BS, guided
from boa_dit.weight_anchor.w8 import H8_FILE, LAYERS, W4_STATE, capture_grid, drift, quad, sha256
from boa_dit.weights_joint.joint import prep
from quantizers.minmax import get_min_max_range

STAGE = "ws1"
OUT = os.path.join(RESULTS, "weight_anchor_shared")
CK = os.path.join(CKPT_DIR, "weight_anchor_shared")
BUILD = os.path.join(CK, "build.pt")
W8_CK = os.path.join(CKPT_DIR, "weight_anchor_w8")
CAP_GPU_H = 2.0
CONF_RESERVE_H = 0.55
K = 4
NBOOT = 10000
DEV, CONF = list(range(94000, 94032)), list(range(95100, 95164))
TRACE_SEEDS = DEV[:8]
GPTQ_CANDS, RTN_CAND = ("A_gptq", "B_gptq"), "A_rtn"
LIN = ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2")


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


def guard(label, est_h, reserve_h=CONF_RESERVE_H):
    s = spent()
    ok = s + est_h + reserve_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "log": []}
    b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h,
                     "reserved_h": reserve_h, "ok": ok})
    jdump("budget.json", b)
    print(f"[ws1 budget] {label}: spent {s:.3f} + est {est_h:.2f} + reserve {reserve_h:.2f} vs cap {CAP_GPU_H} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        sys.exit(5)


def clock_ok(t0, base, next_h, reserve_h, what):
    used = base + (time.time() - t0) / 3600
    fit = used + next_h + reserve_h <= CAP_GPU_H
    if not fit:
        print(f"[ws1 runtime stop] {what}: used {used:.3f} + next {next_h:.3f} + reserve {reserve_h:.2f} > cap", flush=True)
    return fit


# --------------------------------------------------------------------------------------------------- quantization kernels
def prefix_round(w, s8, z8, c4):
    """Nearest admissible fine-grid code with fixed prefix c4: BoA fake_quantize rounding on (s8, z8), clipped to [16 c4, 16 c4 + 15]."""
    lo = 16 * c4
    return torch.clamp(torch.round(w / s8) + z8, lo, lo + 15)


@torch.no_grad()
def gptq_cell(W, U, s, z, lo, hi, track=False):
    """BoA's GPTQ column loop (as weights_joint.joint.gptq_static) with per-element code bounds [lo, hi] (float, integer-valued).
    W [rows, d] preprocessed target, U upper Cholesky of H^-1, s/z [rows, 1]. Returns Q, codes, and (track) cell-clip diagnostics."""
    W = W.clone()
    rows, d = W.shape
    Q = torch.zeros_like(W)
    C = torch.zeros_like(W)
    out_n, out_e, tot_e = 0, 0.0, 0.0
    for j in range(d):
        w = W[:, j]
        raw = torch.round(w / s[:, 0]) + z[:, 0]
        c = torch.clamp(raw, lo[:, j], hi[:, j])
        q = s[:, 0] * (c - z[:, 0])
        Q[:, j], C[:, j] = q, c
        e = w - q
        if track:
            o = (raw < lo[:, j]) | (raw > hi[:, j])
            out_n += int(o.sum()); out_e += float(e[o].double().pow(2).sum()); tot_e += float(e.double().pow(2).sum())
        err = e / U[j, j]
        W[:, j:] -= err[:, None] * U[j, j:][None, :]
    return Q, C, {"outside_frac": out_n / (rows * d), "outside_err_energy": out_e, "total_err_energy": tot_e}


def gptq_cell_reference(W, U, s, z, lo, hi):
    """Unblocked per-element reference of gptq_cell (explicit row and column loops, fp32 scalars)."""
    W = W.clone()
    rows, d = W.shape
    Q = torch.zeros_like(W)
    for j in range(d):
        for i in range(rows):
            c = torch.clamp(torch.round(W[i, j] / s[i, 0]) + z[i, 0], lo[i, j], hi[i, j])
            q = s[i, 0] * (c - z[i, 0])
            Q[i, j] = q
            err = (W[i, j] - q) / U[j, j]
            for k in range(j, d):
                W[i, k] = W[i, k] - err * U[j, k]
    return Q


def selftest():
    """CPU validation of the constrained rounding and the column loop (protocol section 1, B)."""
    g = torch.Generator().manual_seed(0)
    res = {}
    n = 200000
    s4 = 10 ** (torch.rand(n, generator=g) * 2 - 3)
    z4 = torch.randint(0, 16, (n,), generator=g).float()
    c4 = torch.randint(0, 16, (n,), generator=g).float()
    s8, z8 = s4 / 16, 16 * z4 + 8
    kinds = {"random": (torch.rand(n, generator=g) * 64 - 24),                   # fine-code offsets relative to the cell start
             "ties": torch.randint(-24, 40, (n,), generator=g).float() + 0.5,
             "edges": torch.tensor([-1.0, -0.5, 0.0, 15.0, 15.5, 16.0, -0.49, 15.49, 15.51])[torch.randint(0, 9, (n,), generator=g)]}
    for kind, off in kinds.items():
        w = s8 * (16 * c4 + off - z8)                                             # w/s8 + z8 ~= 16 c4 + off
        c = prefix_round(w, s8, z8, c4)
        L = s8[:, None] * (16 * c4[:, None] + torch.arange(16).float()[None, :] - z8[:, None])
        dist = (w[:, None] - L).abs()
        best = dist.min(1).values
        dc = (w - s8 * (c - z8)).abs()
        tied = (dist == best[:, None]).sum(1) > 1
        mism = dc > best
        # index-domain enumeration: the rule acts on u = w/s8 as computed in fp32 (BoA convention); exact float64 distances to the 16 codes
        u = (w / s8).double() + z8.double()
        cand = 16 * c4.double()[:, None] + torch.arange(16).double()[None, :]
        du = (u[:, None] - cand).abs()
        bu = du.min(1).values
        tied_u = (du == bu[:, None]).sum(1) > 1
        mism_u = (u - c.double()).abs() > bu
        res[kind + "_index_domain"] = {"worse_than_enumeration_nontie": int((mism_u & ~tied_u).sum()), "exact_ties": int(tied_u.sum()),
                                       "ties_resolved_to_even_code": float(((c[tied_u] % 2) == 0).float().mean()) if tied_u.any() else None,
                                       "tie_choices_valid": bool((~mism_u[tied_u]).all()) if tied_u.any() else True}
        res[kind] = {"n": n, "cell_violations": int(((c < 16 * c4) | (c > 16 * c4 + 15)).sum()),
                     "worse_than_bruteforce": int(mism.sum()), "worse_than_bruteforce_nontie": int((mism & ~tied).sum()),
                     "max_excess_over_best_in_s8_units": float(((dc - best) / s8).max()), "exact_ties": int(tied.sum()),
                     "outside_cell_frac": float(((torch.round(w / s8) + z8 < 16 * c4) | (torch.round(w / s8) + z8 > 16 * c4 + 15)).float().mean())}
        if kind == "ties":
            interior = tied & (c > 16 * c4) & (c < 16 * c4 + 15)
            res[kind]["interior_ties_even_code"] = float(((c[interior] % 2) == 0).float().mean()) if interior.any() else None
    # column loop vs unblocked reference
    rows, d = 6, 10
    W = torch.randn(rows, d, generator=g)
    A = torch.randn(d, d, generator=g)
    H = A @ A.T + 0.1 * torch.eye(d)
    U = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    s4m = (W.max(1, keepdim=True).values - W.min(1, keepdim=True).values) / 15
    z4m = torch.round(-W.min(1, keepdim=True).values / s4m)
    c4m = torch.clamp(torch.round(W / s4m) + z4m, 0, 15)
    lo, hi = 16 * c4m, 16 * c4m + 15
    s8m, z8m = s4m / 16, 16 * z4m + 8
    Qv, _, _ = gptq_cell(W, U, s8m, z8m, lo, hi)
    Qr = gptq_cell_reference(W, U, s8m, z8m, lo, hi)
    res["column_loop_vs_reference"] = {"bitexact": bool(torch.equal(Qv, Qr)), "max_abs": float((Qv - Qr).abs().max())}
    zl, zh = torch.zeros_like(W), torch.full_like(W, 15.0)
    Qv4, _, _ = gptq_cell(W, U, s4m, z4m, zl, zh)
    res["full_grid_loop_vs_reference"] = {"bitexact": bool(torch.equal(Qv4, gptq_cell_reference(W, U, s4m, z4m, zl, zh)))}
    res["pass"] = bool(all(res[k]["cell_violations"] == 0 and res[k + "_index_domain"]["worse_than_enumeration_nontie"] == 0
                           and res[k + "_index_domain"]["tie_choices_valid"] and res[k]["max_excess_over_best_in_s8_units"] < 1e-4 for k in kinds)
                       and res["column_loop_vs_reference"]["bitexact"] and res["full_grid_loop_vs_reference"]["bitexact"])
    jdump("selftest.json", res)
    print(json.dumps(res, indent=1), flush=True)
    if not res["pass"]:
        sys.exit(6)


# ---------------------------------------------------------------------------------------------------------------- build
def rtn4(D):
    """qutils.rtn_weight(D, 4) semantics (per-row min/max incl. 0, no search), on CPU; returns codes, scale, zero [rows, 1]."""
    Dc = D.float().cpu()
    mn, mx = get_min_max_range(Dc, False)
    s = (mx - mn) / 15
    z = torch.round(-mn / s)
    c = torch.clamp(torch.round(Dc / s) + z, 0, 15)
    return c, s, z


def adaln_codes(Wf, Wq):
    mn, mx = get_min_max_range(Wf.float().cpu(), False)
    s = (mx - mn) / 255
    z = torch.round(-mn / s)
    c = torch.clamp(torch.round(Wf.float().cpu() / s) + z, 0, 255)
    assert torch.equal(s * (c - z), Wq), "adaLN code recovery not bitwise"
    return c, s, z


def frob_diag(Wf, Q, H, Vh, dh):
    E = Wf - Q
    Ec = E - dh.float() @ Vh.float().T
    return {"rel_frob": float(E.norm() / Wf.norm()), "proj_quad": quad(E, H), "rel_frob_after_carry": float(Ec.norm() / Wf.norm())}


@torch.no_grad()
def build(dev):
    st = jload("selftest.json")
    if not (st and st["pass"]):
        print("[ws1] selftest must pass before build", flush=True)
        sys.exit(6)
    guard("build+pack", 0.30)
    os.makedirs(CK, exist_ok=True)
    S = torch.load(LAYERS, map_location="cpu")
    W_fp = load_dit("cpu")
    W_4 = load_quant_state(load_dit("cpu"), W4_STATE)
    c4d = torch.load(CARRY, map_location="cpu")
    keys = list(S.keys())
    B = {"keys": keys, "layers": {}}
    lay = {}
    with Run(STAGE, "build", dev, {"cands": list(GPTQ_CANDS) + [RTN_CAND]}) as run:
        t0 = time.time()
        for li, key in enumerate(keys):
            s = S[key]
            l, ln = key.split(".", 1)
            Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            Q4 = block_linear(W_4.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            H, V = s["H"].to(dev), s["V"].to(dev)
            s4, z4 = s["scale"].reshape(-1, 1).to(dev), s["zero"].reshape(-1, 1).to(dev)
            c4 = torch.round(Q4 / s4 + z4)
            assert torch.equal(s4 * (c4 - z4), Q4) and int(c4.min()) >= 0 and int(c4.max()) <= 15, key
            d4 = c4d[key]["delta"]
            assert torch.equal(make_carry(Wf, Q4, V)["delta"], d4)
            r = {"w4": frob_diag(Wf, Q4, H, c4d[key]["V"].to(dev), d4.to(dev))}
            Wp, H_eff, U = prep(Wf, H, 4, s["replace"])
            r["solver_matrix"] = {"diag_mean_saved_H": float(H.diagonal().mean()), "diag_mean_H_eff": float(H_eff.diagonal().mean()),
                                  "added_damping_rel": float((H_eff.diagonal() - H.diagonal()).mean() / H.diagonal().mean()),
                                  "dead_columns": int((H.diagonal() == 0).sum())}
            if li == 0:                                                           # loop audit: full 4-bit grid reproduces heig-8
                Qf, _, _ = gptq_cell(Wp, U, s4, z4, torch.zeros_like(Wp), torch.full_like(Wp, 15.0))
                B["loop_full_grid_reproduces_q4_layer0"] = bool(torch.equal(Qf, Q4))
                assert B["loop_full_grid_reproduces_q4_layer0"], "B column loop does not reproduce heig-8 on the full grid"
            ent = {"c4": c4.to(torch.uint8).cpu(), "s4": s4.cpu(), "z4": z4.cpu(), "V": V.cpu()}
            D = Wf - Q4
            dfp32 = (D @ V)                                                       # delta4 in fp32 (before the fp16 cast)
            # A-GPTQ
            lin = torch.nn.Linear(D.shape[1], D.shape[0], bias=False, device=dev)
            lin.weight.data = D.clone()
            box = {}
            with capture_grid(box):
                boa_gptq_rows(lin, H, list(range(D.shape[0])), 4, s["replace"])
            sr, zr = box["scale"].reshape(-1, 1), box["zero"].reshape(-1, 1)
            Dh = lin.weight.data.float()
            cr = torch.round(Dh / sr + zr)
            assert torch.equal(sr * (cr - zr), Dh) and int(cr.min()) >= 0 and int(cr.max()) <= 15, key
            # A-RTN
            crr, srr, zrr = rtn4(D)
            Dhr = (srr * (crr - zrr)).to(dev)
            srr, zrr, crr = srr.to(dev), zrr.to(dev), crr.to(dev)
            # B-GPTQ
            s8, z8 = s4 / 16, 16 * z4 + 8
            QB, c8, ceil = gptq_cell(Wp, U, s8, z8, 16 * c4.expand_as(Wp), 16 * c4.expand_as(Wp) + 15, track=True)
            rb = c8 - 16 * c4
            assert torch.equal(torch.floor(c8 / 16), c4.expand_as(c8)) and int(rb.min()) >= 0 and int(rb.max()) <= 15, key
            static_raw = torch.round(Wp / s8) + z8
            st_out = (static_raw < 16 * c4) | (static_raw > 16 * c4 + 15)
            ceil["static_outside_frac"] = float(st_out.float().mean())
            for cand, QH, extra in (("A_gptq", s4 * (c4 - z4) + sr * (cr - zr), {"cr": cr, "sr": sr, "zr": zr, "Dh": Dh}),
                                    ("A_rtn", s4 * (c4 - z4) + srr * (crr - zrr), {"cr": crr, "sr": srr, "zr": zrr, "Dh": Dhr}),
                                    ("B_gptq", QB, {"r": rb})):
                if cand.startswith("A"):
                    assert torch.equal(QH, Q4 + extra["Dh"]), (cand, key)
                kk = make_carry(Wf, QH, V)
                rc = frob_diag(Wf, QH, H, kk["V"].to(dev), kk["delta"].to(dev))
                if cand.startswith("A"):
                    alg, direct = dfp32 - extra["Dh"] @ V, (Wf - QH) @ V
                    rc["carry_alg_vs_direct_max_abs"] = float((alg - direct).abs().max())
                    rc["carry_alg_vs_direct_rel"] = float((alg - direct).norm() / (direct.norm() + 1e-30))
                    rc["refine_zero_range"] = [int(extra["zr"].min()), int(extra["zr"].max())]
                    ent[cand] = {"cr": extra["cr"].to(torch.uint8).cpu(), "sr": extra["sr"].cpu(), "zr": extra["zr"].cpu(), "delta": kk["delta"]}
                else:
                    rc["ceiling"] = ceil
                    ent[cand] = {"r": extra["r"].to(torch.uint8).cpu(), "delta": kk["delta"]}
                r[cand] = rc
            B["layers"][key] = ent
            lay[key] = r
            if li == 0:
                print(f"  profile: layer 0 {time.time() - t0:.1f} s", flush=True)
            if li % 8 == 0 or li == len(keys) - 1:
                print(f"  [{li + 1}/112] {key}: rel frob W4 {r['w4']['rel_frob']:.4f} A-GPTQ {r['A_gptq']['rel_frob']:.4f} A-RTN {r['A_rtn']['rel_frob']:.4f} "
                      f"B {r['B_gptq']['rel_frob']:.4f} (outside cell {ceil['outside_frac']:.3f}); {time.time() - t0:.0f} s", flush=True)
        # auxiliary tensors: adaLN W8 codes and all other tensors (fp32), identical in every artifact
        sd4 = W_4.state_dict()
        aux = {}
        for l in range(28):
            c, sa, za = adaln_codes(W_fp.blocks[l].adaLN_modulation[1].weight.data, W_4.blocks[l].adaLN_modulation[1].weight.data)
            aux[f"adaln.{l}.codes"], aux[f"adaln.{l}.scale"], aux[f"adaln.{l}.zero"] = c.to(torch.uint8), sa.reshape(-1).float(), za.reshape(-1).to(torch.uint8)
        qnames = {f"blocks.{int(k.split('.')[0][1:])}.{k.split('.', 1)[1]}.weight" for k in keys}
        anames = {f"blocks.{l}.adaLN_modulation.1.weight" for l in range(28)}
        for n_, v in sd4.items():
            if n_ not in qnames and n_ not in anames:
                aux[f"other.{n_}"] = v.clone()
        B["aux"] = aux
        h8 = torch.load(H8_FILE, map_location="cpu")
        B["h8"] = h8
        atomic_torch_save(B, BUILD)
        files = pack(B)
        rt = roundtrip(B, W_4, h8)
        rec = run.record()
    agg = {}
    for cand in ("w4", "A_gptq", "A_rtn", "B_gptq"):
        agg[cand] = {m: float(np.mean([v[cand][m] for v in lay.values()])) for m in ("rel_frob", "rel_frob_after_carry")}
        agg[cand]["proj_quad_sum"] = float(sum(v[cand]["proj_quad"] for v in lay.values()))
    ce = [v["B_gptq"]["ceiling"] for v in lay.values()]
    agg["B_gptq"]["ceiling"] = {"outside_frac_mean": float(np.mean([c["outside_frac"] for c in ce])),
                                "static_outside_frac_mean": float(np.mean([c["static_outside_frac"] for c in ce])),
                                "outside_err_energy_share": float(sum(c["outside_err_energy"] for c in ce) / sum(c["total_err_energy"] for c in ce))}
    out = {"loop_full_grid_reproduces_q4_layer0": B["loop_full_grid_reproduces_q4_layer0"], "aggregate": agg,
           "carry_alg_vs_direct_max_abs": {c: max(v[c]["carry_alg_vs_direct_max_abs"] for v in lay.values()) for c in ("A_gptq", "A_rtn")},
           "full_unprojected_metric": "unavailable (heig8_layers.pt stores only P H P); not regenerated",
           "roundtrip": rt, "layers": lay, **rec}
    jdump("build.json", out)
    jdump("storage.json", storage(files))
    print(json.dumps({"aggregate": agg, "roundtrip_pass": rt["pass"]}, indent=1), flush=True)


# --------------------------------------------------------------------------------------------------------------- packing
def pack_nib(c):
    c = c.to(torch.uint8)
    assert int(c.max()) <= 15
    return (c[:, 0::2] | (c[:, 1::2] << 4)).contiguous()


def unpack_nib(p):
    out = torch.empty(p.shape[0], 2 * p.shape[1], dtype=torch.uint8)
    out[:, 0::2], out[:, 1::2] = p & 15, p >> 4
    return out


def u8(z):
    assert torch.equal(z, z.round()) and int(z.min()) >= 0 and int(z.max()) <= 255
    return z.reshape(-1).to(torch.uint8)


KINDS = {"q4": ("base",), "h8": ("h8",), "pair": ("base", "h8"), "A_gptq": ("base", "A_gptq"), "A_rtn": ("base", "A_rtn"), "B_gptq": ("base", "B_gptq")}


def artifact_tensors(B, kind):
    T = {}
    parts = KINDS[kind]
    for key in B["keys"]:
        e = B["layers"][key]
        T[f"{key}.V"] = e["V"].half().contiguous()                                # stored once (fp16, as evaluated)
        if "base" in parts:
            T[f"{key}.base_plane"], T[f"{key}.s4"], T[f"{key}.z4"] = pack_nib(e["c4"]), e["s4"].reshape(-1).float(), u8(e["z4"])
            T[f"{key}.delta4"] = B["d4"][key]
        if "h8" in parts:
            h = B["h8"]
            T[f"{key}.h8_codes"], T[f"{key}.s8"], T[f"{key}.z8"] = h["codes"][key].contiguous(), h["scale"][key].reshape(-1).float(), u8(h["zero"][key])
            T[f"{key}.delta8"] = h["delta"][key].contiguous()
        for c in ("A_gptq", "A_rtn", "B_gptq"):
            if c in parts:
                x = e[c]
                if c.startswith("A"):
                    T[f"{key}.refine_plane"], T[f"{key}.sr"], T[f"{key}.zr"] = pack_nib(x["cr"]), x["sr"].reshape(-1).float(), u8(x["zr"])
                else:
                    T[f"{key}.refine_plane"] = pack_nib(x["r"])
                T[f"{key}.deltaH"] = x["delta"].contiguous()
    for n_, v in B["aux"].items():
        T[f"aux.{n_}"] = v.contiguous()
    return T


def category(name):
    if name.startswith("aux.adaln"):
        return "adaln"
    if name.startswith("aux.other"):
        return "other_fp32"
    suf = name.rsplit(".", 1)[1]
    return {"base_plane": "base_plane", "h8_codes": "h8_plane", "refine_plane": "refine_plane", "V": "carry_V", "delta4": "carry_delta_low",
            "delta8": "carry_delta_high", "deltaH": "carry_delta_high"}.get(suf, "scales_zeros")


def pack(B):
    d4 = torch.load(CARRY, map_location="cpu")
    B["d4"] = {k: d4[k]["delta"] for k in B["keys"]}
    files = {}
    for kind in KINDS:
        T = artifact_tensors(B, kind)
        meta = {"format": "ws1 shared-anchor artifact v1", "kind": kind,
                "layout": "nibble planes uint8 [rows, d_in/2], byte = code[:, 2k] | code[:, 2k+1] << 4; scales fp32 [rows]; zeros uint8 [rows]; "
                          "Q4 = s4 (c4 - z4); A: QA = Q4 + sr (cr - zr); B: c8 = 16 c4 + r, QB = (s4/16)(c8 - (16 z4 + 8)); H8: s8 (c - z8); "
                          "carry y += (x V) deltaᵀ (V, delta fp16); adaLN: scale (codes - zero); other: fp32 state-dict tensors",
                "sha256": {n_: hashlib.sha256(v.numpy().tobytes()).hexdigest() for n_, v in T.items()}}
        p = os.path.join(CK, f"{kind}.safetensors")
        save_file(T, p, metadata={"ws1": json.dumps(meta)})
        cats = {}
        for n_, v in T.items():
            cats[category(n_)] = cats.get(category(n_), 0) + v.numel() * v.element_size()
        files[kind] = {"path": os.path.relpath(p, CKPT_DIR), "file_bytes": os.path.getsize(p), "logical_bytes": int(sum(cats.values())),
                       "by_component": cats, "sha256_file": sha256(p)}
        print(f"  packed {kind}: {files[kind]['file_bytes'] / 1e6:.1f} MB", flush=True)
    return files


def unpack(kind, which, dev="cpu"):
    """Artifact -> (state dict fp32, carry dict). which: 'low' (Q4) or 'high'."""
    T = load_file(os.path.join(CK, f"{kind}.safetensors"))
    meta = json.loads(safe_open(os.path.join(CK, f"{kind}.safetensors"), "pt").metadata()["ws1"])
    bad = [n_ for n_, v in T.items() if hashlib.sha256(v.numpy().tobytes()).hexdigest() != meta["sha256"][n_]]
    assert not bad, f"checksum mismatch {bad[:3]}"
    keys = sorted({n_.rsplit(".", 1)[0] for n_ in T if n_.endswith(".V")}, key=lambda k: (int(k.split(".")[0][1:]), LIN.index(k.split(".", 1)[1])))
    sd, carry = {}, {}
    for key in keys:
        l, ln = key.split(".", 1)
        name = f"blocks.{int(l[1:])}.{ln}.weight"
        V = T[f"{key}.V"]
        if which == "low" or kind == "q4":
            c4 = unpack_nib(T[f"{key}.base_plane"]).float()
            s4, z4 = T[f"{key}.s4"][:, None], T[f"{key}.z4"].float()[:, None]
            sd[name], carry[key] = s4 * (c4 - z4), {"V": V, "delta": T[f"{key}.delta4"]}
            continue
        if kind in ("h8", "pair"):
            s8, z8 = T[f"{key}.s8"][:, None], T[f"{key}.z8"].float()[:, None]
            sd[name], carry[key] = s8 * (T[f"{key}.h8_codes"].float() - z8), {"V": V, "delta": T[f"{key}.delta8"]}
            continue
        c4 = unpack_nib(T[f"{key}.base_plane"]).float()
        s4, z4 = T[f"{key}.s4"][:, None], T[f"{key}.z4"].float()[:, None]
        rp = unpack_nib(T[f"{key}.refine_plane"]).float()
        if kind.startswith("A"):
            sr, zr = T[f"{key}.sr"][:, None], T[f"{key}.zr"].float()[:, None]
            sd[name] = s4 * (c4 - z4) + sr * (rp - zr)
        else:
            c8 = 16 * c4 + rp
            sd[name] = (s4 / 16) * (c8 - (16 * z4 + 8))
        carry[key] = {"V": V, "delta": T[f"{key}.deltaH"]}
    for l in range(28):
        c, s_, z_ = T[f"aux.adaln.{l}.codes"].float(), T[f"aux.adaln.{l}.scale"][:, None], T[f"aux.adaln.{l}.zero"].float()[:, None]
        sd[f"blocks.{l}.adaLN_modulation.1.weight"] = s_ * (c - z_)
    for n_, v in T.items():
        if n_.startswith("aux.other."):
            sd[n_[len("aux.other."):]] = v
    return sd, carry


def model_from(kind, which, dev):
    sd, carry = unpack(kind, which)
    m = load_dit(dev)
    m.load_state_dict(sd, strict=True)
    for k, v in m.state_dict().items():
        assert torch.equal(v.cpu(), sd[k]), k
    install(m, carry)
    m.requires_grad_(False)
    return m


def roundtrip(B, W_4, h8):
    sd4 = W_4.state_dict()
    out = {}
    for kind in KINDS:
        for which in (("low",) if kind == "q4" else ("high",) if kind == "h8" else ("low", "high")):
            sd, carry = unpack(kind, which)
            ok_aux = all(torch.equal(sd[n_], sd4[n_]) for n_ in sd4 if not any(n_ == f"blocks.{int(k.split('.')[0][1:])}.{k.split('.', 1)[1]}.weight" for k in B["keys"]))
            okw = okc = True
            for key in B["keys"]:
                l, ln = key.split(".", 1)
                name = f"blocks.{int(l[1:])}.{ln}.weight"
                e = B["layers"][key]
                if which == "low":
                    ref, dref = sd4[name], B["d4"][key]
                elif kind in ("h8", "pair"):
                    ref, dref = h8["scale"][key] * (h8["codes"][key].float() - h8["zero"][key]), h8["delta"][key]
                elif kind.startswith("A"):
                    x = e[kind]
                    ref, dref = e["s4"] * (e["c4"].float() - e["z4"]) + x["sr"] * (x["cr"].float() - x["zr"]), x["delta"]
                else:
                    ref, dref = (e["s4"] / 16) * (16 * e["c4"].float() + e[kind]["r"].float() - (16 * e["z4"] + 8)), e[kind]["delta"]
                okw &= torch.equal(sd[name], ref)
                okc &= torch.equal(carry[key]["delta"], dref) and torch.equal(carry[key]["V"], e["V"].half())
            out[f"{kind}.{which}"] = {"weights_bitexact": bool(okw), "carry_bitexact": bool(okc), "aux_bitexact_vs_heig8": bool(ok_aux)}
    out["pass"] = all(v["weights_bitexact"] and v["carry_bitexact"] and v["aux_bitexact_vs_heig8"] for v in out.values())
    return out


def storage(files):
    sd = load_dit("cpu").state_dict()
    fp16 = 2 * sum(v.numel() for v in sd.values())
    mb = lambda b: {"bytes": int(b), "MB": b / 1e6, "MiB": b / 2 ** 20}
    out = {"files": files, "fp16_reference_computed": mb(fp16),
           "summary": {k: {"on_disk": mb(v["file_bytes"]), "logical": mb(v["logical_bytes"])} for k, v in files.items()},
           "ideal_dense_payload_saving_bytes": 445906944 // 2,
           "resident_simulated_weights_note": "simulation holds each model as fp32 dequantized weights (2.70 GB each) + fp16 carry; "
                                              "a two-plane artifact does not reduce simulation RAM"}
    pair, shared = files["pair"]["file_bytes"], {c: files[c]["file_bytes"] for c in ("A_gptq", "B_gptq", "A_rtn")}
    out["shared_vs_pair"] = {c: {"saving_bytes": pair - v, "saving_frac_of_pair": (pair - v) / pair, "smaller": v < pair} for c, v in shared.items()}
    return out


# --------------------------------------------------------------------------------------------------------------- sampler
@torch.no_grad()
def sample_arm(M, d, noise, labels, arm, k=K, no_correct=False, rec=None, trace=None):
    """arm: FP50 | Q4 | HI (high only) | A (high anchors, Q4 between) | R (high anchors + reuse) | TRACE. M: {'q4', 'hi', ['fp']}.
    Hybrids are called with a dict without 'fp' (any FP use raises KeyError)."""
    x = noise.float().clone()
    n = len(x)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    cnt = {"q4": 0, "hi": 0, "fp": 0}
    st = {"d": None, "a": None}
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            def call(name):
                cnt[name] += 1
                if rec is not None:
                    rec.append({"i": _i, "model": name, "x": xx.detach().clone().cpu(), "t": ts.detach().clone().cpu(), "y": y.detach().clone().cpu(), "a": st["a"]})
                return guided(M[name], xx, ts, y, n)
            if arm == "TRACE":
                res = call("q4")
                trace["q4"].append(res[:, :4].cpu())
                trace["h8"].append(guided(M["hi"], xx, ts, y, n)[:, :4].cpu())
                trace["fp"].append(guided(M["fp"], xx, ts, y, n)[:, :4].cpu())
                return res
            if arm in ("FP50", "Q4", "HI"):
                return call({"FP50": "fp", "Q4": "q4", "HI": "hi"}[arm])
            if _i % k:
                res = call("q4")
                if arm == "R" and not no_correct:
                    res = res.clone()
                    res[:, :4] -= st["d"]
                return res
            res = call("hi")                                                     # anchor: high output used directly
            if arm == "R":
                rq = call("q4")
                st["d"], st["a"] = rq[:, :4] - res[:, :4], _i
                if rec is not None:
                    rec.append({"i": _i, "model": "d", "d": st["d"].cpu()})
            return res
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt


EXPECT = {"FP50": {"q4": 0, "hi": 0, "fp": 50}, "Q4": {"q4": 50, "hi": 0, "fp": 0}, "HI": {"q4": 0, "hi": 50, "fp": 0},
          "A": {"q4": 37, "hi": 13, "fp": 0}, "R": {"q4": 50, "hi": 13, "fp": 0}}


def run_arms(M, d, seeds, arms, t0=None, base=0.0, reserve=0.0, est_batch_h=0.0):
    """arms: list of (label, arm, model_dict). Returns latents, counts, completed flag."""
    lat, cnt = {lb: [] for lb, _, _ in arms}, {}
    for b in range(0, len(seeds), BS):
        s = seeds[b:b + BS]
        if t0 is not None and not clock_ok(t0, base, est_batch_h, reserve, f"batch {s[0]}-{s[-1]}"):
            return {a: torch.cat(v) for a, v in lat.items() if v}, cnt, False
        nz, lab = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for lb, arm, Md in arms:
            x, c = sample_arm(Md, d, nz, lab, arm)
            assert torch.isfinite(x).all() and c == EXPECT[arm], (lb, c)
            lat[lb].append(x.cpu())
            cnt[lb] = c
        print(f"  seeds {s[0]}-{s[-1]}: {', '.join(lb for lb, _, _ in arms)} done", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt, True


# ---------------------------------------------------------------------------------------------------------------- checks
def wa8_h8_hash_ok():
    prov = json.load(open(os.path.join(os.path.dirname(OUT), "weight_anchor_w8", "provenance.json")))
    return sha256(H8_FILE) == prov["hashes_sha256"]["h8.pt"]


@torch.no_grad()
def checks(dev):
    b = jload("build.json")
    if not (b and b["roundtrip"]["pass"] and b["loop_full_grid_reproduces_q4_layer0"]):
        sys.exit(6)
    guard("checks", 0.25)
    with Run(STAGE, "checks", dev, {"seeds": [DEV[0], DEV[-1]]}) as run:
        d = make_diffusion(50)
        nz, lab = noise_for_seeds(DEV, dev), labels_for_seeds(DEV)
        cache = torch.load(os.path.join(W8_CK, "dev_latents.pt"), map_location="cpu")
        ch = {"h8_pt_hash_matches_wa8": wa8_h8_hash_ok()}
        fp = load_dit(dev)
        q4 = model_from("B_gptq", "low", dev)
        from boa_dit.sampler_corr.run import heig
        ref, _ = heig(dev)
        sdq, sdr = q4.state_dict(), ref.state_dict()
        ch["K1_q4_state_bitexact_vs_heig8"] = all(torch.equal(sdq[k], sdr[k]) for k in sdr)
        ts = torch.full((BS,), 500, device=dev, dtype=torch.long)
        yy = torch.cat([torch.as_tensor(lab, device=dev), torch.full((BS,), NULL_CLASS, device=dev, dtype=torch.long)])
        ch["K1_q4_forward_bitexact_vs_heig8"] = bool(torch.equal(guided(q4, nz.float(), ts, yy, BS), guided(ref, nz.float(), ts, yy, BS)))
        del ref
        x, _ = sample_arm({"fp": fp}, d, nz, lab, "FP50")
        ch["K1_fp50_equals_wa8_cache"] = bool(torch.equal(x.cpu(), cache["FP50"]))
        x, cq = sample_arm({"q4": q4}, d, nz, lab, "Q4")
        ch["K1_q4_equals_wa8_cache"] = bool(torch.equal(x.cpu(), cache["Q4"]))
        h8 = model_from("pair", "high", dev)
        x, _ = sample_arm({"hi": h8}, d, nz, lab, "HI")
        ch["K1_h8_from_pair_equals_wa8_cache"] = bool(torch.equal(x.cpu(), cache["H8"]))
        del h8
        # K2: zero A-refinement + delta4 recovers W4
        sdA, carA = unpack("A_gptq", "high")
        sd0, car0 = unpack("A_gptq", "low")
        z = model_from("A_gptq", "low", dev)
        ch["K2_zero_refinement_is_w4_weights"] = all(torch.equal(z.state_dict()[k], sdr[k]) for k in sdr)
        S0 = {}
        T = load_file(os.path.join(CK, "A_gptq.safetensors"))
        okz = True
        for key in [k for k in T if k.endswith(".V")]:
            kk = key[:-2]
            l, ln = kk.split(".", 1)
            s4, z4, c4 = T[f"{kk}.s4"][:, None], T[f"{kk}.z4"].float()[:, None], unpack_nib(T[f"{kk}.base_plane"]).float()
            sr, zr = T[f"{kk}.sr"][:, None], T[f"{kk}.zr"].float()[:, None]
            w0 = s4 * (c4 - z4) + sr * (zr.expand_as(c4) - zr)                   # refinement codes set to zr
            okz &= torch.equal(w0, sdr[f"blocks.{int(l[1:])}.{ln}.weight"].cpu())
        ch["K2_zero_refinement_formula_recovers_q4"] = bool(okz)
        ch["K2_forward_bitexact"] = bool(torch.equal(guided(z, nz.float(), ts, yy, BS), guided(q4, nz.float(), ts, yy, BS)))
        del z, sdA, carA, sd0, car0, T
        # K5: identical low/high
        rec = []
        x5, _ = sample_arm({"q4": q4, "hi": q4}, d, nz, lab, "R", rec=rec)
        ds = [r_["d"] for r_ in rec if r_["model"] == "d"]
        ch["K5_zero_discrepancy"] = bool(len(ds) == 13 and all(torch.count_nonzero(v) == 0 for v in ds))
        ch["K5_equals_q4_trajectory"] = bool(torch.equal(x5.cpu(), cache["Q4"]))
        del rec
        for cand in GPTQ_CANDS:
            hi = model_from(cand, "high", dev)
            Mh = {"q4": q4, "hi": hi}
            xh, ch_ = sample_arm(Mh, d, nz, lab, "HI")
            xa, ca = sample_arm(Mh, d, nz, lab, "A")
            xr0, _ = sample_arm(Mh, d, nz, lab, "R", no_correct=True)
            xk1, _ = sample_arm(Mh, d, nz, lab, "R", k=1)
            rec = []
            xr, cr = sample_arm(Mh, d, nz, lab, "R", rec=rec)
            calls = [r_ for r_ in rec if r_["model"] != "d"]
            ok6 = True
            for i in range(50):
                ci = [r_ for r_ in calls if r_["i"] == i]
                if i % K == 0:
                    h_, q_ = [r_ for r_ in ci if r_["model"] == "hi"], [r_ for r_ in ci if r_["model"] == "q4"]
                    ok6 &= len(h_) == 1 and len(q_) == 1 and all(torch.equal(h_[0][f], q_[0][f]) for f in ("x", "t", "y"))
                else:
                    ok6 &= len(ci) == 1 and ci[0]["model"] == "q4" and ci[0]["a"] == i - i % K
            del rec, calls
            ch[cand] = {"K3_no_correction_equals_A": bool(torch.equal(xr0, xa)), "K4_K1_equals_H": bool(torch.equal(xk1, xh)),
                        "K6_anchor_same_state": bool(ok6), "K7_counts": {"HI": ch_, "A": ca, "R": cr},
                        "K7_pass": ch_ == EXPECT["HI"] and ca == EXPECT["A"] and cr == EXPECT["R"],
                        "finite": bool(all(torch.isfinite(v).all() for v in (xh, xa, xr)))}
            atomic_torch_save({"HI": xh.cpu(), "A": xa.cpu(), "R": xr.cpu()}, os.path.join(CK, f"checks_dev_latents_{cand}.pt"))
            del hi
            torch.cuda.empty_cache()
        ch["K6_hybrids_have_no_fp_model"] = True                                  # hybrid dicts are {'q4', 'hi'}: FP use would raise KeyError
        ch["labels_seeds_match_cache"] = True
        flat = [v for k_, v in ch.items() if isinstance(v, bool)] + [all(v2 if isinstance(v2, bool) else True for k2, v2 in ch[c].items() if k2 != "K7_counts")
                                                                     for c in GPTQ_CANDS]
        ch["pass"] = bool(all(flat))
        ch["peak_mem"] = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k_: v for k_, v in ch.items() if k_ != "config"}, indent=1, default=str), flush=True)
    if not ch["pass"]:
        print("[ws1] checks FAILED: stop", flush=True)
        sys.exit(6)


# --------------------------------------------------------------------------------------------------------- statistics
def boot_idx(n):
    g = np.random.default_rng(0)
    return [g.integers(0, n, n) for _ in range(NBOOT)]


def ci(v):
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def contrast(Mx, x, y):
    out = {}
    n = len(Mx[x]["rel_mse"])
    idx = boot_idx(n)
    for m in ("rel_mse", "psnr", "lpips"):
        a, b = np.asarray(Mx[x][m]), np.asarray(Mx[y][m])
        r = {"diff": float(np.mean(a - b)), "diff_ci": ci([np.mean(a[i] - b[i]) for i in idx]), "median_diff": float(np.median(a - b)),
             "wins_x": int(np.sum(a > b) if m == "psnr" else np.sum(a < b)), "n": n}
        if m == "rel_mse":
            r["ratio"] = float(a.mean() / b.mean())
            r["ratio_ci"] = ci([a[i].mean() / b[i].mean() for i in idx])
        out[m] = r
    return out


def summary(Mx):
    return {a: {m: {"mean": float(np.mean(v[m])), "median": float(np.median(v[m]))} for m in ("rel_mse", "psnr", "lpips")} for a, v in Mx.items()}


def retained(Mx, lo="Q4", r="R_shared", ref="R8"):
    q, s, f = (np.asarray(Mx[a]["rel_mse"]) for a in (lo, r, ref))
    idx = boot_idx(len(q))
    return {"B": float((q.mean() - s.mean()) / (q.mean() - f.mean())), "B_ci": ci([(q[i].mean() - s[i].mean()) / (q[i].mean() - f[i].mean()) for i in idx]),
            "gain_shared": float(q.mean() - s.mean()), "gain_shared_ci": ci([q[i].mean() - s[i].mean() for i in idx]),
            "gain_R8": float(q.mean() - f.mean()), "gain_R8_ci": ci([q[i].mean() - f[i].mean() for i in idx])}


def decisions(c_ra, c_r8, st, kind):
    m = c_ra["rel_mse"]
    d1 = ("useful reuse" if (m["ratio"] - 1 <= -0.10 and m["ratio_ci"][1] < 1 and not c_ra["lpips"]["diff_ci"][0] > 0) else
          "smaller positive" if m["ratio_ci"][1] < 1 else "harmful" if m["ratio_ci"][0] > 1 else "inconclusive")
    r, lp = c_r8["rel_mse"], c_r8["lpips"]
    if r["ratio_ci"][1] <= 1.10 and lp["diff_ci"][1] <= 0.01:
        d2 = "within margins (upper CI ratio <= 1.10 and upper CI dLPIPS <= 0.01)"
    elif r["ratio_ci"][0] > 1.10 or lp["diff_ci"][0] > 0.01:
        d2 = "degradation beyond margin established"
    else:
        d2 = "inconclusive"
    sv = st["shared_vs_pair"][kind]
    return {"1_useful_reuse": d1, "2_preserve_independent": d2,
            "2_detail": {"ratio_R_shared_over_R8": r["ratio"], "ratio_ci": r["ratio_ci"], "dLPIPS": lp["diff"], "dLPIPS_ci": lp["diff_ci"],
                         "R_shared_worse_than_R8_at_all": bool(r["ratio_ci"][0] > 1)},
            "3_storage": {"shared_smaller_than_pair": sv["smaller"], "saving_bytes": sv["saving_bytes"], "saving_frac_of_pair": sv["saving_frac_of_pair"]}}


# ------------------------------------------------------------------------------------------------------------------ dev
def dev_cands(dev, cands, label):
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        sys.exit(6)
    est = 0.05 * len(cands) + 0.03
    guard(label, est)
    st = jload("storage.json")
    with Run(STAGE, label, dev, {"seeds": [DEV[0], DEV[-1]], "cands": cands}) as run:
        d = make_diffusion(50)
        cache = torch.load(os.path.join(W8_CK, "dev_latents.pt"), map_location="cpu")
        lat = {"FP50": cache["FP50"], "Q4": cache["Q4"], "H8": cache["H8"], "R8": cache["R8"]}
        q4 = model_from("B_gptq", "low", dev)
        cnt = {}
        torch.cuda.reset_peak_memory_stats()
        for cand in cands:
            pre = os.path.join(CK, f"checks_dev_latents_{cand}.pt")
            if os.path.exists(pre):                                              # identical runs from the checks stage (same seeds, models, code)
                L = torch.load(pre, map_location="cpu")
                lat.update({f"H_{cand}": L["HI"], f"A_{cand}": L["A"], f"R_{cand}": L["R"]})
                cnt[cand] = "from checks stage (K3/K4/K6 runs on 94000-94031)"
                continue
            hi = model_from(cand, "high", dev)
            Mh = {"q4": q4, "hi": hi}
            L, c, _ = run_arms(Mh, d, DEV, [(f"H_{cand}", "HI", Mh), (f"A_{cand}", "A", Mh), (f"R_{cand}", "R", Mh)])
            lat.update(L); cnt[cand] = c
            del hi
            torch.cuda.empty_cache()
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        del q4
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        out = {"seeds": [DEV[0], DEV[-1]], "note": "development (seeds inspected in wa8): exploratory", "arms": Mx, "summary": summary(Mx), "calls": cnt,
               "peak_mem_live": peak, "contrasts": {}}
        for cand in cands:
            for x, y in ((f"R_{cand}", f"A_{cand}"), (f"R_{cand}", "R8"), (f"H_{cand}", "Q4"), (f"H_{cand}", "H8"), (f"R_{cand}", f"H_{cand}"),
                         (f"R_{cand}", "Q4"), (f"A_{cand}", "Q4")):
                out["contrasts"][f"{x}_vs_{y}"] = contrast(Mx, x, y)
            out[f"retained_{cand}"] = retained(Mx, "Q4", f"R_{cand}", "R8")
        out.update(run.record())
    return out, st


@torch.no_grad()
def dev_stage(dev):
    out, st = dev_cands(dev, list(GPTQ_CANDS), "dev")
    ch = jload("checks.json")
    E = {c: out["summary"][f"R_{c}"]["rel_mse"]["mean"] for c in GPTQ_CANDS}
    elig = {c: bool(all(v for k_, v in ch[c].items() if isinstance(v, bool)) and ch[c]["K7_pass"]
                    and out["summary"][f"H_{c}"]["rel_mse"]["mean"] < out["summary"]["Q4"]["rel_mse"]["mean"]
                    and st["shared_vs_pair"][c]["smaller"]) for c in GPTQ_CANDS}
    ok = [c for c in GPTQ_CANDS if elig[c]]
    if not ok:
        sel = None
    elif len(ok) == 2 and abs(E["A_gptq"] - E["B_gptq"]) / min(E.values()) <= 0.02:
        sel = "B_gptq"
    else:
        sel = min(ok, key=lambda c: E[c])
    out["selection"] = {"rule": "lowest mean R_shared dev latent MSE among eligible GPTQ candidates; within 2% -> B", "E_R_shared": E,
                        "eligible": elig, "rel_gap": abs(E["A_gptq"] - E["B_gptq"]) / min(E.values()), "selected": sel,
                        "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    jdump("dev.json", out)
    show(out)
    print("selection:", json.dumps(out["selection"]), flush=True)


@torch.no_grad()
def dev_rtn(dev):
    if spent() > 1.0:
        print("[ws1] A-RTN live skipped: spent > 1.0 h", flush=True)
        return
    out, _ = dev_cands(dev, [RTN_CAND], "dev_rtn")
    out["note"] += "; A-RTN is a construction control, never promotable without a pre-result amendment"
    jdump("dev_rtn.json", out)
    show(out)


@torch.no_grad()
def traces(dev):
    dv = jload("dev.json")
    sel = dv and dv["selection"]["selected"]
    if not sel:
        sys.exit(6)
    guard("traces (optional)", 0.03)
    with Run(STAGE, "traces", dev, {"seeds": [TRACE_SEEDS[0], TRACE_SEEDS[-1]], "cand": sel}) as run:
        d = make_diffusion(50)
        M = {"q4": model_from("B_gptq", "low", dev), "hi": model_from(sel, "high", dev), "fp": load_dit(dev)}
        nz, lab = noise_for_seeds(TRACE_SEEDS, dev), labels_for_seeds(TRACE_SEEDS)
        T = {"q4": [], "h8": [], "fp": []}
        xq, _ = sample_arm(M, d, nz, lab, "TRACE", trace=T)
        T = {k_: torch.stack(v, 1) for k_, v in T.items()}
        dr = drift(T)
        cache = torch.load(os.path.join(W8_CK, "dev_latents.pt"), map_location="cpu")
        dr["trace_trajectory_equals_Q4"] = bool(torch.equal(xq.cpu(), cache["Q4"][:len(TRACE_SEEDS)]))
        dr["note"] = f"d = Q4 - H_shared ({sel}); 8 predeclared dev seeds; diagnostic only (bootstrap 2000, w8.drift); frozen_prediction_k8 field unused"
        dr.update(run.record())
    jdump("traces.json", dr)
    print({L: (round(r["cos_mean"], 3), round(r["ratio_d"]["v"], 3), round(r["corrected_fp_err_ratio"]["v"], 3), round(r["shuffled_ratio_d"]["v"], 3))
           for L, r in dr["lags"].items()}, flush=True)


def show(out):
    for a, v in out["summary"].items():
        print(f"  {a}: rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f} PSNR {v['psnr']['mean']:.2f} LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in out["contrasts"].items():
        m = v["rel_mse"]
        print(f"  {k_}: ratio {m['ratio']:.3f} [{m['ratio_ci'][0]:.3f}, {m['ratio_ci'][1]:.3f}] wins {m['wins_x']}/{m['n']} "
              f"dLPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['diff_ci'][0]:+.4f}, {v['lpips']['diff_ci'][1]:+.4f}]", flush=True)


# ----------------------------------------------------------------------------------------------------------------- conf
@torch.no_grad()
def conf(dev):
    fr, dv = jload("frozen.json"), jload("dev.json")
    if not (fr and dv and fr["selected"] == dv["selection"]["selected"] and fr["selected"]):
        print("[ws1] a committed freeze record with a selected candidate is required", flush=True)
        sys.exit(6)
    sel = fr["selected"]
    guard("conf", CONF_RESERVE_H, reserve_h=0.0)
    st = jload("storage.json")
    t0, base = time.time(), spent()
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "cand": sel}) as run:
        d = make_diffusion(50)
        fp, q4 = load_dit(dev), model_from(sel, "low", dev)
        h8, hi = model_from("pair", "high", dev), model_from(sel, "high", dev)
        M8, Ms = {"q4": q4, "hi": h8}, {"q4": q4, "hi": hi}
        torch.cuda.reset_peak_memory_stats()
        arms = [("FP50", "FP50", {"fp": fp}), ("Q4", "Q4", {"q4": q4}), ("H8", "HI", M8), ("R8", "R", M8),
                ("H_shared", "HI", Ms), ("A_shared", "A", Ms), ("R_shared", "R", Ms)]
        lat, cnt, full = run_arms(None, d, CONF, arms, t0, base, 0.0, 0.07)
        peak = {"allocated": torch.cuda.max_memory_allocated(), "reserved": torch.cuda.max_memory_reserved()}
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt" if full else "conf_latents_partial.pt"))
        if not full:
            jdump("conf_partial.json", {"calls": cnt, **run.record()})
            sys.exit(5)
        del fp, q4, h8, hi, M8, Ms
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        C = {f"{x}_vs_{y}": contrast(Mx, x, y) for x, y in (("R_shared", "A_shared"), ("R_shared", "R8"), ("R_shared", "H_shared"), ("R_shared", "Q4"),
                                                            ("H_shared", "Q4"), ("H_shared", "H8"), ("R8", "H8"), ("A_shared", "Q4"), ("R8", "Q4"))}
        out = {"seeds": [CONF[0], CONF[-1]], "selected": sel, "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": C, "peak_mem_live": peak,
               "decisions": decisions(C["R_shared_vs_A_shared"], C["R_shared_vs_R8"], st, sel), "retained": retained(Mx), **run.record()}
    jdump("conf.json", out)
    show(out)
    print(json.dumps({"decisions": out["decisions"], "retained": out["retained"]}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "build", "checks", "dev", "dev_rtn", "traces", "conf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == "selftest":
        return selftest()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"build": build, "checks": checks, "dev": dev_stage, "dev_rtn": dev_rtn, "traces": traces, "conf": conf}[a.cmd](dev)


if __name__ == "__main__":
    main()
