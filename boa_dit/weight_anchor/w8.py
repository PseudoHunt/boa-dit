"""W8 anchors for cross-precision error reuse (results/weight_anchor_w8/PROTOCOL.md). Stage tag wa8.
Bed: heig-8 W4A16 (sampler_corr.run.heig, unchanged), DDIM-50, cfg 1.5, fp16 autocast, batch 32 (weight_anchor.run.guided).
H8 = fixed-calibration W8 + rank-8 carry: the audited heig-8 solver call (qutils.boa_gptq_rows) replayed on the original FP weights with the saved
projected metric (ckpt/temporal/heig8_layers.pt) at bits=8; carry delta8 = (W - Q8) V with the shared V (deflate.carry.make_carry).
Arms: FP50 Q4 H8 A8 R8 S8 R_FP; every arm on its own trajectory; at an anchor, Q4 and the anchor model see the same latent.
Stages: build | checks | dev | dev_k8 (optional) | conf
Run: python -m boa_dit.weight_anchor.w8 <stage>
"""
import argparse
import contextlib
import csv
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.deflate.carry import install, make_carry
from boa_dit.model import NULL_CLASS, block_linear, labels_for_seeds, load_dit, load_quant_state, make_diffusion, noise_for_seeds
from boa_dit.qutils import boa_gptq_rows
from boa_dit.sample import CTX
from boa_dit.sampler_corr.run import CARRY, HEIG, heig
from boa_dit.weight_anchor.controls import metrics
from boa_dit.weight_anchor.run import BS, boot, guided, model_bytes, ratio_ci, rel
from quantizers.boa import BoA

STAGE = "wa8"
OUT = os.path.join(RESULTS, "weight_anchor_w8")
CK = os.path.join(CKPT_DIR, "weight_anchor_w8")
H8_FILE = os.path.join(CK, "h8.pt")
LAYERS = os.path.join(CKPT_DIR, "temporal", "heig8_layers.pt")
W4_STATE = os.path.join(CKPT_DIR, f"ckpt_{HEIG}", "model.pt")
CAP_GPU_H = 1.0
CONF_RESERVE_H = 0.15
K = 4
DEV, CONF, C1 = list(range(94000, 94032)), list(range(94100, 94164)), list(range(92000, 92032))
ARMS = ("FP50", "Q4", "H8", "A8", "R8", "S8", "R_FP")
PLAIN = {"FP50": "fp", "Q4": "q4", "H8": "h8"}


def expect(k):
    na = len(range(0, 50, k))
    return {"FP50": {"q4": 0, "h8": 0, "fp": 50}, "Q4": {"q4": 50, "h8": 0, "fp": 0}, "H8": {"q4": 0, "h8": 50, "fp": 0},
            "A8": {"q4": 50 - na, "h8": na, "fp": 0}, "R8": {"q4": 50, "h8": na, "fp": 0}, "S8": {"q4": 0, "h8": na, "fp": 0},
            "R_FP": {"q4": 50, "h8": 0, "fp": na}}


def jdump(name, obj):
    atomic_write_json(os.path.join(OUT, name), obj)


def jload(name):
    f = os.path.join(OUT, name)
    return json.load(open(f)) if os.path.exists(f) else None


def spent():
    f = os.path.join(RESULTS, "gpu_hours.csv")
    return sum(float(r[-1]) for r in csv.reader(open(f)) if r and r[0] == STAGE) if os.path.exists(f) else 0.0


def guard(label, est_h, reserve_h=0.0):
    s = spent()
    ok = s + est_h + reserve_h <= CAP_GPU_H
    b = jload("budget.json") or {"cap_gpu_h": CAP_GPU_H, "log": []}
    b["log"].append({"stage": label, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "spent_before_h": s, "estimate_h": est_h,
                     "reserved_h": reserve_h, "ok": ok})
    jdump("budget.json", b)
    print(f"[wa8 budget] {label}: spent {s:.3f} GPU-h + estimate {est_h:.2f} h + reserve {reserve_h:.2f} h vs cap {CAP_GPU_H} -> "
          f"{'OK' if ok else 'STOP (cap)'}", flush=True)
    if not ok:
        sys.exit(5)


class Clock:
    """Runtime stop: before each unit of work, spent (ledger) + this run's elapsed + next estimate must fit the cap."""

    def __init__(self, reserve_h=0.0):
        self.t0, self.base, self.reserve = time.time(), spent(), reserve_h

    def ok(self, next_h, what):
        used = self.base + (time.time() - self.t0) / 3600
        fit = used + next_h + self.reserve <= CAP_GPU_H
        if not fit:
            print(f"[wa8 runtime stop] {what}: used {used:.3f} h + next {next_h:.3f} h + reserve {self.reserve:.2f} h > cap", flush=True)
        return fit


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------------- build
@contextlib.contextmanager
def capture_grid(box):
    """Record the (scale, zero) the BoA solver chose (same wrapper as temporal.rebuild); the solver itself is untouched."""
    orig = BoA.quant

    def quant(self, *a, **kw):
        orig(self, *a, **kw)
        box["scale"] = self.quantizer.scale.detach().float().clone()
        box["zero"] = self.quantizer.zero.detach().float().clone()
    BoA.quant = quant
    try:
        yield box
    finally:
        BoA.quant = orig


@torch.no_grad()
def replay(Wf, H, bits, replace):
    """heig-8's solver call (deflate.quantize_dc: boa_gptq_groups with one group == boa_gptq_rows) on the original FP weight."""
    lin = nn.Linear(Wf.shape[1], Wf.shape[0], bias=False, device=Wf.device, dtype=torch.float32)
    lin.weight.data = Wf.clone()
    box = {}
    with capture_grid(box):
        boa_gptq_rows(lin, H, list(range(Wf.shape[0])), bits, replace)
    scale, zero = box["scale"].reshape(-1, 1), box["zero"].reshape(-1, 1)
    Q = lin.weight.data.float()
    codes = torch.round(Q / scale + zero)
    return Q, scale, zero, codes


def quad(E, H):
    return float(torch.einsum("ij,jk,ik->", E.double(), H.double(), E.double()))


def theoretical_bytes(m, carry_numel):
    rows_q = n_q = rows_a = n_a = 0
    for b in m.blocks:
        for ln in ("attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"):
            w = block_linear(b, ln).weight
            rows_q += w.shape[0]; n_q += w.numel()
        w = b.adaLN_modulation[1].weight
        rows_a += w.shape[0]; n_a += w.numel()
    tot = sum(p.numel() for p in m.parameters()) + sum(x.numel() for x in m.buffers())
    other = tot - n_q - n_a
    qp = {4: n_q // 2, 8: n_q}
    out = {"assumptions": "quantized block linears packed at 4 or 8 bits + per-row fp16 scale and 8-bit zero point; carry fp16 V + delta; "
                          "adaLN at 8 bits + per-row fp16 scale and 8-bit zero; all other tensors (biases, embeddings, final layer, norms) fp16",
           "n_quantized_linear_params": n_q, "n_adaln_params": n_a, "n_other_params": other,
           "carry_bytes": 2 * carry_numel, "adaln_bytes": n_a + 3 * rows_a, "other_bytes": 2 * other}
    for b in (4, 8):
        out[f"w{b}_linear_bytes"] = qp[b] + 3 * rows_q
        out[f"w{b}_whole_model_bytes"] = out[f"w{b}_linear_bytes"] + out["carry_bytes"] + out["adaln_bytes"] + out["other_bytes"]
    out["fp16_whole_model_bytes"] = 2 * tot
    return out


@torch.no_grad()
def build(dev):
    guard("build", 0.25, CONF_RESERVE_H)
    os.makedirs(CK, exist_ok=True)
    S = torch.load(LAYERS, map_location="cpu")
    W_fp = load_dit("cpu")
    W_4 = load_quant_state(load_dit("cpu"), W4_STATE)
    c4 = torch.load(CARRY, map_location="cpu")
    keys = list(S.keys())
    assert len(keys) == 112 and all(S[k]["bits"] == 4 for k in keys)
    art = {"codes": {}, "scale": {}, "zero": {}, "V": {}, "delta": {}}
    lay = {}
    with Run(STAGE, "build", dev, {"layers": len(keys), "bits": [4, 8]}) as run:
        t0 = time.time()
        for li, key in enumerate(keys):
            s = S[key]
            l, ln = key.split(".", 1)
            Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            W4s = block_linear(W_4.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            H, V = s["H"].to(dev), s["V"].to(dev)
            r = {}
            # replay audit at bits=4
            Q4, sc4, z4, cd4 = replay(Wf, H, 4, s["replace"])
            k4 = make_carry(Wf, Q4, V)
            r["w4_replay"] = {"weight_bitexact": bool(torch.equal(Q4, W4s)), "weight_max_abs": float((Q4 - W4s).abs().max()),
                              "scale_bitexact": bool(torch.equal(sc4.cpu(), s["scale"].reshape(-1, 1))),
                              "zero_bitexact": bool(torch.equal(z4.cpu(), s["zero"].reshape(-1, 1))),
                              "delta_bitexact": bool(torch.equal(k4["delta"], c4[key]["delta"])), "V_bitexact": bool(torch.equal(k4["V"], c4[key]["V"])),
                              "codes_min": int(cd4.min()), "codes_max": int(cd4.max()),
                              "dequant_bitexact": bool(torch.equal(sc4 * (cd4 - z4), Q4))}
            # W8
            Q8, sc8, z8, cd8 = replay(Wf, H, 8, s["replace"])
            k8 = make_carry(Wf, Q8, V)
            assert int(cd8.min()) >= 0 and int(cd8.max()) <= 255
            deq = sc8 * (cd8 - z8)
            V16 = k8["V"].to(dev).float()
            r["w8"] = {"codes_min": int(cd8.min()), "codes_max": int(cd8.max()), "dequant_bitexact": bool(torch.equal(deq, Q8)),
                       "delta_shape": list(k8["delta"].shape), "V_shape": list(k8["V"].shape), "delta_dtype": str(k8["delta"].dtype),
                       "V_equal_w4_V": bool(torch.equal(k8["V"], c4[key]["V"])), "finite": bool(torch.isfinite(Q8).all() and torch.isfinite(k8["delta"].float()).all())}
            for nm, Q, kk in (("w4", Q4, k4), ("w8", Q8, k8)):
                E = Wf - Q
                Ec = E - kk["delta"].to(dev).float() @ V16.T                          # residual after the rank-8 carry
                r[nm + "_err"] = {"rel_weight_err": float(E.norm() / Wf.norm()), "H_quad": quad(E, H), "H_quad_after_carry": quad(Ec, H)}
            lay[key] = r
            art["codes"][key] = cd8.to(torch.uint8).cpu()
            art["scale"][key], art["zero"][key] = sc8.cpu(), z8.cpu()
            art["V"][key], art["delta"][key] = k8["V"], k8["delta"]
            if li == 0:
                el = time.time() - t0
                print(f"  profile: layer 0 took {el:.1f} s (qkv, d_in 1152); fc2 layers are larger", flush=True)
            if li % 8 == 0 or li == len(keys) - 1:
                print(f"  [{li + 1}/112] {key}: W4 replay exact {all(v for k_, v in r['w4_replay'].items() if k_.endswith('bitexact'))}; "
                      f"rel err W4 {r['w4_err']['rel_weight_err']:.4f} W8 {r['w8_err']['rel_weight_err']:.5f}; {time.time() - t0:.0f} s", flush=True)
        settings = {"label": "H8: fixed-calibration W8 + rank-8 carry", "solver": "boa_dit.qutils.boa_gptq_rows (unchanged), bits=8, sym=False, "
                    "qparam_comput=Hessian, act_order_col=False, replace from cache (1/256), BoA preprocessing (dead columns, 1% damping)",
                    "target": "original FP weights (DiT-XL-2-256x256.pt)", "metric": "saved projected H = P H P from heig8_layers.pt",
                    "basis": "saved rank-8 V (shared with W4)", "carry": "deflate.carry.make_carry: fp32 (W - Q8) V stored fp16, V fp16",
                    "deploy": "weights stored fp32 dequantized scale*(code - zero) as the W4 bed; forward under fp16 autocast",
                    "unchanged": "biases, adaLN W8 RTN, non-block tensors from the heig-8 W4 state dict"}
        art["settings"] = settings
        atomic_torch_save(art, H8_FILE)
        rec = run.record()
    hashes = {"h8.pt": sha256(H8_FILE), "heig8_layers.pt": sha256(LAYERS), "heig8_w4_model.pt": sha256(W4_STATE), "heig8_carry.pt": sha256(CARRY),
              "DiT-XL-2-256x256.pt": sha256(os.path.join(CKPT_DIR, "DiT-XL-2-256x256.pt"))}
    allx = lambda f: all(v["w4_replay"][f] for v in lay.values())
    agg = {nm: {m: float(sum(v[nm + "_err"][m] for v in lay.values())) for m in ("H_quad", "H_quad_after_carry")} for nm in ("w4", "w8")}
    for nm in ("w4", "w8"):
        agg[nm]["rel_weight_err_mean"] = float(np.mean([v[nm + "_err"]["rel_weight_err"] for v in lay.values()]))
    cnum = sum(art["V"][k].numel() + art["delta"][k].numel() for k in keys)
    out = {"settings": settings, "w4_replay_all": {f: allx(f) for f in ("weight_bitexact", "scale_bitexact", "zero_bitexact", "delta_bitexact",
                                                                          "V_bitexact", "dequant_bitexact")},
           "w4_replay_codes_range": [min(v["w4_replay"]["codes_min"] for v in lay.values()), max(v["w4_replay"]["codes_max"] for v in lay.values())],
           "w8_codes_range": [min(v["w8"]["codes_min"] for v in lay.values()), max(v["w8"]["codes_max"] for v in lay.values())],
           "w8_dequant_bitexact_all": all(v["w8"]["dequant_bitexact"] for v in lay.values()),
           "w8_finite_all": all(v["w8"]["finite"] for v in lay.values()), "aggregate_errors": agg,
           "bytes_theoretical": theoretical_bytes(W_fp, cnum), "hashes_sha256": hashes, "layers": lay, **rec}
    jdump("build.json", out)
    print(json.dumps({k: out[k] for k in ("w4_replay_all", "w4_replay_codes_range", "w8_codes_range", "w8_dequant_bitexact_all", "aggregate_errors")},
                     indent=1), flush=True)


def h8_model(dev):
    """heig-8 W4 state dict with the 112 quantized weights replaced by dequantized Q8, carry delta8 installed."""
    art = torch.load(H8_FILE, map_location="cpu")
    m = load_dit(dev)
    load_quant_state(m, W4_STATE)
    for key, c in art["codes"].items():
        l, ln = key.split(".", 1)
        w = block_linear(m.blocks[int(l[1:])], ln).weight
        w.data.copy_((art["scale"][key] * (c.float() - art["zero"][key])).to(dev))
    install(m, {k: {"V": art["V"][k], "delta": art["delta"][k]} for k in art["codes"]})
    m.requires_grad_(False)
    return m


def models(dev):
    fp = load_dit(dev)
    q4, _ = heig(dev)
    return {"fp": fp, "q4": q4, "h8": h8_model(dev)}


# --------------------------------------------------------------------------------------------------------------- sampler
@torch.no_grad()
def sample_arm(M, d, noise, labels, arm, k=K, anchor=None, zero_aux=False, rec=None, trace=None):
    """One trajectory of `arm`. anchor overrides the anchor model of A8/R8/S8 (C3 uses 'q4'). rec: list collecting the inputs of every call (C4).
    trace (arm 'TRACE'): Q4-50 trajectory that also records same-state h8 and fp outputs (channels 0-3)."""
    x = noise.float().clone()
    n = len(x)
    lab = torch.as_tensor(labels, device=noise.device, dtype=torch.long)
    y = torch.cat([lab, torch.full_like(lab, NULL_CLASS)])
    cnt = {"q4": 0, "h8": 0, "fp": 0}
    st = {"d": None, "a": None, "hold": None}
    am = "fp" if arm == "R_FP" else (anchor or "h8")
    for i, ti in enumerate(list(range(d.num_timesteps))[::-1]):
        CTX.step, CTX.n = i, n

        def model_fn(xx, ts, _i=i):
            def call(name):
                cnt[name] += 1
                if rec is not None:
                    rec.append({"i": _i, "model": name, "x": xx.detach().clone().cpu(), "t": ts.detach().clone().cpu(), "a": st["a"]})
                return guided(M[name], xx, ts, y, n)
            if arm == "TRACE":
                res = call("q4")
                trace["q4"].append(res[:, :4].cpu())
                trace["h8"].append(guided(M["h8"], xx, ts, y, n)[:, :4].cpu())
                trace["fp"].append(guided(M["fp"], xx, ts, y, n)[:, :4].cpu())
                return res
            if arm in PLAIN:
                res = call(PLAIN[arm])
                if zero_aux:
                    res = res.clone()
                    res[:, 4:] = 0
                return res
            if _i % k:
                if arm == "S8":
                    return st["hold"].clone()                                    # no denoiser call
                res = call("q4")
                if arm in ("R8", "R_FP"):
                    res = res.clone()
                    res[:, :4] -= st["d"]
                return res
            res = call(am)                                                       # anchor: used directly
            if arm in ("R8", "R_FP"):
                rq = call("q4")                                                  # same xx, ts, y
                st["d"], st["a"] = rq[:, :4] - res[:, :4], _i
                if rec is not None:
                    rec.append({"i": _i, "model": "d", "d": st["d"].cpu()})
            if arm == "S8":
                st["hold"] = res
            return res
        t = torch.full((n,), ti, device=noise.device, dtype=torch.long)
        x = d.ddim_sample(model_fn, x, t, clip_denoised=False, eta=0.0)["sample"]
    CTX.step = None
    return x, cnt


def gen(M, d, seeds, arms, k=K, clock=None, est_batch_h=0.0):
    lat, cnt = {a: [] for a in arms}, {}
    for b in range(0, len(seeds), BS):
        s = seeds[b:b + BS]
        if clock is not None and not clock.ok(est_batch_h, f"batch {s[0]}-{s[-1]}"):
            return {a: torch.cat(v) for a, v in lat.items() if v}, cnt, False
        nz, lab = noise_for_seeds(s, "cuda:0"), labels_for_seeds(s)
        for a in arms:
            x, c = sample_arm(M, d, nz, lab, a, k=k)
            assert torch.isfinite(x).all(), a
            lat[a].append(x.cpu())
            cnt[a] = c
        print(f"  seeds {s[0]}-{s[-1]}: {', '.join(arms)} done", flush=True)
    return {a: torch.cat(v) for a, v in lat.items()}, cnt, True


# ---------------------------------------------------------------------------------------------------------------- checks
@torch.no_grad()
def checks(dev):
    b = jload("build.json")
    if not (b and all(b["w4_replay_all"].values()) and b["w8_dequant_bitexact_all"] and b["w8_finite_all"]
            and b["w8_codes_range"][0] >= 0 and b["w8_codes_range"][1] <= 255):
        print("[wa8] build audit not passed: stop", flush=True)
        sys.exit(6)
    guard("checks", 0.12, CONF_RESERVE_H)
    with Run(STAGE, "checks", dev, {"seeds": [C1[0], C1[-1]]}) as run:
        M = models(dev)
        d = make_diffusion(50)
        nz, lab = noise_for_seeds(C1, dev), labels_for_seeds(C1)
        old = torch.load(os.path.join(CKPT_DIR, "weight_anchor_controls", "dev_latents.pt"), map_location="cpu")
        lat, cnt = {}, {}
        for a in ARMS:
            x, c = sample_arm(M, d, nz, lab, a)
            lat[a], cnt[a] = x.cpu(), c
            print(f"  {a}: {c}", flush=True)
        ch = {"C1": {f"{a}=={b_}": bool(torch.equal(lat[a], old[b_])) for a, b_ in (("FP50", "FP50"), ("Q4", "Q"), ("R_FP", "R1"))}}
        x2, c2 = sample_arm(M, d, nz, lab, "R8", k=1)
        ch["C2_R8_K1_equals_H8"], ch["C2_counts"] = bool(torch.equal(x2.cpu(), lat["H8"])), c2
        rec = []
        x3, c3 = sample_arm(M, d, nz, lab, "R8", anchor="q4", rec=rec)
        ds = [r["d"] for r in rec if r["model"] == "d"]
        ch["C3_Q_equals_H_zero_correction"] = bool(all(torch.count_nonzero(v) == 0 for v in ds)) and len(ds) == 13
        ch["C3_equals_Q4"] = bool(torch.equal(x3.cpu(), lat["Q4"]))
        rec = []
        sample_arm(M, d, nz, lab, "R8", rec=rec)
        calls = [r for r in rec if r["model"] != "d"]
        ok4, n_anchor = True, 0
        for i in range(50):
            ci = [r for r in calls if r["i"] == i]
            if i % K == 0:
                h, q = [r for r in ci if r["model"] == "h8"], [r for r in ci if r["model"] == "q4"]
                ok4 &= len(h) == 1 and len(q) == 1 and torch.equal(h[0]["x"], q[0]["x"]) and torch.equal(h[0]["t"], q[0]["t"])
                n_anchor += 1
            else:
                ok4 &= len(ci) == 1 and ci[0]["model"] == "q4" and ci[0]["a"] == i - i % K
        ch["C4_anchor_same_state"], ch["C4_n_anchors"] = bool(ok4), n_anchor
        del rec, calls
        ch["C5_counts"] = cnt
        ch["C5_pass"] = all(cnt[a] == expect(K)[a] for a in ARMS)
        x8, _ = sample_arm(M, d, nz[:8], lab[:8], "H8")
        ch["C6_build_audit"] = {k_: b[k_] for k_ in ("w4_replay_all", "w4_replay_codes_range", "w8_codes_range", "w8_dequant_bitexact_all", "w8_finite_all")}
        ch["C6_outputs_finite"] = bool(all(torch.isfinite(v).all() for v in lat.values()))
        ch["C6b_batch8_vs_batch32_H8_max_abs"] = float((x8.cpu() - lat["H8"][:8]).abs().max())
        ch["C6b_bitwise"] = bool(torch.equal(x8.cpu(), lat["H8"][:8]))
        xz, _ = sample_arm(M, d, nz, lab, "FP50", zero_aux=True)
        ch["C6c_aux_zero_equals_FP50"] = bool(torch.equal(xz.cpu(), lat["FP50"]))
        # rebuilt-W4 forward vs the original heig-8 forward (bitwise; weights are bitwise equal by the build audit)
        art = torch.load(H8_FILE, map_location="cpu")
        S = torch.load(LAYERS, map_location="cpu")
        mq = load_dit(dev)
        load_quant_state(mq, W4_STATE)
        W_fp = load_dit("cpu")
        rc = {}
        for key in art["codes"]:
            l, ln = key.split(".", 1)
            Wf = block_linear(W_fp.blocks[int(l[1:])], ln).weight.data.to(dev).float()
            Q, _, _, _ = replay(Wf, S[key]["H"].to(dev), 4, S[key]["replace"])
            block_linear(mq.blocks[int(l[1:])], ln).weight.data.copy_(Q)
            rc[key] = make_carry(Wf, Q, S[key]["V"].to(dev))                    # fp32 V, as the heig-8 build used
        install(mq, rc)
        del S, W_fp
        ts = torch.full((BS,), 500, device=dev, dtype=torch.long)
        y = torch.cat([torch.as_tensor(lab, device=dev), torch.full((BS,), NULL_CLASS, device=dev, dtype=torch.long)])
        o1, o2 = guided(mq, nz.float(), ts, y, BS), guided(M["q4"], nz.float(), ts, y, BS)
        ch["C6_rebuilt_w4_forward_bitexact"] = bool(torch.equal(o1, o2))
        # same-state accuracy at one call (sanity, not a result): ||eps_X - eps_FP||^2 at t=500 on noise
        of, oh = guided(M["fp"], nz.float(), ts, y, BS), guided(M["h8"], nz.float(), ts, y, BS)
        ch["sanity_t500_eps_err"] = {"q4": float((o2[:, :4] - of[:, :4]).pow(2).sum()), "h8": float((oh[:, :4] - of[:, :4]).pow(2).sum())}
        ch["pass"] = bool(all(ch["C1"].values()) and ch["C2_R8_K1_equals_H8"] and ch["C3_Q_equals_H_zero_correction"] and ch["C3_equals_Q4"]
                          and ch["C4_anchor_same_state"] and ch["C5_pass"] and ch["C6_outputs_finite"] and ch["C6c_aux_zero_equals_FP50"]
                          and ch["C6_rebuilt_w4_forward_bitexact"])
        ch["resident_model_bytes_simulation"] = {k_: model_bytes(m) for k_, m in M.items()}
        ch["resident_carry_bytes_simulation"] = int(sum(v.numel() * 2 for v in art["V"].values()) + sum(v.numel() * 2 for v in art["delta"].values()))
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k_: v for k_, v in ch.items() if k_ != "config"}, indent=1), flush=True)
    if not ch["pass"]:
        print("[wa8] checks FAILED: stop", flush=True)
        sys.exit(6)


# ------------------------------------------------------------------------------------------------------- dev and traces
def contrasts(M):
    pairs = [("R8", "A8"), ("R8", "S8"), ("R8", "Q4"), ("R8", "H8"), ("R8", "R_FP"), ("H8", "Q4"), ("A8", "Q4"), ("S8", "Q4"), ("R_FP", "Q4"),
             ("A8", "H8"), ("S8", "A8")]
    out = {}
    for x, y in pairs:
        if x in M and y in M:
            r = {}
            for m, ratio in (("rel_mse", True), ("psnr", False), ("lpips", False)):
                a, b = np.asarray(M[x][m]), np.asarray(M[y][m])
                r[m] = boot(a, b, ratio)
                if ratio:
                    r[m]["diff"] = float(np.mean(a - b))
                    r[m]["diff_ci"] = boot(a, b, False)["ci"]
                r[m]["median_diff"] = float(np.median(a - b))
                r[m]["wins_x"] = int(np.sum(a < b) if m != "psnr" else np.sum(a > b))
            out[f"{x}_vs_{y}"] = r
    return out


def summary(M):
    return {a: {m: {"mean": float(np.mean(v[m])), "median": float(np.median(v[m]))} for m in ("rel_mse", "psnr", "lpips")} for a, v in M.items()}


def reading(c):
    """pre-registered reading of R8 vs A8."""
    v, lo, hi = c["rel_mse"]["rel_change"], c["rel_mse"]["ci"][0], c["rel_mse"]["ci"][1]
    lp_bad = c["lpips"]["ci"][0] > 0
    if v <= -0.10 and hi < 0 and not lp_bad:
        return "useful reuse (>= 10% lower latent MSE than A8, CI excludes 0, no statistically clear LPIPS regression)"
    if hi < 0:
        return "smaller positive" + (" (LPIPS clearly worse)" if lp_bad else " (below the 10% threshold)" if v > -0.10 else "")
    if lo > 0:
        return "harmful (R8 worse than A8)"
    return "inconclusive"


def retained(M):
    q, r, f = (np.asarray(M[a]["rel_mse"]) for a in ("Q4", "R8", "R_FP"))
    g = np.random.default_rng(0)
    idx = [g.integers(0, len(q), len(q)) for _ in range(2000)]
    B = [(q[i].mean() - r[i].mean()) / (q[i].mean() - f[i].mean()) for i in idx]
    num = [q[i].mean() - r[i].mean() for i in idx]
    den = [q[i].mean() - f[i].mean() for i in idx]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    den_ci = ci(den)
    return {"B": float((q.mean() - r.mean()) / (q.mean() - f.mean())), "B_ci": ci(B), "gain_R8": float(q.mean() - r.mean()), "gain_R8_ci": ci(num),
            "gain_R_FP": float(q.mean() - f.mean()), "gain_R_FP_ci": den_ci, "interpretable": bool(den_ci[0] > 0),
            "E": {"Q4": float(q.mean()), "R8": float(r.mean()), "R_FP": float(f.mean())}}


def drift(T, n_img=None):
    """Discrepancy persistence on same-state traces T[name]: [N, 50, 4, 32, 32] along the Q4-50 trajectory."""
    q, h, f = (T[k].double() for k in ("q4", "h8", "fp"))
    if n_img:
        q, h, f = q[:n_img], h[:n_img], f[:n_img]
    D, E = q - h, q - f
    N = D.shape[0]
    sq = lambda X: X.flatten(2).pow(2).sum(2)                                   # [N, 50] (or pairs)
    nd = sq(D).sqrt()
    stage = lambda t: "early" if t <= 16 else ("middle" if t <= 33 else "late")
    out = {"n_images": N, "per_call_mean_sq": {"d": sq(D).mean(0).tolist(), "e_q4_fp": sq(E).mean(0).tolist(), "e_h8_fp": sq(h - f).mean(0).tolist()},
           "lags": {}, "fixed_period": {}}
    shift = torch.roll(torch.arange(N), -1)
    for L in (1, 2, 4, 8):
        a = torch.arange(0, 50 - L)
        t = a + L
        Dt, Da, Et, Ea = D[:, t], D[:, a], E[:, t], E[:, a]
        dd, dt2 = sq(Dt - Da), sq(Dt)
        cos = (Dt * Da).flatten(2).sum(2) / (nd[:, t] * nd[:, a])
        valid = nd[:, a] > 0
        r = {"pairs": int(len(a)), "zero_norm_anchor_pairs_excluded": int((~valid).sum()),
             "cos_mean": float(cos[valid].mean()), "cos_median": float(cos[valid].median()),
             "norm_ratio_median": float((nd[:, t] / nd[:, a])[valid].median()), "norm_ratio_mean": float((nd[:, t] / nd[:, a])[valid].mean()),
             "sum_d_t_minus_d_a": float(dd.sum()), "sum_d_t": float(dt2.sum()),
             "ratio_d": ratio_ci(dd.sum(1).numpy(), dt2.sum(1).numpy()),
             "shuffled_ratio_d": ratio_ci(sq(Dt - D[shift][:, a]).sum(1).numpy(), dt2.sum(1).numpy()),
             "corrected_fp_err_ratio": ratio_ci(sq(Et - Da).sum(1).numpy(), sq(Et).sum(1).numpy()),
             "fp_anchor_reuse_err_ratio": ratio_ci(sq(Et - Ea).sum(1).numpy(), sq(Et).sum(1).numpy()),
             "by_stage": {}}
        for sname in ("early", "middle", "late"):
            m = torch.tensor([stage(int(tt)) == sname for tt in t])
            if m.any():
                r["by_stage"][sname] = {"ratio_d": ratio_ci(dd[:, m].sum(1).numpy(), dt2[:, m].sum(1).numpy()),
                                        "corrected_fp_err_ratio": ratio_ci(sq(Et[:, m] - Da[:, m]).sum(1).numpy(), sq(Et[:, m]).sum(1).numpy())}
        out["lags"][str(L)] = r
    for KK in (2, 4, 8):
        t = torch.tensor([tt for tt in range(50) if tt % KK])
        a = t - t % KK
        Dt, Da, Et, Ea = D[:, t], D[:, a], E[:, t], E[:, a]
        out["fixed_period"][str(KK)] = {
            "n_anchors": len(range(0, 50, KK)), "non_anchor_calls": int(len(t)),
            "ratio_d": ratio_ci(sq(Dt - Da).sum(1).numpy(), sq(Dt).sum(1).numpy()),
            "corrected_fp_err_ratio": ratio_ci(sq(Et - Da).sum(1).numpy(), sq(Et).sum(1).numpy()),
            "fp_anchor_reuse_err_ratio": ratio_ci(sq(Et - Ea).sum(1).numpy(), sq(Et).sum(1).numpy()),
            "uncorrected_fp_err_sum": float(sq(Et).sum()), "corrected_fp_err_sum": float(sq(Et - Da).sum())}
    k8 = out["fixed_period"]["8"]["corrected_fp_err_ratio"]["v"]
    out["frozen_prediction_k8"] = {"rule": "R8-vs-A8 latent MSE change at K=8 predicted negative iff offline K=8 corrected-error ratio < 1",
                                   "offline_ratio": k8, "predicted_sign": "negative (reuse helps)" if k8 < 1 else "non-negative",
                                   "frozen_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    return out


@torch.no_grad()
def dev_stage(dev):
    ch = jload("checks.json")
    if not (ch and ch["pass"]):
        sys.exit(6)
    guard("dev+traces", 0.12, CONF_RESERVE_H)
    clock = Clock(CONF_RESERVE_H)
    os.makedirs(CK, exist_ok=True)
    with Run(STAGE, "dev", dev, {"seeds": [DEV[0], DEV[-1]], "arms": ARMS, "K": K}) as run:
        M = models(dev)
        d = make_diffusion(50)
        lat, cnt, full = gen(M, d, DEV, ARMS, clock=clock, est_batch_h=0.05)
        assert full and all(cnt[a] == expect(K)[a] for a in ARMS), cnt
        atomic_torch_save(lat, os.path.join(CK, "dev_latents.pt"))
        # traces along the common Q4-50 trajectory
        n_tr = len(DEV) if clock.ok(0.03, "traces (32 images)") else 8
        nz, lab = noise_for_seeds(DEV[:n_tr], dev), labels_for_seeds(DEV[:n_tr])
        T = {"q4": [], "h8": [], "fp": []}
        xq, _ = sample_arm(M, d, nz, lab, "TRACE", trace=T)
        T = {k_: torch.stack(v, 1) for k_, v in T.items()}                       # [N, 50, 4, 32, 32]
        trace_ok = bool(torch.equal(xq.cpu(), lat["Q4"][:n_tr]))
        atomic_torch_save(T, os.path.join(CK, "dev_traces.pt"))
        dr = drift(T)
        dr["trace_trajectory_equals_Q4_arm"] = trace_ok
        dr["label"] = f"development only; {n_tr} images" + (" (reduced count)" if n_tr < len(DEV) else "")
        jdump("drift.json", {**dr, "seeds": [DEV[0], DEV[n_tr - 1]]})
        del M
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        out = {"seeds": [DEV[0], DEV[-1]], "note": "development: correctness and diagnosis only; nothing selected", "arms": Mx, "summary": summary(Mx),
               "calls": cnt, "contrasts": contrasts(Mx)}
        out["reading_R8_vs_A8_dev"] = reading(out["contrasts"]["R8_vs_A8"])
        out["retained_benefit_dev"] = retained(Mx)
        out.update(run.record())
    jdump("dev.json", out)
    report_contrasts(out)
    fp = dr["fixed_period"]
    print("drift fixed-period corrected-error ratios:", {k_: round(v["corrected_fp_err_ratio"]["v"], 3) for k_, v in fp.items()},
          "| ratio_d:", {k_: round(v["ratio_d"]["v"], 3) for k_, v in fp.items()}, flush=True)


def report_contrasts(out):
    for a, v in out["summary"].items():
        print(f"  {a}: latent rel MSE mean {v['rel_mse']['mean']:.5f} median {v['rel_mse']['median']:.5f}  PSNR {v['psnr']['mean']:.2f}  "
              f"LPIPS {v['lpips']['mean']:.4f}", flush=True)
    for k_, v in out["contrasts"].items():
        print(f"  {k_}: latent {100 * v['rel_mse']['rel_change']:+.1f}% [{100 * v['rel_mse']['ci'][0]:+.1f}, {100 * v['rel_mse']['ci'][1]:+.1f}] "
              f"LPIPS {v['lpips']['diff']:+.4f} [{v['lpips']['ci'][0]:+.4f}, {v['lpips']['ci'][1]:+.4f}] wins {v['rel_mse']['wins_x']}", flush=True)
    for k_ in ("reading_R8_vs_A8_dev", "reading_R8_vs_A8"):
        if k_ in out:
            print(k_, ":", out[k_], flush=True)
    rb = out.get("retained_benefit") or out.get("retained_benefit_dev")
    print("retained benefit:", json.dumps(rb), flush=True)


@torch.no_grad()
def dev_k8(dev):
    dr = jload("drift.json")
    if not (dr and "frozen_prediction_k8" in dr):
        print("[wa8] drift prediction must be frozen (and committed) before K=8 endpoints", flush=True)
        sys.exit(6)
    guard("dev_k8 (optional)", 0.05, CONF_RESERVE_H)
    with Run(STAGE, "dev_k8", dev, {"seeds": [DEV[0], DEV[-1]], "arms": ["R8", "A8"], "K": 8}) as run:
        M = models(dev)
        d = make_diffusion(50)
        lat, cnt, full = gen(M, d, DEV, ("R8", "A8"), k=8)
        assert full and all(cnt[a] == expect(8)[a] for a in ("R8", "A8")), cnt
        old = torch.load(os.path.join(CK, "dev_latents.pt"), map_location="cpu")
        lat["FP50"] = old["FP50"]
        del M
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        c = contrasts(Mx)["R8_vs_A8"]
        sign = "negative (reuse helps)" if c["rel_mse"]["rel_change"] < 0 else "non-negative"
        out = {"note": "optional development-only K=8 check of the frozen drift prediction; not validation", "arms": Mx, "summary": summary(Mx),
               "calls": cnt, "R8_vs_A8_k8": c, "observed_sign": sign, "predicted": dr["frozen_prediction_k8"],
               "agrees": sign == dr["frozen_prediction_k8"]["predicted_sign"], **run.record()}
    jdump("dev_k8.json", out)
    print(f"  K=8 R8 vs A8: {100 * c['rel_mse']['rel_change']:+.1f}% [{100 * c['rel_mse']['ci'][0]:+.1f}, {100 * c['rel_mse']['ci'][1]:+.1f}]; "
          f"predicted {dr['frozen_prediction_k8']['predicted_sign']}, observed {sign}", flush=True)


@torch.no_grad()
def conf(dev):
    if not (jload("dev.json") and jload("frozen.json")):
        print("[wa8] dev and the freeze record are required before confirmation", flush=True)
        sys.exit(6)
    guard("conf", CONF_RESERVE_H)
    clock = Clock()
    with Run(STAGE, "conf", dev, {"seeds": [CONF[0], CONF[-1]], "arms": ARMS, "K": K}) as run:
        M = models(dev)
        d = make_diffusion(50)
        lat, cnt, full = gen(M, d, CONF, ARMS, clock=clock, est_batch_h=0.05)
        atomic_torch_save(lat, os.path.join(CK, "conf_latents.pt" if full else "conf_latents_partial.pt"))
        if not full:
            jdump("conf_partial.json", {"completed_images": int(len(next(iter(lat.values())))) if lat else 0, "calls": cnt, **run.record()})
            print("[wa8] confirmation stopped at the cap: partial", flush=True)
            sys.exit(5)
        assert all(cnt[a] == expect(K)[a] for a in ARMS), cnt
        del M
        torch.cuda.empty_cache()
        Mx = metrics(lat, dev)
        out = {"seeds": [CONF[0], CONF[-1]], "arms": Mx, "summary": summary(Mx), "calls": cnt, "contrasts": contrasts(Mx)}
        out["reading_R8_vs_A8"] = reading(out["contrasts"]["R8_vs_A8"])
        hq = out["contrasts"]["H8_vs_Q4"]["rel_mse"]["ci"][1]
        out["anchor_adequate_H8_better_than_Q4"] = bool(hq < 0)
        out["retained_benefit"] = retained(Mx)
        out.update(run.record())
    jdump("conf.json", out)
    report_contrasts(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "checks", "dev", "dev_k8", "conf"])
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    {"build": build, "checks": checks, "dev": dev_stage, "dev_k8": dev_k8, "conf": conf}[a.cmd](dev)


if __name__ == "__main__":
    main()
