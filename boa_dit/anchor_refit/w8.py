"""w8s1: W8 anchors every two calls (results/w8_spacing/PROTOCOL.md). abn1 harness (activation_bridge.run.sample) unchanged. Ledger tag w8s1.
Stages: checks | run          Run: python -m boa_dit.anchor_refit.w8 checks|run
"""
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts"))
import h1031_budget as HB
from boa_dit.activation_bridge import run as AB
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import NULL_CLASS, labels_for_seeds, make_diffusion, noise_for_seeds
from boa_dit.weight_anchor.run import guided
from boa_dit.weight_anchor.shared import summary

TAG = "w8s1"
OUT = os.path.join(RESULTS, "w8_spacing")
CK = os.path.join(CKPT_DIR, "w8_spacing")
SEEDS = list(range(126000, 126064))
OVN = list(range(110000, 110032))
BS = 32
AB.ARMS["W8-K2"] = AB.A("static", "h8", K=2, bridge=True)
NAMES = {"FP-K4": "CB", "W8-K4": "CB.w8", "W8-K2": "W8-K2", "FP50": "FP50", "H8_50": "H8_50"}


def jdump(n, o):
    atomic_write_json(os.path.join(OUT, n), o)


def guard(name, est):
    ok, info = HB.ok(TAG, est)
    print(f"[w8s1 budget] {name}: {info} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        jdump(f"{name}.json", {"status": "skipped (budget)", **info})
        sys.exit(4)


@torch.no_grad()
def checks(dev):
    guard("checks", 0.08)
    with Run(TAG, "checks", dev, {}) as run:
        ch = {"provenance": AB.provenance()}
        d = make_diffusion(AB.T)
        Ac, bc = AB.coeffs(d)
        M = AB.load_models(dev, ("fp", "h8", "static"))
        sp = AB.ARMS["W8-K2"]
        ch["K2_expected"] = AB.expect(sp)
        ch["K2_anchor_indices"] = list(range(0, AB.T, 2))
        ch["K2_bridge_coef_equals_half_b"] = all(abs(AB.bridge_coef(Ac, bc, a, 2) - 0.5 * bc[a + 1]) <= 1e-15 for a in range(0, 48, 2))
        ch["K1_coef_zero"] = all(AB.bridge_coef(Ac, bc, a, 1) == 0.0 for a in range(AB.T - 1))
        nz, lb = noise_for_seeds(OVN[:4], dev), labels_for_seeds(OVN[:4])
        x, c = AB.sample(M, sp, d, nz, lb)
        ch["K2_counts"] = c
        h8, _ = AB.sample(M, AB.ARMS["H8_50"], d, nz, lb)
        k1, c1 = AB.sample(M, AB.A("static", "h8", K=1, bridge=True), d, nz, lb)
        ch["K1_W8_equals_H8_50_bitwise"] = bool(torch.equal(k1, h8) and c1 == {"cheap": 50, "high": 50, "corrections": 0})
        L1 = torch.load(os.path.join(AB.CK, "s1_latents.pt"), map_location="cpu")
        L2 = torch.load(os.path.join(AB.CK, "s2_latents.pt"), map_location="cpu")
        nz, lb = noise_for_seeds(OVN, dev), labels_for_seeds(OVN)
        ch["reproduce_abn1_bs32"] = {"CB": bool(torch.equal(AB.sample(M, AB.ARMS["CB"], d, nz, lb)[0].cpu(), L1["CB"][:32])),
                                    "CB.w8": bool(torch.equal(AB.sample(M, AB.ARMS["CB.w8"], d, nz, lb)[0].cpu(), L2["CB.w8"][:32]))}
        ch["pass"] = bool(ch["K2_expected"] == {"cheap": 50, "high": 25, "corrections": 24} and c == ch["K2_expected"] and ch["K2_bridge_coef_equals_half_b"]
                          and ch["K1_coef_zero"] and ch["K1_W8_equals_H8_50_bitwise"] and all(ch["reproduce_abn1_bs32"].values()))
        ch.update(run.record())
    jdump("checks.json", ch)
    print(json.dumps({k: v for k, v in ch.items() if k not in ("provenance", "config")}, indent=1), flush=True)
    return 0 if ch["pass"] else 6


def ev_ms(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record(); fn(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e)


@torch.no_grad()
def run_stage(dev):
    ch = json.load(open(os.path.join(OUT, "checks.json")))
    assert ch["pass"], "checks not passed"
    arms = ["FP50", "H8_50", "FP-K4", "W8-K4", "W8-K2"]
    guard("run", 0.25)
    with Run(TAG, "run", dev, {"seeds": [SEEDS[0], SEEDS[-1]], "batch": BS}) as run:
        AB.provenance()
        M = AB.load_models(dev, ("fp", "h8", "static"))
        d = make_diffusion(AB.T)
        lat, cnt, tim = {a: [] for a in arms}, {}, {}
        for b0 in range(0, len(SEEDS), BS):
            s = SEEDS[b0:b0 + BS]
            nz, lb = noise_for_seeds(s, dev), labels_for_seeds(s)
            for a in arms:
                torch.cuda.synchronize(); t0 = time.perf_counter()
                x, c = AB.sample(M, AB.ARMS[NAMES[a]], d, nz, lb)
                torch.cuda.synchronize(); dt = time.perf_counter() - t0
                assert torch.isfinite(x).all() and c == AB.expect(AB.ARMS[NAMES[a]]), (a, c)
                lat[a].append(x.cpu()); cnt[a] = c; tim.setdefault(a, []).append(dt)
                print(f"  seeds {s[0]}-{s[-1]} {a}: {dt:.1f} s {c}", flush=True)
        lat = {a: torch.cat(v) for a, v in lat.items()}
        os.makedirs(CK, exist_ok=True)
        atomic_torch_save(lat, os.path.join(CK, "latents.pt"))
        # per-call simulated cost (bs 32, CFG 64 rows, warm, CUDA events): FP vs H8 vs cheap
        x = noise_for_seeds(SEEDS[:BS], dev)
        lb_ = torch.as_tensor(labels_for_seeds(SEEDS[:BS]), device=dev)
        y = torch.cat([lb_, torch.full_like(lb_, NULL_CLASS)])
        ts = torch.full((BS,), 500, device=dev, dtype=torch.long)
        mods = {"FP": M["fp"], "H8": M["h8"], "cheap": M["static"].m}
        for _ in range(5):
            for m in mods.values():
                guided(m, x, ts, y, BS)
        order = [k for k in mods for _ in range(20)]
        np.random.default_rng(0).shuffle(order)
        per = {k: [] for k in mods}
        for k in order:
            per[k].append(ev_ms(lambda: guided(mods[k], x, ts, y, BS)))
        pc = {k: float(np.median(v)) for k, v in per.items()}
        del M
        torch.cuda.empty_cache()
        pairs = [("W8-K2", "FP-K4"), ("W8-K2", "W8-K4"), ("W8-K4", "FP-K4"), ("FP-K4", "H8_50"), ("W8-K4", "H8_50"), ("W8-K2", "H8_50")]
        Mx, C = AB.evaluate(lat, dev, pairs)
        out = {"status": "complete", "seeds": [SEEDS[0], SEEDS[-1]], "n": len(SEEDS), "batch": BS, "arms": Mx, "summary": summary(Mx), "contrasts": C,
               "readings": {k: AB.reading(C[k]) for k in ("W8-K2_vs_FP-K4", "W8-K2_vs_W8-K4", "W8-K4_vs_FP-K4")}, "calls": cnt,
               "simulated_trajectory_s_per_batch32": {a: {"median": float(np.median(v)), "all": v} for a, v in tim.items()},
               "simulated_per_call_ms_bs32": pc, "simulated_C_W8_over_C_FP": pc["H8"] / pc["FP"],
               "equal_anchor_cost_threshold_C_W8_over_C_FP": 13 / 25,
               "cost_note": "fake-quant simulation (H8 = dequantized W8 weights in fp16 compute; cheap = fake-quant W4A8): NOT kernel costs; no efficiency claim",
               **run.record()}
    jdump("run.json", out)
    AB.show(Mx, C)
    print(json.dumps({"per_call_ms": pc, "C_W8/C_FP": out["simulated_C_W8_over_C_FP"]}), flush=True)
    return 0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    torch.cuda.set_device("cuda:0")
    sys.exit({"checks": checks, "run": run_stage}[sys.argv[1]]("cuda:0") or 0)
