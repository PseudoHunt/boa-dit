"""§2: network-sensitivity factor F_l = E_t[J_t^T J_t] (block-l output -> the step's raw 8x32x32 model output) by
random probing, and the leverage of heig-8's block-l error in F_l's eigenbasis. See results/engine/rule.md §2.
Run: python -m boa_dit.engine.sensitivity --device cuda:1
"""
import argparse
import copy
import os

import torch

from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.carry import install_from
from boa_dit.model import load_dit, load_quant_state, make_diffusion

BASE = "deflate_heig_r8_w4"
BLOCKS = (2, 10, 20, 27)
STEPS = (2, 22, 42)                 # middle steps of buckets 0, 4, 8
SEEDS = list(range(1032, 1048))
N_PROBES = 64
D = 1152


def suffix_out(fp, h, c, l):
    x = h
    for blk in fp.blocks[l + 1:]:
        x = blk(x, c)
    return fp.unpatchify(fp.final_layer(x, c))


def probe_F(fp, h, c, l, seed, chunk=8):
    """F = sum_probes sum_t g_t g_t^T / (n_probes * n_tokens), g = J^T r, r ~ N(0, I) on each row's output."""
    F = torch.zeros(D, D, device=h.device, dtype=torch.float64)
    g = torch.Generator(device=h.device).manual_seed(seed)
    n_tok = h.shape[0] * h.shape[1]
    for i in range(0, h.shape[0], chunk):
        hi = h[i:i + chunk].detach().clone().requires_grad_(True)
        with torch.enable_grad():
            out = suffix_out(fp, hi, c[i:i + chunk], l)
            for p in range(N_PROBES):
                r = torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype)
                (gr,) = torch.autograd.grad(out, hi, grad_outputs=r, retain_graph=p < N_PROBES - 1)
                gf = gr.reshape(-1, D).double()
                F += gf.T @ gf
        del out, hi
    return F / (N_PROBES * n_tok)


def metrics(F, e):
    ev, V = torch.linalg.eigh(F)
    e = e.reshape(-1, D).double()
    tot = float(e.pow(2).sum())
    proj = (e @ V).pow(2).sum(0)                                              # energy per eigendirection (ascending)
    rho = float((e @ F * e).sum()) / (tot * float(F.diagonal().sum()) / D)
    return {"lmax_over_lmedian": float(ev[-1] / ev.median()), "lmax_over_lmin": float(ev[-1] / ev[0]),
            "err_top32": float(proj[-32:].sum()) / tot, "err_bottom32": float(proj[:32].sum()) / tot,
            "isotropic_32": 32 / D, "rho": rho, "trace": float(F.diagonal().sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    torch.backends.cuda.matmul.allow_tf32 = False
    diffusion = make_diffusion(50)
    out = {"blocks": {}}
    with Run("engine", "sensitivity", dev, vars(a)) as run:
        fp = load_dit(dev)
        q = load_dit(dev)
        load_quant_state(q, os.path.join(CKPT_DIR, f"ckpt_{BASE}", "model.pt"))
        install_from(q, BASE)
        with torch.no_grad():
            X, C = capture_block0_inputs(fp, diffusion, SEEDS, 1.5, dev)
        N = len(SEEDS)
        Fs = {l: [] for l in BLOCKS}
        for s in STEPS:
            rows = slice(s * 2 * N, (s + 1) * 2 * N)
            x, c = X[rows].float(), C[rows]
            for l in range(max(BLOCKS) + 1):
                if l in BLOCKS:
                    fpb = copy.deepcopy(fp.blocks[l])
                    fpb.adaLN_modulation[1].weight.data.copy_(q.blocks[l].adaLN_modulation[1].weight.data)
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                        e = (q.blocks[l](x, c).float() - fpb(x, c).float())
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    x = q.blocks[l](x, c).float()
                if l in BLOCKS:
                    F = probe_F(fp, x, c, l, seed=1000 * l + s)
                    Fs[l].append((F, e))
                    m = metrics(F, e)
                    out["blocks"].setdefault(str(l), {})[f"step{s}"] = m
                    print(f"block {l} step {s}: spread {m['lmax_over_lmedian']:.1f}, rho {m['rho']:.3f}, "
                          f"err top32 {m['err_top32']:.3f} bottom32 {m['err_bottom32']:.4f}", flush=True)
                    del fpb
            torch.cuda.empty_cache()
        for l in BLOCKS:
            Fbar = sum(F for F, _ in Fs[l]) / len(Fs[l])
            e_all = torch.cat([e.reshape(-1, D) for _, e in Fs[l]])
            out["blocks"][str(l)]["bucket_mean"] = metrics(Fbar, e_all)
            print(f"block {l} mean: {out['blocks'][str(l)]['bucket_mean']}", flush=True)
        n_pass = sum(1 for l in BLOCKS if out["blocks"][str(l)]["bucket_mean"]["rho"] >= 1
                     and out["blocks"][str(l)]["bucket_mean"]["lmax_over_lmedian"] >= 10)
        out["read2a_blocks_passing"] = n_pass
        out["read2a_pass"] = n_pass >= 2
        out["rho_below_1_everywhere"] = all(out["blocks"][str(l)]["bucket_mean"]["rho"] < 1 for l in BLOCKS)
        out.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "engine", "sensitivity.json"), out)


if __name__ == "__main__":
    main()
