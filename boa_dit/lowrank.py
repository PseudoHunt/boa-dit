"""Gate A: time-indexed low-rank residual correction on top of a quantized checkpoint.

build : for each linear and each regime Gram H (damped), M = E H^1/2, SVD -> L = U_k S_k, R = V_k^T H^-1/2,
        stored for k_max (rank-k arms take the leading k components of the same SVD).
eval  : y = Q(W) x + L_r (R_r x), r = regime of the current step's bucket; latent MSE on the 512 eval seeds.
"""
import argparse
import json
import os

import numpy as np
import torch

from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.eval_latent_mse import EVAL_SEEDS, fp_reference, per_image
from boa_dit.fp_grams import load_block_grams
from boa_dit.model import LINEARS, block_linear, decode, load_dit, load_quant_state, load_vae, make_diffusion
from boa_dit.sample import CTX, sample_seeds

T3 = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs",
                                 "time3_split.json")))["bucket_to_regime"]
REGIMES = {"static": [0] * 10, "time2": [0] * 5 + [1] * 5, "time3": T3}
ARMS = {"time2-16": ("time2", 16), "static-32": ("static", 32), "static-16": ("static", 16),
        "time2-32": ("time2", 32), "time3-16": ("time3", 16), "static-48": ("static", 48), "static-8": ("static", 8)}
ORDER = ["time2-16", "static-32", "static-16", "time2-32", "time3-16", "static-48"]
KMAX = 48
OUT = os.path.join(RESULTS, "stageA")


def base_name(bits):
    return f"uniform_w{bits}_K1_lam1_brcfg"


def corr_path(bits, regimes=None):
    return os.path.join(CKPT_DIR, f"lowrank_w{bits}.pt" if regimes is None else f"lowrank_w{bits}_{'_'.join(regimes)}.pt")


@torch.no_grad()
def factor(E, H, k, dev):
    """L [d_out, k], R [k, d_in] minimizing ||(E - L R) H^1/2||_F, H GPTQ-damped. fp64 on GPU, CPU fallback."""
    H = H.double().to(dev)
    H = H + 0.01 * H.diagonal().mean() * torch.eye(H.shape[0], device=dev, dtype=H.dtype)
    try:
        s, V = torch.linalg.eigh(H)
    except RuntimeError:
        s, V = torch.linalg.eigh(H.cpu())
        s, V = s.to(dev), V.to(dev)
    s = s.clamp_min(s.max() * 1e-12)
    Hh = (V * s.sqrt()) @ V.T
    Hmh = (V / s.sqrt()) @ V.T
    M = E.double().to(dev) @ Hh
    U, S, Vt = torch.linalg.svd(M, full_matrices=False)
    L = U[:, :k] * S[:k]
    R = Vt[:k] @ Hmh
    # diagnostic: fraction of the weighted residual energy captured at each rank
    cap = (S[:k] ** 2).cumsum(0) / (S ** 2).sum()
    return L.float().cpu(), R.float().cpu(), cap.float().cpu()


@torch.no_grad()
def build(bits, dev, regimes=None):
    """regimes=None: all of REGIMES (Gate A); else only those (e.g. ['static'] for the G2 static-8 baseline)."""
    path = corr_path(bits, regimes)
    if os.path.exists(path):
        return torch.load(path, map_location="cpu")
    W_fp = load_dit("cpu")
    Q = load_dit("cpu")
    load_quant_state(Q, os.path.join(CKPT_DIR, f"ckpt_{base_name(bits)}", "model.pt"))
    corr, captured = {}, {}
    for l in range(28):
        G = load_block_grams(l)
        for ln in LINEARS:
            E = block_linear(W_fp.blocks[l], ln).weight.data - block_linear(Q.blocks[l], ln).weight.data
            for rname, b2r in REGIMES.items():
                if regimes is not None and rname not in regimes:
                    continue
                for r in sorted(set(b2r)):
                    H = G[ln][[b for b in range(10) if b2r[b] == r]].mean(0)
                    L, R, cap = factor(E, H, KMAX, dev)
                    corr[(l, ln, rname, r)] = (L, R)
                    captured[f"b{l}.{ln}.{rname}.{r}"] = cap[[15, 31, 47]].tolist()
        del G
        print(f"block {l} factored", flush=True)
    atomic_torch_save(corr, path)
    atomic_write_json(os.path.join(OUT, f"captured_energy_w{bits}{'' if regimes is None else '_' + '_'.join(regimes)}.json"),
                      {"note": "fraction of ||E H^1/2||_F^2 captured at rank 16/32/48", "per_layer_regime": captured})
    return corr


def extra_bits(rname, k):
    n_reg = len(set(REGIMES[rname]))
    dims = {"attn.qkv": (3456, 1152), "attn.proj": (1152, 1152), "mlp.fc1": (4608, 1152), "mlp.fc2": (1152, 4608)}
    extra = 28 * n_reg * k * sum(o + i for o, i in dims.values())
    base = 28 * sum(o * i for o, i in dims.values())
    return 16.0 * extra / base


class Corrector:
    """Forward hooks adding L_r (R_r x) to each block linear, r picked from CTX.step."""

    def __init__(self, model, corr, rname, k, dev):
        self.b2r = REGIMES[rname]
        self.hooks = []
        for l, blk in enumerate(model.blocks):
            for ln in LINEARS:
                mats = {r: (corr[(l, ln, rname, r)][0][:, :k].to(dev, torch.float16),
                            corr[(l, ln, rname, r)][1][:k].to(dev, torch.float16)) for r in sorted(set(self.b2r))}
                self.hooks.append(block_linear(blk, ln).register_forward_hook(self._hook(mats)))

    def _hook(self, mats):
        def h(mod, inp, out):
            L, R = mats[self.b2r[CTX.step // 5]]
            x = inp[0].to(torch.float16)
            return out + ((x @ R.T) @ L.T).to(out.dtype)
        return h

    def remove(self):
        for h in self.hooks:
            h.remove()


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bits", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ORDER))
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--regimes", default=None, help="comma list: build only these regimes (e.g. static)")
    ap.add_argument("--fid", action="store_true", help="also FID-2k (seeds 0-1999) for each arm, grid of seeds 0-15")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    regimes = a.regimes.split(",") if a.regimes else None
    with Run("gateA", f"lowrank_build_w{a.bits}", dev, vars(a)):
        corr = build(a.bits, dev, regimes)
    xf, pf = fp_reference(dev)
    model, diffusion = load_dit(dev), make_diffusion(50)
    load_quant_state(model, os.path.join(CKPT_DIR, f"ckpt_{base_name(a.bits)}", "model.pt"))
    for arm in a.arms.split(","):
        out_json = os.path.join(OUT, "eval", f"latent_{arm}_w{a.bits}.json")
        if os.path.exists(out_json):
            continue
        rname, k = ARMS[arm]
        with Run("gateA", f"eval_{arm}_w{a.bits}", dev, {"arm": arm, "bits": a.bits}) as run:
            c = Corrector(model, corr, rname, k, dev)
            try:
                part = os.path.join(CKPT_DIR, f"lowrank_{arm}_w{a.bits}.partial")
                xq = sample_seeds(model, diffusion, EVAL_SEEDS, 1.5, 32, partial=part, resume=True, log=False)
            finally:
                c.remove()
            rel, mse, psnr = per_image(xq, xf)
            res = {"arm": arm, "bits": a.bits, "regimes": rname, "rank": k, "base": base_name(a.bits),
                   "extra_bits_per_param": extra_bits(rname, k), "n": len(rel),
                   "rel_mse_mean": float(rel.mean()), "rel_mse_sem": float(rel.std() / np.sqrt(len(rel))),
                   "latent_psnr_mean": float(psnr.mean()), "rel_mse_per_image": rel.tolist(), **run.record()}
        atomic_write_json(out_json, res)
        os.remove(part)
        print(f"{arm} W{a.bits}: rel MSE {res['rel_mse_mean']:.5f}", flush=True)
    if a.fid:
        for arm in a.arms.split(","):
            fid_arm(model, diffusion, corr, arm, a.bits, dev)


@torch.no_grad()
def fid_arm(model, diffusion, corr, arm, bits, dev):
    """FID-2k on seeds 0-1999 with the correction attached (same protocol as eval_fid); grid of seeds 0-15."""
    import numpy as np
    from boa_dit.eval_fid import REF_STATS, fid_from_stats, inception_stats
    out_json = os.path.join(OUT, "eval", f"fid2000_{arm}_w{bits}.json")
    if os.path.exists(out_json):
        return
    rname, k = ARMS[arm]
    with Run("gateA", f"fid2000_{arm}_w{bits}", dev, {"arm": arm, "bits": bits}) as run:
        c = Corrector(model, corr, rname, k, dev)
        part = os.path.join(CKPT_DIR, f"lowrank_{arm}_w{bits}_fid2k.partial")
        try:
            x = sample_seeds(model, diffusion, list(range(2000)), 1.5, 32, partial=part, resume=True, log=False)
        finally:
            c.remove()
        vae = load_vae(dev)
        imgs = decode(vae, x.to(dev))
        os.makedirs(os.path.join(CKPT_DIR, "deflate_grids"), exist_ok=True)
        atomic_torch_save(imgs[:16].clone(), os.path.join(CKPT_DIR, "deflate_grids", f"{arm}_w{bits}.pt"))
        mu, sig = inception_stats(imgs.permute(0, 2, 3, 1).numpy(), dev)
        ref = np.load(REF_STATS)
        res = {"arm": arm, "bits": bits, "n": 2000, "fid": fid_from_stats(mu, sig, ref["mu"], ref["sigma"]),
               "extra_bits_per_param": extra_bits(rname, k), **run.record()}
    atomic_write_json(out_json, res)
    os.remove(part)
    print(f"{arm} W{bits}: FID-2k {res['fid']:.2f}", flush=True)


if __name__ == "__main__":
    main()
