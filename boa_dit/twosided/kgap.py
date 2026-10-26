"""Kronecker-gap diagnostic for the block objective of W_o (attn.proj), blocks 2 and 10.

Same pass as D2: block l's inputs come from the uniform W4 prefix, factors from block l's FP linears + W8 adaLN, at the
same 520 sampled tokens (token weights w_t = cfg branch weight / (26 B), sum 1). For a perturbation E of W_o:
  exact      = sum_t w_t || A_t (g1_t * (E x_t)) ||^2   (per-sample gates g1_t, g2_t inside A_t; x_t = proj input)
  kfac_tok   = tr(E Hc_tok E^T Hr_tok), Hc_tok = sum_t w_t x x^T, Hr_tok = sum_t w_t G1 A^T A G1 (same tokens: the
               factorization gap alone)
  kfac_E5    = tr(E H_col E^T H_row(W_o)) with the solver's factors as E5 used them: H_col = full proj Gram (all tokens,
               uniform buckets, cfg), H_row = bucket-mean-gate H_row(W_o)
  kfac_E5gs  = the same with the per-sample-gate H_row(W_o) (the rebuilt Variant 2)
Perturbations: the actual uniform-W4 W_o error, and 64 random quantization-like errors (uniform rounding noise at the
row's W4 step, times log-normal(0, 1) row and column scales, seed k). Reports Spearman(form, exact) over the 65 and the
ratio form / exact. Run: python -m boa_dit.twosided.kgap --device cuda:0
"""
import argparse
import copy
import os

import numpy as np
import torch

from boa_dit import weights as W
from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_write_json
from boa_dit.deflate.quantize_dc import MeanGramAccumulator
from boa_dit.model import load_dit, load_quant_state, make_diffusion
from boa_dit.quantize import CALIB_SEEDS
from boa_dit.twosided.factors import TOK_PER_BUCKET_BRANCH, FactorAccumulator
from boa_dit.twosided.quantize_ts import run_block_cur

BLOCKS = (2, 10)
W4 = os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt")


def perturbations(Wfp, Wq, n=64):
    step = ((Wfp.max(1).values - Wfp.min(1).values) / 15).unsqueeze(1)
    out = [("actual_w4", Wq - Wfp)]
    for k in range(n):
        g = torch.Generator(device="cpu").manual_seed(k)
        u = (torch.rand(Wfp.shape, generator=g) - 0.5).to(Wfp.device) * step
        r = torch.exp(torch.randn(Wfp.shape[0], 1, generator=g)).to(Wfp.device)
        c = torch.exp(torch.randn(1, Wfp.shape[1], generator=g)).to(Wfp.device)
        out.append((f"rand{k}", u * r * c))
    return out


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    q = load_dit(dev)
    load_quant_state(q, W4)
    fp = load_dit("cpu")
    diffusion = make_diffusion(50)
    S, N, B = diffusion.num_timesteps, len(CALIB_SEEDS), 10
    res = {}
    with Run("gate1", "kgap", dev, vars(a)) as run:
        X, C = capture_block0_inputs(fp.to(dev), diffusion, CALIB_SEEDS, 1.5, dev)
        fp = fp.cpu()
        for l in range(max(BLOCKS) + 1):
            bq = q.blocks[l]
            if l in BLOCKS:
                bf = copy.deepcopy(fp.blocks[l]).to(dev)
                bf.adaLN_modulation[1].weight.data.copy_(bq.adaLN_modulation[1].weight.data)
                facc = FactorAccumulator(bf, l, N, dev, {"tok", "projx"}, gates="sample")
                acc = MeanGramAccumulator(bf, dev)
                try:
                    run_block_cur(bf, X, C, S, N, B, accs=[acc, facc])
                finally:
                    facc.remove(); acc.remove()
                means, _ = acc.means()
                del acc
                W1, W2 = bf.mlp.fc1.weight.data.float(), bf.mlp.fc2.weight.data.float()
                Wo = bf.attn.proj.weight.data.float()
                F_gs = facc.finalize(W1, W2, Wo)                                     # per-sample H_row(W_o)
                facc.gates = "bucket"
                F_b = facc.finalize(W1, W2, Wo)                                      # bucket-mean (as E5 used)
                facc.gates = "sample"
                Hc_full = W.combine(means, "attn.proj", [1.0 / B] * B, B, "cfg", 1.5).double()
                wb = facc.wbr
                wt = torch.tensor([wb[br] / (TOK_PER_BUCKET_BRANCH * B) for _, br in facc.meta], device=dev, dtype=torch.float64)
                xs, g1 = facc.tx.double(), facc.tg1.double()
                As = torch.cat([A for _, A in facc.token_A(W1, W2, F_gs)])           # [T, D, D] per-sample gates
                Hc_tok = (xs * wt[:, None]).T @ xs
                Hr_tok = torch.zeros_like(Hc_tok)
                for t0 in range(0, len(wt), 32):
                    Ab = As[t0:t0 + 32].double()
                    GA = Ab * g1[t0:t0 + 32, None, :]                                # A_t diag(g1_t)
                    Hr_tok += torch.einsum("t,tij,tik->jk", wt[t0:t0 + 32], GA, GA)
                forms = {k: [] for k in ("exact", "kfac_tok", "kfac_E5", "kfac_E5gs")}
                names = []
                for name, E in perturbations(Wo, bq.attn.proj.weight.data.float()):
                    E = E.double()
                    v = (xs @ E.T) * g1                                              # [T, D]: g1 * (E x_t)
                    ex = 0.0
                    for t0 in range(0, len(wt), 64):
                        Av = torch.einsum("tij,tj->ti", As[t0:t0 + 64].double(), v[t0:t0 + 64])
                        ex += float((wt[t0:t0 + 64] * Av.pow(2).sum(1)).sum())
                    forms["exact"].append(ex)
                    forms["kfac_tok"].append(float(((E @ Hc_tok @ E.T) * Hr_tok).sum()))
                    forms["kfac_E5"].append(float(((E @ Hc_full @ E.T) * F_b["H_row_Wo"]).sum()))
                    forms["kfac_E5gs"].append(float(((E @ Hc_full @ E.T) * F_gs["H_row_Wo"]).sum()))
                    names.append(name)
                ex = np.array(forms["exact"])
                rb = {"perturbations": names, "forms": forms}
                for k in ("kfac_tok", "kfac_E5", "kfac_E5gs"):
                    f = np.array(forms[k])
                    ratio = f / ex
                    rb[k] = {"spearman_all65": spearman(f, ex), "spearman_random64": spearman(f[1:], ex[1:]),
                             "ratio_actual": float(ratio[0]), "ratio_random_median": float(np.median(ratio[1:])),
                             "ratio_random_min": float(ratio[1:].min()), "ratio_random_max": float(ratio[1:].max())}
                    print(f"block {l} {k}: spearman {rb[k]['spearman_all65']:.3f} (random {rb[k]['spearman_random64']:.3f}); "
                          f"ratio actual {rb[k]['ratio_actual']:.3f}, random median {rb[k]['ratio_random_median']:.3f} "
                          f"[{rb[k]['ratio_random_min']:.3f}, {rb[k]['ratio_random_max']:.3f}]", flush=True)
                res[str(l)] = rb
                del As, facc, bf, F_gs, F_b, means
                torch.cuda.empty_cache()
            run_block_cur(bq, X, C, S, N, B, write_back=True)
        rec = run.record()
    atomic_write_json(os.path.join(RESULTS, "gate1", "kgap.json"), {"blocks": res, **rec})


if __name__ == "__main__":
    main()
