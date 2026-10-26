"""D2 (and factors.json for the uniform W4 prefix): one factor pass along the uniform W4 model.

Block l's inputs come from the uniform W4 model's blocks 0..l-1 (W8 adaLN). Factors are computed on block l with its FP
linears and the W8 adaLN (the state the solver sees). e_t = attn(x; W4 qkv, W4 proj) - attn(x; FP) at the same 520
tokens. For blocks 0, 2, 10, 27: spreads of R and of the diagonal-normalized H_row(fc1), and the fraction of e_t energy
(gated g1bar*e and pre-gate e) in R's top-16 / bottom-16 eigendirections. Every block: eigen summaries -> factors.json.
Full factors of block 2 are saved to ckpt/gate1_factors_b2.pt for the real-layer solver objective test.
Run: python -m boa_dit.twosided.d2 --device cuda:0
"""
import argparse
import copy
import os

import torch

from boa_dit.calib import capture_block0_inputs
from boa_dit.common import CKPT_DIR, RESULTS, Run, atomic_torch_save, atomic_write_json
from boa_dit.model import load_dit, load_quant_state, make_diffusion
from boa_dit.quantize import CALIB_SEEDS
from boa_dit.twosided.factors import FactorAccumulator, attn_weights, diag_normalized, eig_summary, spreads
from boa_dit.twosided.quantize_ts import run_block_cur, update_factors_json

D2_BLOCKS = (0, 2, 10, 27)
W4 = os.path.join(CKPT_DIR, "ckpt_uniform_w4_K1_lam1_brcfg", "model.pt")


def energy_fracs(e, Vecs, k=16):
    """fraction of sum_t ||e_t||^2 in the top-k / bottom-k eigendirections (Vecs ascending, columns)."""
    tot = float(e.pow(2).sum())
    top = float((e @ Vecs[:, -k:]).pow(2).sum()) / tot
    bot = float((e @ Vecs[:, :k]).pow(2).sum()) / tot
    return {"top16": top, "bottom16": bot, "isotropic_16": 16 / e.shape[1]}


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
    out, fsum = {}, {}
    with Run("gate1", "d2_factor_pass_uniform_w4", dev, vars(a)) as run:
        # block-0 inputs: FP trajectory capture, exactly as quantize.py (the W4 model's embedders are FP)
        X, C = capture_block0_inputs(fp.to(dev), diffusion, CALIB_SEEDS, 1.5, dev)
        fp = fp.cpu()
        for l in range(28):
            bq = q.blocks[l]
            bf = copy.deepcopy(fp.blocks[l]).to(dev)
            bf.adaLN_modulation[1].weight.data.copy_(bq.adaLN_modulation[1].weight.data)   # W8 adaLN, as quantized
            assert torch.equal(bf.adaLN_modulation[1].weight.data, bq.adaLN_modulation[1].weight.data)
            facc = FactorAccumulator(bf, l, N, dev, {"fc1", "attn", "tok"}, attn_q=attn_weights(bq.attn))
            try:
                run_block_cur(bf, X, C, S, N, B, accs=[facc])
            finally:
                facc.remove()
            F = facc.finalize(bf.mlp.fc1.weight.data.float(), bf.mlp.fc2.weight.data.float(), bf.attn.proj.weight.data.float())
            del facc
            fs = {k: eig_summary(F[k]) for k in ("H_row_fc1", "C_mean", "G_attn", "R", "H_row_Wo", "H_row_V1", "H_row_V2")}
            fs["gate_energy_ratio_median"] = F["gate_energy_ratio"]
            fsum[l] = fs
            if l in D2_BLOCKS:
                ev, Vecs = torch.linalg.eigh(F["R"])
                e = F["e_tok"]
                g1 = F["g1"][torch.tensor(F["tok_bucket"], device=dev)]
                rec = {"R": spreads(F["R"]), "R_per_bucket": [spreads(F["R_b"][b]) for b in range(B)],
                       "H_row_fc1_diag_normalized": spreads(diag_normalized(F["H_row_fc1"])),
                       "H_row_fc1": spreads(F["H_row_fc1"]), "H_row_Wo": spreads(F["H_row_Wo"]),
                       "err_energy_gated": energy_fracs(g1 * e, Vecs), "err_energy_pregate": energy_fracs(e, Vecs),
                       "n_tokens": int(e.shape[0]), "gate_energy_ratio_median": F["gate_energy_ratio"]}
                out[str(l)] = rec
                print(f"block {l}: R lmax/lmed {rec['R']['lmax_over_lmedian']:.2f} lmax/lmin {rec['R']['lmax_over_lmin']:.2f}; "
                      f"fc1 diag-norm lmax/lmed {rec['H_row_fc1_diag_normalized']['lmax_over_lmedian']:.2f}; "
                      f"err gated top16 {rec['err_energy_gated']['top16']:.3f} bottom16 {rec['err_energy_gated']['bottom16']:.4f}",
                      flush=True)
                if l == 2:
                    atomic_torch_save({k: (v.float().cpu() if torch.is_tensor(v) else v) for k, v in F.items()
                                       if k in ("H_row_fc1", "G_attn", "H_row_V1", "H_row_V2", "H_row_Wo", "R")},
                                      os.path.join(CKPT_DIR, "gate1_factors_b2.pt"))
            del F, bf
            torch.cuda.empty_cache()
            run_block_cur(bq, X, C, S, N, B, write_back=True)                    # W4 block -> next block's inputs
            print(f"block {l} done", flush=True)
        rec = run.record()
    atomic_write_json(os.path.join(RESULTS, "gate1", "d2.json"), {"blocks": out, **rec})
    update_factors_json("d2_uniform_w4", {"source": "uniform W4 prefix (D2 pass), FP linears + W8 adaLN of each block",
                                          "blocks": {str(k): v for k, v in fsum.items()}})


if __name__ == "__main__":
    main()
