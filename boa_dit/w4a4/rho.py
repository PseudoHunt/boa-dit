"""ρ diagnostic on PixArt-Σ (rule.md): blocks 2, 9, 18, 27; 32 design-cache rows (held-out calibration prompts 0-15 x 2 branches) x steps
2, 10, 18. F_l = E_t[J_t^T J_t] from block-l output to the 8-channel transformer output through the FP suffix, by 64
random probes; e_t = base model block-l output - FP block-l output on FP-reference block inputs.
rho = sum_t e_t^T F e_t / (sum_t ||e_t||^2 tr(F)/1152); spread = lmax/lmedian. Gate: rho >= 1 and spread >= 10 for
the step-mean F in >= 2 of 4 blocks. Run: python -m boa_dit.w4a4.rho --arm hsvd_r4_b0.5 --device cuda:1
"""
import argparse
import copy
import os

import torch

from boa_dit.common import RESULTS, Run, atomic_write_json
from boa_dit.engine.sensitivity import metrics
from boa_dit.w4a4.common import CK, load_embeds, load_pipe
from boa_dit.w4a4.screen import install, text_for

BLOCKS = (2, 9, 18, 27)
STEPS = (2, 10, 18)
N_PROBES, D = 64, 1152


def head(tr, h, emb_t):
    shift, scale = (tr.scale_shift_table[None] + emb_t[:, None]).chunk(2, dim=1)
    x = tr.norm_out(h) * (1 + scale) + shift
    x = tr.proj_out(x)
    p, c = tr.config.patch_size, tr.out_channels
    H = W = int(x.shape[1] ** 0.5)
    x = x.reshape(-1, H, W, p, p, c)
    return torch.einsum("nhwpqc->nchpwq", x).reshape(-1, c, H * p, W * p)


def probe_F(tr, h, kw, emb_t, l, seed, chunk=4):
    F = torch.zeros(D, D, device=h.device, dtype=torch.float64)
    g = torch.Generator(device=h.device).manual_seed(seed)
    for i in range(0, h.shape[0], chunk):
        hi = h[i:i + chunk].detach().float().clone().requires_grad_(True)
        kwi = {k: (v[i:i + chunk] if torch.is_tensor(v) else v) for k, v in kw.items()}
        with torch.enable_grad():
            x = hi
            for blk in tr.transformer_blocks[l + 1:]:
                x = blk(x, **kwi)
            out = head(tr, x, emb_t[i:i + chunk])
            for p in range(N_PROBES):
                r = torch.randn(out.shape, generator=g, device=out.device, dtype=out.dtype)
                (gr,) = torch.autograd.grad(out, hi, grad_outputs=r, retain_graph=p < N_PROBES - 1)
                gf = gr.reshape(-1, D).double()
                F += gf.T @ gf
        del out, hi, x
    return F / (N_PROBES * h.shape[0] * h.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="hsvd_r4_b0.5")
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()
    dev = a.device
    torch.cuda.set_device(dev)
    out = {"prompt_set": "design (held-out calibration prompts 32-63, rule_addendum_design.md)", "arm": a.arm, "blocks": {}}
    with Run("w4a4", f"rho_{a.arm}", dev, vars(a)) as run:
        pipe = load_pipe(dev)
        tr = pipe.transformer.float()                                         # fp32 suffix for the probes
        del pipe.vae
        q = copy.deepcopy(tr).to(torch.bfloat16)
        install(q, a.arm)
        c = torch.load(os.path.join(CK, "design_cache.pt"), map_location="cpu")
        e = load_embeds("design")
        acc = {l: [] for l in BLOCKS}
        for s in STEPS:
            idx = [j for j in range(len(c["t"])) if int(c["prompt"][j]) < 16 and int(c["step"][j]) == s]
            pe, pm = text_for(e, c["prompt"][idx], c["branch"][idx], dev)
            ins, emb = {}, {}
            hooks = [b.register_forward_pre_hook(lambda m, args, kw, l=l: ins.__setitem__(l, (args[0] if args else kw["hidden_states"], kw)),
                                                 with_kwargs=True) for l, b in enumerate(tr.transformer_blocks)]
            hooks.append(tr.adaln_single.register_forward_hook(lambda m, i, o: emb.__setitem__("t", o)))
            with torch.no_grad():
                tr(c["latent"][idx].to(dev).float(), encoder_hidden_states=pe.float(), encoder_attention_mask=pm,
                   timestep=c["t"][idx].to(dev), added_cond_kwargs={"resolution": None, "aspect_ratio": None}, return_dict=False)
            for h in hooks:
                h.remove()
            emb_t = emb["t"][1].float()
            for l in BLOCKS:
                x, kw = ins[l]
                kw = {k: v for k, v in kw.items() if k != "hidden_states"}
                with torch.no_grad():
                    yf = tr.transformer_blocks[l](x, **kw).float()
                    kwb = {k: (v.to(torch.bfloat16) if torch.is_tensor(v) and v.is_floating_point() else v) for k, v in kw.items()}
                    yq = q.transformer_blocks[l](x.to(torch.bfloat16), **kwb).float()
                F = probe_F(tr, yf, kw, emb_t, l, seed=1000 * l + s)
                ev = yq - yf
                acc[l].append((F, ev))
                m = metrics(F, ev)
                out["blocks"].setdefault(str(l), {})[f"step{s}"] = m
                print(f"block {l} step {s}: spread {m['lmax_over_lmedian']:.1f} rho {m['rho']:.3f}", flush=True)
            torch.cuda.empty_cache()
        for l in BLOCKS:
            Fb = sum(F for F, _ in acc[l]) / len(acc[l])
            out["blocks"][str(l)]["step_mean"] = metrics(Fb, torch.cat([ev.reshape(-1, D) for _, ev in acc[l]]))
        n = sum(1 for l in BLOCKS if out["blocks"][str(l)]["step_mean"]["rho"] >= 1
                and out["blocks"][str(l)]["step_mean"]["lmax_over_lmedian"] >= 10)
        out.update({"gate_blocks_passing": n, "gate_pass": n >= 2,
                    "rho_below_1_everywhere": all(out["blocks"][str(l)]["step_mean"]["rho"] < 1 for l in BLOCKS)})
        out.update(run.record())
    atomic_write_json(os.path.join(RESULTS, "w4a4", "rho.json"), out)
    print({k: v for k, v in out.items() if k.startswith(("gate", "rho"))}, flush=True)


if __name__ == "__main__":
    main()
