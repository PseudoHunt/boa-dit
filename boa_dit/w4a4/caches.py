"""W4A4 caches from the bf16 model (rule.md):
  embeds : T5 once for every prompt set (common.embed_all)
  calib  : block-0 inputs for 32 calibration prompts x 20 steps x 2 branches (rows ordered prompt, step, branch;
           branch 0 = uncond) -> ckpt/w4a4/calib_block0.pt {hidden [R,4096,1152] bf16, enc [R,300,1152], temb [R,6912],
           enc_mask [R,1,300] (the bias the blocks receive), meta}
  screen : 64 screening prompts; at steps 0,2,..,18 the transformer's inputs and its output, per branch
           -> ckpt/w4a4/screen_cache.pt {latent [R,4,128,128], t [R], prompt [R], branch [R], step [R], eps [R,8,128,128]}
Run: python -m boa_dit.w4a4.caches --what embeds,calib,screen --device cuda:0
"""
import argparse
import os

import torch

from boa_dit.common import Run, atomic_torch_save
from boa_dit.w4a4.common import CK, embed_all, load_embeds, load_pipe, sample

SCREEN_STEPS = tuple(range(0, 20, 2))


class StepCounter:
    def __init__(self, transformer):
        self.step = -1
        self.h = transformer.register_forward_pre_hook(self._pre, with_kwargs=True)

    def _pre(self, mod, args, kwargs):
        self.step += 1

    def remove(self):
        self.h.remove()


@torch.no_grad()
def calib(dev, bs=8):
    path = os.path.join(CK, "calib_block0.pt")
    if os.path.exists(path):
        return
    e = load_embeds("calib")
    n = len(e["prompts"])
    pipe = load_pipe(dev)
    store = {k: [] for k in ("hidden", "enc", "temb", "enc_mask")}
    meta = []
    for i in range(0, n, bs):
        idx = list(range(i, min(n, i + bs)))
        sc = StepCounter(pipe.transformer)

        def pre(mod, args, kwargs, idx=idx, sc=sc):
            hs = args[0] if args else kwargs["hidden_states"]
            B = hs.shape[0] // 2
            store["hidden"].append(hs.to(torch.bfloat16).cpu())
            store["enc"].append(kwargs["encoder_hidden_states"].to(torch.bfloat16).cpu())
            store["temb"].append(kwargs["timestep"].to(torch.bfloat16).cpu())
            store["enc_mask"].append(kwargs["encoder_attention_mask"].cpu())
            meta.extend([(idx[r % B], sc.step, r // B) for r in range(2 * B)])
        h = pipe.transformer.transformer_blocks[0].register_forward_pre_hook(pre, with_kwargs=True)
        try:
            sample(pipe, e, idx, dev)
        finally:
            h.remove(); sc.remove()
        assert sc.step == 19, sc.step
        print(f"calib prompts {idx[0]}-{idx[-1]} captured", flush=True)
    out = {k: torch.cat(v) for k, v in store.items()}
    m = torch.tensor(meta)
    order = torch.argsort(m[:, 0] * 1000 + m[:, 1] * 2 + m[:, 2])          # (prompt, step, branch)
    out = {k: v[order] for k, v in out.items()}
    out["meta"] = m[order]
    atomic_torch_save(out, path)
    print("calib cache:", {k: tuple(v.shape) for k, v in out.items()}, flush=True)


@torch.no_grad()
def screen(dev, bs=8):
    path = os.path.join(CK, "screen_cache.pt")
    if os.path.exists(path):
        return
    e = load_embeds("screen")
    n = len(e["prompts"])
    pipe = load_pipe(dev)
    rec = {k: [] for k in ("latent", "t", "prompt", "branch", "step", "eps")}
    for i in range(0, n, bs):
        idx = list(range(i, min(n, i + bs)))
        sc = StepCounter(pipe.transformer)

        def hook(mod, args, kwargs, out, idx=idx, sc=sc):
            if sc.step not in SCREEN_STEPS:
                return
            hs = args[0] if args else kwargs["hidden_states"]
            B = hs.shape[0] // 2
            o = out[0] if isinstance(out, tuple) else out.sample
            rec["latent"].append(hs.to(torch.bfloat16).cpu())
            rec["t"].append(kwargs["timestep"].float().cpu())
            rec["eps"].append(o.to(torch.bfloat16).cpu())
            rec["prompt"].append(torch.tensor([idx[r % B] for r in range(2 * B)]))
            rec["branch"].append(torch.tensor([r // B for r in range(2 * B)]))
            rec["step"].append(torch.full((2 * B,), sc.step))
        h = pipe.transformer.register_forward_hook(hook, with_kwargs=True)
        try:
            sample(pipe, e, idx, dev)
        finally:
            h.remove(); sc.remove()
        print(f"screen prompts {idx[0]}-{idx[-1]} cached", flush=True)
    out = {k: torch.cat(v) for k, v in rec.items()}
    order = torch.argsort(out["prompt"] * 1000 + out["step"] * 2 + out["branch"])
    out = {k: v[order] for k, v in out.items()}
    atomic_torch_save(out, path)
    print("screen cache:", {k: tuple(v.shape) for k, v in out.items()}, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="embeds,calib,screen")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.cuda.set_device(a.device)
    with Run("w4a4", f"caches_{a.what.replace(',', '+')}", a.device, vars(a)):
        for w in a.what.split(","):
            {"embeds": embed_all, "calib": calib, "screen": screen}[w](a.device)


if __name__ == "__main__":
    main()
