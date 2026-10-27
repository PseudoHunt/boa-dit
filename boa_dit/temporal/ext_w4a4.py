"""Extension E1 (rule_extension.md): temporal feedback on the strong W4A4 baseline, PixArt-Σ.
Base: corrected H-SVDQuant r4 (`hsvd_r4_b0.5_kv16_u2_rs`): W4 groups of 64 + 16-bit low-rank branch, dynamic per-token
INT4 activations in groups of 64 on X̃ = X/d (post-GELU fc2 input: shifted unsigned), cross-attn K/V 16-bit.
  build                         : block-0 calibration cache + the unchanged w4a4 builder (logged under stage temporal)
  arm --mode base|feedback|unif : MJHQ-n images + latents (batch 1), PSNR / LPIPS / latent vs bf16 batch-1 refs, design C/I
      [--w16] conditional diagnostic: FP weights (W·d folded), same activation quantizer, no branch
      [--set screen|confirm]      screen = mjhq1k[:64], confirm = mjhq1k[64:320]
Feedback (per site, per sample and CFG row, reset at step 0): u = x̃ + s; grid from u (as the base computes it from its
input); q = Q_A4(u); q̂ = bf16(q); r⁰ = u - q̂; s = fp16(clamp(r⁰, ±Δ_g/2)). Dynamic max-based grids do not clip
(signed); the unsigned fc2 path clips at its zero bound and that clip is never fed back.
Run: python -m boa_dit.temporal.ext_w4a4 build | arm --mode feedback --set screen
"""
import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

from boa_dit.common import Run, atomic_torch_save, atomic_write_json
from boa_dit.temporal import pixart as PX
from boa_dit.temporal.common import OUT, STAGE
from boa_dit.w4a4 import common as WC
from boa_dit.w4a4.fake import GS, QMAX, QMIN, QLinear, set_module

BASE = "hsvd_r4_b0.5_kv16_u2_rs"
EXT = os.path.join(OUT, "ext")
SETS = {"screen": (0, 64), "confirm": (64, 320)}


def _groups(x):
    sh = x.shape
    return x.reshape(*sh[:-1], sh[-1] // GS, GS), sh


class QLinearFB(QLinear):
    """QLinear (fake.py) with mode 'base' (unchanged), 'feedback' or 'unif' on its activation quantizer."""

    def __init__(self, *a, mode="base", site=0, **k):
        super().__init__(*a, **k)
        self.mode, self.site, self.s = mode, site, None
        self.stats = torch.zeros(4, dtype=torch.float64)                     # clipped, clamped, elements, clip energy

    def _quant(self, u, o=None):
        """dynamic group quantizer on u (signed INT4, or shifted unsigned); optional subtractive offset o (in units of Δ_g)."""
        if self.act_shift is not None:
            uu = (u + self.act_shift)
            uc = uu.clamp_min(0)
            g, sh = _groups(uc)
            sc = g.amax(-1, keepdim=True).clamp_min(1e-8) / 15.0
            q = torch.clamp(torch.round(g / sc), 0, 15) * sc
            return (q.reshape(sh) - self.act_shift), (uc - self.act_shift), sc, sh
        g, sh = _groups(u)
        sc = g.abs().amax(-1, keepdim=True).clamp_min(1e-8) / QMAX
        if o is None:
            q = torch.clamp(torch.round(g / sc), QMIN, QMAX) * sc
        else:
            oo = _groups(o)[0] * sc
            q = torch.clamp(torch.round((g + oo) / sc), QMIN, QMAX) * sc - oo
        return q.reshape(sh), u, sc, sh

    def forward(self, x):
        if self.mode == "base":
            return super().forward(x)
        dt = x.dtype
        xs = x.float() if self.d is None else x.float() / self.d
        t = PX.STATE.step
        dev = xs.device
        if self.stats.device != dev:
            self.stats = self.stats.to(dev)
        if self.mode == "unif":
            gen = torch.Generator(device=dev).manual_seed(100003 * self.site + t)
            o = torch.rand(xs.shape, generator=gen, device=dev) - 0.5
            q, _, sc, sh = self._quant(xs, o)
            g = _groups(xs)[0]
            clipped = ((g / sc + _groups(o)[0]).round().clamp(QMIN, QMAX) != (g / sc + _groups(o)[0]).round()).sum()
            self.stats += torch.stack([clipped.double(), torch.zeros((), device=dev, dtype=torch.float64),
                                       torch.tensor(float(xs.numel()), device=dev, dtype=torch.float64), torch.zeros((), device=dev, dtype=torch.float64)])
            xq = q
        else:                                                                 # feedback
            if t == 0 or self.s is None or self.s.shape != xs.shape:
                self.s = torch.zeros_like(xs, dtype=torch.float16)
            u = xs + self.s.float()
            q, uc, sc, sh = self._quant(u)
            qh = q.to(torch.bfloat16).float()
            r0 = uc - qh
            half = (sc / 2).expand(_groups(r0)[0].shape).reshape(sh)
            r = torch.maximum(torch.minimum(r0, half), -half)
            self.stats += torch.stack([(uc != u).sum().double(), (r != r0).sum().double(),
                                       torch.tensor(float(u.numel()), device=dev, dtype=torch.float64), (uc - u).double().pow(2).sum()])
            self.s = r.to(torch.float16)
            xq = qh
        y = F.linear(xq.to(torch.bfloat16), self.R, self.bias)
        if self.L1 is not None and self.L1.shape[1] > 0:
            y = y + (xs.to(torch.bfloat16) @ self.L1) @ self.L2
        return y.to(dt)


def install(tr, mode, w16=False):
    qs = torch.load(os.path.join(WC.CK, f"q_{BASE}.pt"), map_location="cpu")
    dev = next(tr.parameters()).device
    mods = []
    for i, (key, q) in enumerate(sorted(qs.items())):
        l, ln = key.split(".", 1)
        blk = tr.transformer_blocks[int(l)]
        lin = WC.block_linear(blk, ln)
        g = lambda k: None if q.get(k) is None else q[k].to(dev)
        d = g("d")
        if w16:                                                               # FP weights in the X̃ space, no branch
            R = lin.weight.float() * (d[None, :] if d is not None else 1.0)
            m = QLinearFB(lin, R, None, None, d, act_shift=g("act_shift"), mode=mode, site=i)
        else:
            m = QLinearFB(lin, g("R"), g("L1"), g("L2"), d, act_shift=g("act_shift"), mode=mode, site=i)
        set_module(blk, ln, m.to(dev))
        mods.append(m)
    return mods


@torch.no_grad()
def gen(name, pipe, dev, lo, hi, pilot=True):
    """mjhq1k[lo:hi] at batch 1 (seed = prompt index), chunk files of 8 under PX.img_dir(name). Pilot: time the first new
    image, project the rest (x1.15 for diagnostics and scoring) and exit 4 if it would cross the ceiling (env TEMPORAL_CAP)."""
    import time
    from boa_dit.temporal.decide import CAP, spent
    d = PX.img_dir(name)
    os.makedirs(d, exist_ok=True)
    e = WC.load_embeds("mjhq1k")
    todo = sum(1 for i in range(lo, hi, PX.CHUNK) if not os.path.exists(os.path.join(d, f"{i:05d}.pt"))) * PX.CHUNK
    t_first = None
    for i in range(lo, hi, PX.CHUNK):
        f = os.path.join(d, f"{i:05d}.pt")
        if os.path.exists(f):
            continue
        imgs, lats = [], []
        for j in range(i, min(hi, i + PX.CHUNK)):
            torch.cuda.synchronize(); t0 = time.time()
            sc = PX.StepCounter(pipe.transformer)
            try:
                lat = PX.sample_lat(pipe, e, [j], dev)
            finally:
                sc.remove()
            imgs.append(PX.decode(pipe, lat)); lats.append(lat.float().cpu())
            if pilot and t_first is None:
                torch.cuda.synchronize(); t_first = time.time() - t0
                proj = t_first * todo * 1.15 / 3600
                print(f"  pilot {name}: {t_first:.1f} s/image -> projected {proj:.3f} h for {todo} images", flush=True)
                if spent() + proj > CAP:
                    print("  ABORT: projected cost exceeds the extension ceiling", flush=True)
                    sys.exit(4)
        atomic_torch_save({"img": torch.cat(imgs), "lat": torch.cat(lats)}, f)


def load(name, lo, hi):
    ch = [torch.load(os.path.join(PX.img_dir(name), f"{i:05d}.pt")) for i in range(lo, hi, PX.CHUNK)]
    return torch.cat([c["img"] for c in ch]), torch.cat([c["lat"] for c in ch])


@torch.no_grad()
def arm(mode, w16, setname, dev):
    lo, hi = SETS[setname]
    name = f"e1_{'w16a4' if w16 else 'w4a4'}_{mode}_{setname}"
    os.makedirs(EXT, exist_ok=True)
    with Run(STAGE, f"ext_{name}", dev, {"mode": mode, "w16": w16, "set": setname}) as run:
        ref = "bf16_b1" if setname == "screen" else "bf16_b1_confirm"           # batch-1 bf16 references
        if not os.path.exists(os.path.join(PX.img_dir(ref), f"{(hi - 1) // PX.CHUNK * PX.CHUNK:05d}.pt")):
            gen(ref, WC.load_pipe(dev), dev, lo, hi)
        pipe = WC.load_pipe(dev)
        mods = install(pipe.transformer, mode, w16)
        gen(name, pipe, dev, lo, hi)
        st = sum(m.stats.cpu() for m in mods)
        diag = PX.net_diag_seq(pipe.transformer, dev, "static") if setname == "screen" else None
        del pipe
        torch.cuda.empty_cache()
        import lpips
        net = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
        ri, rl = load(ref, lo, hi)
        qi, ql = load(name, lo, hi)
        psnr, lp = [], []
        for i in range(len(ri)):
            a, b = ri[i:i + 1].to(dev).float(), qi[i:i + 1].to(dev).float()
            psnr.append(float(10 * torch.log10(255.0 ** 2 / (a - b).pow(2).mean().clamp_min(1e-10))))
            lp.append(float(net(a / 127.5 - 1, b / 127.5 - 1).flatten()[0]))
        rel = ((ql - rl).flatten(1).pow(2).sum(1) / rl.flatten(1).pow(2).sum(1)).tolist()
        res = {"arm": name, "mode": mode, "w16": w16, "set": setname, "prompts": [lo, hi], "batch": 1,
               "psnr_mean": float(np.mean(psnr)), "lpips_mean": float(np.mean(lp)), "latent_rel_mse_mean": float(np.mean(rel)),
               "psnr_per_image": psnr, "lpips_per_image": lp, "latent_rel_mse_per_image": rel, "diag": diag,
               "clip_rate": float(st[0] / max(st[2], 1)), "clamp_rate": float(st[1] / max(st[2], 1)), "clip_energy": float(st[3])}
        res.update(run.record())
    atomic_write_json(os.path.join(EXT, f"{name}.json"), res)
    print({k: v for k, v in res.items() if not isinstance(v, (list, dict))}, flush=True)


def build(dev):
    """Block-0 calibration cache + the unchanged builder for the corrected base."""
    from boa_dit.w4a4 import caches as C, quantize as Q
    tag = lambda stage, name, d, cfg: Run(STAGE, f"ext_e1_{name}", d, cfg)
    C.Run = Q.Run = tag
    if not os.path.exists(os.path.join(WC.CK, "calib_block0.pt")):
        sys.argv = [sys.argv[0], "--what", "calib", "--device", dev]
        C.main()
    if not os.path.exists(os.path.join(WC.CK, f"q_{BASE}.pt")):
        sys.argv = [sys.argv[0], "--method", "hsvd", "--r", "4", "--kv16", "--fc2_unsigned", "--range_search", "--device", dev]
        Q.main()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "arm"])
    ap.add_argument("--mode", default="base", choices=["base", "feedback", "unif"])
    ap.add_argument("--w16", action="store_true")
    ap.add_argument("--set", default="screen", choices=list(SETS))
    a = ap.parse_args()
    dev = "cuda:0"
    torch.cuda.set_device(dev)
    build(dev) if a.cmd == "build" else arm(a.mode, a.w16, a.set, dev)


if __name__ == "__main__":
    main()
