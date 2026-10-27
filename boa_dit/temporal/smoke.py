"""Smoke gate before the DiT stage-3 arms (rule_feedback.md): W16A8 single table, no dither, seeds 0-1, latent rel. MSE
must be < 0.068 (heig-8's W4 weight-only error). Exit code 0 = pass. Also runs the feedback quantizer on the same
seeds as a functional check. Run: python -m boa_dit.temporal.smoke"""
import os
import sys

import torch

from boa_dit.common import Run, atomic_write_json
from boa_dit.model import labels_for_seeds, load_dit, make_diffusion, noise_for_seeds
from boa_dit.sample import ddim_sample
from boa_dit.temporal.common import OUT, STAGE
from boa_dit.temporal.dit_act import Quantize

GATE = 0.068


@torch.no_grad()
def main():
    dev = "cuda:0"
    s = [0, 1]
    res = {}
    with Run(STAGE, "fb_smoke_gate", dev, {"seeds": s}) as run:
        m, d = load_dit(dev), make_diffusion(50)
        xf = ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5)
        for fb in (False, True):
            q = Quantize(m, "single8", "none", dev, feedback=fb)
            x = ddim_sample(m, d, noise_for_seeds(s, dev), labels_for_seeds(s), 1.5)
            res["feedback" if fb else "single8"] = {"rel_mse": float(((x - xf).flatten(1).pow(2).sum(1) / xf.flatten(1).pow(2).sum(1)).mean()),
                                                    "clip_rate_mean": sum(q.clip_rate()) / 50, "identity": q.ident,
                                                    "clamp_rate": q.clamp_rate() if fb else None}
            q.remove()
        res["gate"] = GATE
        res["pass"] = bool(res["single8"]["rel_mse"] < GATE)
        res.update(run.record())
    atomic_write_json(os.path.join(OUT, "stage3", "smoke_gate.json"), res)
    print(res, flush=True)
    sys.exit(0 if res["pass"] else 3)


if __name__ == "__main__":
    main()
