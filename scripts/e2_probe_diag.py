import torch, numpy as np
from boa_dit.common import Run
from boa_dit.f4 import STAGE
from boa_dit.followup import spatial2 as S
from boa_dit.temporal import pixart as PX
from boa_dit.w4a4 import common as WC
dev = "cuda:0"
class Stop(Exception): pass
with Run(STAGE, "e2_probe_diag", dev, {"why": "probe sanity: no-probe re-entrant call and direction spread"}):
    pipe, _ = S.setup(dev); tr = pipe.transformer; e = WC.load_embeds("design")
    S.CTL.mode = "probe"; st = {"busy": False}
    def hook(mod, args, kwargs, out):
        t = PX.STATE.step
        if st["busy"] or t != 2: return
        st["busy"] = True
        o0 = out[0]; B = o0.shape[0] // 2; e0 = S.guided(o0, B)
        def call(p):
            S.CTL.probe, S.CTL.probe_step = p, 2; PX.STATE.step = 2
            o = mod(*args, **kwargs)[0]; S.CTL.probe = None; return S.guided(o, B)
        r = {}
        for p in [None, ("low", 0), ("low", 1), ("high", 0), ("high", 1)]:
            r[str(p)] = call(p)
        for k, v in r.items(): print(k, "rel |Δ|² vs e0", float((v - e0).pow(2).sum() / e0.pow(2).sum()))
        print("low0 vs low1", float((r["('low', 0)"] - r["('low', 1)"]).pow(2).sum() / e0.pow(2).sum()))
        print("low0 vs high0", float((r["('low', 0)"] - r["('high', 0)"]).pow(2).sum() / e0.pow(2).sum()))
        raise Stop()
    h = tr.register_forward_hook(hook, with_kwargs=True); sc = PX.StepCounter(tr)
    def gp(mod, args, _o=sc._pre):
        if not st["busy"]: _o(mod, args)
    tr._forward_pre_hooks[sc.h.id] = gp
    try: PX.sample_lat(pipe, e, [0], dev)
    except Stop: pass
    finally: h.remove(); sc.remove()
