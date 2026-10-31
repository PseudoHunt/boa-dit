"""snk1: real-kernel SANA anchoring feasibility on the A30 (results/sana_a30/PROTOCOL.md).
Run with the nk1 venv:  /home/nk_env/bin/python scripts/sana_a30.py env|smoke|kernels|timing
"""
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "results", "sana_a30")
CK = os.path.join(ROOT, "ckpt", "sana_a30")
LEDGER = os.path.join(ROOT, "results", "gpu_hours.csv")
TAG = "snk1"
CAP_H = 1.5
HUB = "/home/.cache/huggingface/hub"
BASE = f"{HUB}/models--Efficient-Large-Model--Sana_1600M_1024px_BF16_diffusers/snapshots/e2b3c0cbffebcd09d83805e88b9f5f106afc74ac"
INT4 = f"{HUB}/models--nunchaku-ai--nunchaku-sana/snapshots/437e41b979ec875f63c18e67d55373aa5230c308/svdq-int4_r32-sana1.6b.safetensors"
MJHQ = f"{HUB}/datasets--playgroundai--MJHQ-30K/snapshots/15b0a659e066e763d0e9a6cd8f00e25f8af5e084/meta_data.json"
STEPS, GUIDE, RES = 20, 4.5, 1024
TIMING = [358, 352, 361, 359]
DEVP = list(range(352, 368))
CONFP = list(range(464, 528))
SCHED = {"K4": [0, 4, 8, 12, 16], "K8": [0, 8, 16]}


# ------------------------------------------------------------------------------------------------------------------- ledger
class Run:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t0 = time.time()
        self.start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return self

    def hours(self):
        return (time.time() - self.t0) / 3600

    def __exit__(self, et, *a):
        with open(LEDGER, "a") as f:
            csv.writer(f).writerow([TAG, self.name + ("" if et is None else "(FAILED)"), "cuda:0", self.start,
                                    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), f"{self.hours():.4f}"])
        return False


def spent():
    return sum(float(r[-1]) for r in csv.reader(open(LEDGER)) if r and r[0] == TAG) if os.path.exists(LEDGER) else 0.0


def guard(name, est):
    s = spent()
    ok = s + est <= CAP_H
    print(f"[snk1 budget] {name}: spent {s:.3f} + est {est:.2f} vs cap {CAP_H} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        jdump(f"{name}.json", {"status": "skipped (budget)", "spent_h": s, "estimate_h": est})
        sys.exit(4)


def jdump(n, o):
    os.makedirs(OUT, exist_ok=True)
    json.dump(o, open(os.path.join(OUT, n), "w"), indent=1, default=str)


def jload(n):
    p = os.path.join(OUT, n)
    return json.load(open(p)) if os.path.exists(p) else None


def prompts():
    import random
    meta = json.load(open(MJHQ))
    keys = sorted(meta)
    screen = random.Random(0).sample(keys, 64)
    rest = sorted(set(keys) - set(screen))
    img = random.Random(1).sample(rest, 1000)
    return [meta[k]["prompt"] for k in img]                                   # == boa_dit.w4a4.common.mjhq_sets()["mjhq1k"]


# ------------------------------------------------------------------------------------------------------------------- models
class Router(nn.Module):
    """installed as pipe.transformer; the native pipeline loop / CFG / scheduler are untouched. mode:
    'H' | 'Q' : single model;  'HYB': Q every call + H at anchors, returns H at anchors (timing control);
    'A': H at anchors, Q between (no Q at anchors);  'R': Q every call, H at anchors, d = Q - H held (beta 1);
    'S': H at anchors, stale H between;  'R0': like R but the held d is not applied (reuse disabled, same routing)."""

    def __init__(self, H, Q):
        super().__init__()
        self.H, self.Q = H, Q
        self.mode, self.anchors = "H", set()
        self.reset()

    @property
    def config(self):
        return self.H.config

    @property
    def dtype(self):
        return torch.bfloat16

    @property
    def device(self):
        return torch.device("cuda")

    def reset(self):
        self.i, self.d, self.stale = 0, None, None
        self.calls = {"H": 0, "Q": 0}
        self.trace = []

    def _h(self, *a, **k):
        self.calls["H"] += 1
        return self.H(*a, **k)[0]

    def _q(self, *a, **k):
        self.calls["Q"] += 1
        return self.Q(*a, **k)[0]

    def forward(self, *a, **k):
        i, m = self.i, self.mode
        anc = i in self.anchors
        if m == "H":
            out = self._h(*a, **k)
        elif m == "Q":
            out = self._q(*a, **k)
        elif m == "HYB":
            q = self._q(*a, **k)
            out = self._h(*a, **k) if anc else q
        elif m == "A":
            out = self._h(*a, **k) if anc else self._q(*a, **k)
        elif m == "S":
            if anc:
                self.stale = self._h(*a, **k)
            out = self.stale
        elif m in ("R", "R0"):
            q = self._q(*a, **k)
            if anc:
                h = self._h(*a, **k)
                self.d = q.float() - h.float()
                out = h
            else:
                out = q if (self.d is None or m == "R0") else (q.float() - self.d).to(q.dtype)
        else:
            raise ValueError(m)
        self.i += 1
        return (out,)


def load(dev="cuda"):
    from diffusers import SanaPipeline
    from nunchaku import NunchakuSanaTransformer2DModel
    t0 = time.perf_counter()
    Q = NunchakuSanaTransformer2DModel.from_pretrained(INT4)
    tq = time.perf_counter() - t0
    pipe = SanaPipeline.from_pretrained(BASE, variant="bf16", torch_dtype=torch.bfloat16).to(dev)
    pipe.vae.to(torch.bfloat16)
    pipe.text_encoder.to(torch.bfloat16)
    H = pipe.transformer
    R = Router(H, Q)
    pipe.transformer = R
    pipe.set_progress_bar_config(disable=True)
    return pipe, R, {"load_int4_s": tq, "load_total_s": time.perf_counter() - t0}


@torch.no_grad()
def embed(pipe, p):
    import inspect
    d = inspect.signature(type(pipe).__call__).parameters                      # the pipeline's own __call__ defaults (complex human instruction etc.)
    kw = {k: d[k].default for k in ("complex_human_instruction", "max_sequence_length", "clean_caption")}
    pe, pm, ne, nm = pipe.encode_prompt(p, True, negative_prompt="", num_images_per_prompt=1, device="cuda", **kw)
    return {"prompt_embeds": pe, "prompt_attention_mask": pm, "negative_prompt_embeds": ne, "negative_prompt_attention_mask": nm}


@torch.no_grad()
def run(pipe, R, mode, emb, seed, anchors=(), steps=STEPS, out="latent", sched_state=None):
    R.mode, R.anchors = mode, set(anchors)
    R.reset()
    r = pipe(**emb, negative_prompt=None, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=steps, generator=torch.Generator().manual_seed(seed),
             output_type=out)
    return r.images, dict(R.calls)


def rel(x, y):
    return float((x.double() - y.double()).norm() / y.double().norm())


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------------- stages
def env():
    q = lambda f: subprocess.run(["nvidia-smi", f"--query-gpu={f}", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    import diffusers, transformers, importlib.metadata as md
    out = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "gpu": {f: q(f) for f in ("name", "compute_cap", "driver_version", "memory.total", "memory.free", "mig.mode.current", "power.limit",
                                     "power.default_limit", "clocks.max.sm", "clocks.max.mem", "persistence_mode")},
           "torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(), "diffusers": diffusers.__version__,
           "transformers": transformers.__version__, "nunchaku": md.version("nunchaku"), "nunchaku_tag": "v1.2.1 (commit af49f24911b49147c074ed96c324fb66884d1f87)",
           "wheel": {"path": "/home/nk_wheels/nunchaku-1.2.1+cu13.0torch2.11-cp310-cp310-linux_x86_64.whl",
                     "sha256": sha("/home/nk_wheels/nunchaku-1.2.1+cu13.0torch2.11-cp310-cp310-linux_x86_64.whl")},
           "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "python": sys.version.split()[0], "venv": sys.prefix,
           "checkpoints": {"int4": {"repo": "nunchaku-ai/nunchaku-sana", "revision": "437e41b979ec875f63c18e67d55373aa5230c308", "file": os.path.basename(INT4), "sha256": sha(INT4)},
                           "base": {"repo": "Efficient-Large-Model/Sana_1600M_1024px_BF16_diffusers", "revision": "e2b3c0cbffebcd09d83805e88b9f5f106afc74ac",
                                    "files": {f: sha(os.path.join(BASE, f)) for f in ("transformer/diffusion_pytorch_model.bf16.safetensors",
                                              "text_encoder/model.bf16-00001-of-00002.safetensors", "text_encoder/model.bf16-00002-of-00002.safetensors",
                                              "vae/diffusion_pytorch_model.bf16.safetensors")}}},
           "scheduler_config": json.load(open(os.path.join(BASE, "scheduler", "scheduler_config.json"))),
           "mjhq_meta_sha256": sha(MJHQ),
           "pip_freeze": subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True).stdout.split("\n")}
    P = prompts()
    out["prompts"] = {"timing": {i: P[i] for i in TIMING}, "dev": {i: P[i] for i in DEVP}, "conf_indices": [CONFP[0], CONFP[-1]]}
    jdump("env.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("pip_freeze", "prompts")}, indent=1, default=str))


@torch.no_grad()
def smoke():
    guard("smoke", 0.08)
    from PIL import Image
    with Run("smoke") as run_:
        pipe, R, lt = load()
        P = prompts()
        res = {"load": lt}
        # official example (prompt, seed 42) through the native pipeline with each model installed directly, twice each; then via the router
        ex = "A cute 🐼 eating 🎋, ink drawing style"
        imgs, lat = {}, {}
        for name, model in (("H", R.H), ("Q", R.Q)):
            pipe.transformer = model
            for rep in range(2):
                o = pipe(prompt=ex, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS,
                         generator=torch.Generator().manual_seed(42), output_type="latent").images
                lat[(name, rep)] = o.float().cpu()
            img = pipe(prompt=ex, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS,
                       generator=torch.Generator().manual_seed(42)).images[0]
            os.makedirs(CK, exist_ok=True)
            img.save(os.path.join(CK, f"example_{name}.png"))
            imgs[name] = np.asarray(img).astype(np.float64)
        pipe.transformer = R
        for name in ("H", "Q"):
            o, c = run(pipe, R, name, embed(pipe, ex), 42)
            lat[(name, "router")] = o.float().cpu()
            res[f"router_calls_{name}"] = c
        for name in ("H", "Q"):
            a, b, r = lat[(name, 0)], lat[(name, 1)], lat[(name, "router")]
            res[name] = {"finite": bool(torch.isfinite(a).all()), "repeat_bitwise": bool(torch.equal(a, b)), "repeat_rel_l2": rel(b, a),
                         "router_vs_direct_bitwise": bool(torch.equal(r, a)), "router_vs_direct_rel_l2": rel(r, a), "latent_shape": list(a.shape)}
        mse = ((imgs["H"] - imgs["Q"]) ** 2).mean()
        res["example_psnr_Q_vs_H_dB"] = float(10 * np.log10(255 ** 2 / mse))
        tol = max(res["H"]["repeat_rel_l2"], res["Q"]["repeat_rel_l2"])
        res["tolerance_rel_l2"] = 0.0 if tol == 0 else 10 * tol
        res["note"] = "official example settings (prompt, seed 42, guidance 4.5, 20 steps, 1024); upstream publishes no reference image hash, so parity = official code path, finite, coherent"
        res["pass"] = all(res[n]["finite"] for n in ("H", "Q")) and all(res[n]["router_vs_direct_rel_l2"] <= res["tolerance_rel_l2"] for n in ("H", "Q"))
        res["gpu_h"] = run_.hours()
    jdump("smoke.json", res)
    print(json.dumps(res, indent=1, default=str))


def nbytes(m):
    return int(sum(p.numel() * p.element_size() for p in m.parameters()) + sum(b.numel() * b.element_size() for b in m.buffers()))


@torch.no_grad()
def kernels():
    guard("kernels", 0.15)
    from torch.profiler import ProfilerActivity, profile
    with Run("kernels") as run_:
        torch.cuda.reset_peak_memory_stats()
        pipe, R, lt = load()
        P = prompts()
        emb = embed(pipe, P[TIMING[0]])
        out = {"load": lt, "bytes": {"H_transformer": nbytes(R.H), "Q_transformer_torch_visible": nbytes(R.Q), "text_encoder": nbytes(pipe.text_encoder),
                                     "vae": nbytes(pipe.vae)},
               "bytes_note": "Q bytes = torch-visible parameters/buffers incl. Nunchaku's packed modules; allocator totals below are authoritative"}
        out["allocated_after_load"] = torch.cuda.memory_allocated()
        mods = {}
        for n, m in R.Q.named_modules():
            t = type(m).__name__
            if not list(m.children()):
                mods.setdefault(t, []).append(n)
        out["Q_leaf_module_types"] = {t: {"count": len(v), "examples": v[:6]} for t, v in mods.items()}
        out["Q_bf16_linear_names"] = [n for n, m in R.Q.named_modules() if isinstance(m, nn.Linear)]
        # identical inputs to both: the scheduler's actual timesteps / latents at steps 0, 10, 19 of an H20 trajectory
        st = {}

        def cb(p, i, t, kw):
            if i + 1 in (10, 19):
                st[i + 1] = (kw["latents"].clone(), ts[i + 1])
            return {}
        R.mode, R.anchors = "H", set()
        R.reset()
        lat0 = pipe.prepare_latents(1, 32, RES, RES, torch.float32, "cuda", torch.Generator().manual_seed(TIMING[0]))
        pipe.scheduler.set_timesteps(STEPS, device="cuda")
        ts = pipe.scheduler.timesteps
        pipe(**emb, negative_prompt=None, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS, generator=torch.Generator().manual_seed(TIMING[0]),
             output_type="latent", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
        st[0] = (lat0, ts[0])
        pe = torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]]).to(torch.bfloat16)
        pm = torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])
        iface = {}
        for i in (0, 10, 19):
            x, t = st[i]
            xin = torch.cat([x] * 2).to(torch.bfloat16)
            tt = t.expand(2) * R.H.config.timestep_scale
            h = R.H(xin, encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=tt, return_dict=False)[0]
            q = R.Q(xin, encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=tt, return_dict=False)[0]
            iface[i] = {"timestep": float(t), "H_shape": list(h.shape), "Q_shape": list(q.shape), "H_dtype": str(h.dtype), "Q_dtype": str(q.dtype),
                        "rel_l2_Q_vs_H": rel(q, h), "finite": bool(torch.isfinite(q).all() and torch.isfinite(h).all())}
        out["interface"] = iface
        pipe.transformer = R.H
        dl = pipe(prompt=P[TIMING[0]], height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS,
                  generator=torch.Generator().manual_seed(TIMING[0]), output_type="latent").images
        pipe.transformer = R
        rl, _ = run(pipe, R, "H", emb, TIMING[0])
        out["router_vs_direct_H_bitwise"] = bool(torch.equal(rl, dl))
        out["router_vs_direct_H_rel_l2"] = rel(rl, dl)
        out["interface_note"] = "same latent (CFG-duplicated), timestep (x timestep_scale), embeddings and mask; both return the flow prediction"
        # profiler: one call each (after a warm call)
        x, t = st[10]
        xin, tt = torch.cat([x] * 2).to(torch.bfloat16), t.expand(2)
        prof = {}
        for name, m in (("Q", R.Q), ("H", R.H)):
            m(xin, encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=tt, return_dict=False)
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]) as p:
                m(xin, encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=tt, return_dict=False)
                torch.cuda.synchronize()
            ev = [e for e in p.key_averages() if e.device_type.name == "CUDA" or getattr(e, "self_device_time_total", 0) > 0]
            rows = sorted(((e.key, e.count, getattr(e, "self_device_time_total", 0) / 1e3) for e in p.key_averages()), key=lambda r: -r[2])
            tot = sum(r[2] for r in rows)
            w4 = [r for r in rows if "W4A4" in r[0] or "gemm_w4a4" in r[0].lower()]
            prof[name] = {"cuda_ms_total_self": tot, "top25": rows[:25], "w4a4_kernels": w4, "w4a4_ms": sum(r[2] for r in w4),
                          "w4a4_launches": sum(r[1] for r in w4)}
            p.export_chrome_trace(os.path.join(CK, f"trace_{name}.json"))
        out["profile"] = prof
        # loaded pipeline memory: hybrid K4 loop + decode, both denoisers + text encoder + VAE resident
        torch.cuda.reset_peak_memory_stats()
        run(pipe, R, "HYB", emb, TIMING[0], SCHED["K4"], out="pil")
        tot = torch.cuda.get_device_properties(0).total_memory
        out["memory"] = {"peak_allocated": torch.cuda.max_memory_allocated(), "peak_reserved": torch.cuda.max_memory_reserved(), "device_total": tot,
                         "fits_with_1GiB_reserve": torch.cuda.max_memory_reserved() + 2 ** 30 <= tot}
        out["gpu_h"] = run_.hours()
    jdump("kernels.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("Q_leaf_module_types", "Q_bf16_linear_names")}, indent=1, default=str)[:6000])


def ev_time(fn):
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e)


@torch.no_grad()
def timing():
    guard("timing", 0.2)
    with Run("timing") as run_:
        pipe, R, lt = load()
        P = prompts()
        embs = [embed(pipe, P[i]) for i in TIMING]
        out = {"load": lt, "note": "no profiler; CUDA events; text encoder resident; fake nothing (real packed W4A4 for Q)"}
        # per call
        e = embs[0]
        pe = torch.cat([e["negative_prompt_embeds"], e["prompt_embeds"]]).to(torch.bfloat16)
        pm = torch.cat([e["negative_prompt_attention_mask"], e["prompt_attention_mask"]])
        pipe.scheduler.set_timesteps(STEPS, device="cuda")
        ts = pipe.scheduler.timesteps
        x = torch.randn(2, 32, 32, 32, generator=torch.Generator().manual_seed(0)).to("cuda", torch.bfloat16)
        jobs = [(m, i) for m in ("H", "Q") for i in (0, 10, 19)]
        call = lambda m, i: (R.H if m == "H" else R.Q)(x, encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=ts[i].expand(2), return_dict=False)
        for _ in range(5):
            for m, i in jobs:
                call(m, i)
        torch.cuda.synchronize()
        order = [j for j in jobs for _ in range(20)]
        np.random.default_rng(0).shuffle(order)
        meas = {j: [] for j in jobs}
        for m, i in order:
            meas[(m, i)].append(ev_time(lambda: call(m, i)))
        pc = {f"{m}@{i}": {"median_ms": float(np.median(v)), "min_ms": float(min(v)), "max_ms": float(max(v)), "n": len(v)} for (m, i), v in meas.items()}
        out["per_call"] = pc
        out["ratio_per_call"] = {i: pc[f"Q@{i}"]["median_ms"] / pc[f"H@{i}"]["median_ms"] for i in (0, 10, 19)}
        out["ratio_per_call_pooled"] = float(np.median([np.median(meas[("Q", i)]) for i in (0, 10, 19)]) / np.median([np.median(meas[("H", i)]) for i in (0, 10, 19)]))
        print(f"  per-call ratio Q/H: {out['ratio_per_call']}", flush=True)
        # complete loops
        arms = {"H20": ("H", ()), "Q20": ("Q", ()), "HYB-K4": ("HYB", SCHED["K4"]), "HYB-K8": ("HYB", SCHED["K8"])}
        for a, (m, an) in arms.items():
            run(pipe, R, m, embs[0], TIMING[0], an)
        loops = {a: [] for a in arms}
        calls = {}
        for rep in range(5):
            for a, (m, an) in arms.items():
                k = rep % 4
                torch.cuda.synchronize()
                ms = ev_time(lambda: calls.__setitem__(a, run(pipe, R, m, embs[k], TIMING[k], an)[1]))
                loops[a].append(ms / 1e3)
        out["loops_s"] = {a: {"median": float(np.median(v)), "min": float(min(v)), "max": float(max(v)), "all": v, "calls": calls[a]} for a, v in loops.items()}
        h = out["loops_s"]["H20"]["median"]
        out["loop_ratio_vs_H20"] = {a: out["loops_s"][a]["median"] / h for a in arms}
        km = jload("kernels.json") or {}
        fits = bool(km.get("memory", {}).get("fits_with_1GiB_reserve", False))
        gate = {s: bool(out["loop_ratio_vs_H20"][f"HYB-{s}"] <= 0.90 and fits) for s in ("K4", "K8")}
        out["gate"] = {"rule": "hybrid loop median <= 0.90 x H20 loop median and both fit", "fits": fits, "pass": gate,
                       "selected": "K4" if gate["K4"] else ("K8" if gate["K8"] else None),
                       "idealized_threshold": {"K4": 0.75, "K8": 0.85}, "measured_C_ratio": out["ratio_per_call_pooled"]}
        print(f"  loops {out['loop_ratio_vs_H20']} gate {out['gate']}", flush=True)
        # loaded pipeline latency (text encode + loop + decode)
        pl = {}
        for a in ["H20", "Q20"] + [f"HYB-{s}" for s in ("K4", "K8")]:
            m, an = arms[a]
            ts_ = []
            for rep in range(4):
                R.mode, R.anchors = m, set(an)
                R.reset()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                pipe(prompt=P[TIMING[rep % 4]], height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS,
                     generator=torch.Generator().manual_seed(TIMING[rep % 4]))
                torch.cuda.synchronize()
                ts_.append(time.perf_counter() - t0)
            pl[a] = {"warmup_s": ts_[0], "median_s": float(np.median(ts_[1:])), "all_s": ts_[1:]}
        out["loaded_pipeline_s"] = pl
        out["memory_peak_reserved"] = torch.cuda.max_memory_reserved()
        out["gpu_h"] = run_.hours()
    jdump("timing.json", out)
    print(json.dumps({k: out[k] for k in ("ratio_per_call", "loop_ratio_vs_H20", "gate", "loaded_pipeline_s")}, indent=1, default=str))




# =================================================================================================================== conditional reuse pilot
class FakeQ(nn.Module):
    """deterministic stand-in cheap model for routing-logic checks only (Q itself is not bitwise repeatable): 0.97 x H's output."""

    def __init__(self, H):
        super().__init__()
        self.H = H

    def forward(self, *a, **k):
        return (self.H(*a, **k)[0] * 0.97,)


def selected():
    t = jload("timing.json")
    assert t and t["gate"]["selected"], "timing gate did not pass: quality stages are not permitted"
    return t["gate"]["selected"]


@torch.no_grad()
def own_loop(pipe, R, mode, emb, seed, anchors=(), stop=None, resume=None):
    """replica of SanaPipeline's denoising loop (same ops) supporting stop/resume with the full scheduler state (deepcopy) and router state."""
    R.mode, R.anchors = mode, set(anchors)
    pe = torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]])
    pm = torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])
    if resume is None:
        R.reset()
        sch = pipe.scheduler
        sch.set_timesteps(STEPS, device="cuda")
        lat = pipe.prepare_latents(1, 32, RES, RES, torch.float32, "cuda", torch.Generator().manual_seed(seed))
        i0 = 0
    else:
        sch, lat, i0 = copy.deepcopy(resume["sched"]), resume["lat"].clone(), resume["i"]
        R.i, R.d, R.stale, R.calls = resume["router"]["i"], resume["router"]["d"], resume["router"]["stale"], dict(resume["router"]["calls"])
    for i, t in enumerate(sch.timesteps):
        if i < i0:
            continue
        if stop is not None and i == stop:
            return {"sched": copy.deepcopy(sch), "lat": lat.clone(), "i": i,
                    "router": {"i": R.i, "d": None if R.d is None else R.d.clone(), "stale": None if R.stale is None else R.stale.clone(), "calls": dict(R.calls)}}
        x = torch.cat([lat] * 2)
        ts = t.expand(x.shape[0]) * R.config.timestep_scale
        npred = R(x.to(torch.bfloat16), encoder_hidden_states=pe.to(torch.bfloat16), encoder_attention_mask=pm, timestep=ts, return_dict=False,
                  attention_kwargs=None)[0].float()
        u, c = npred.chunk(2)
        npred = u + GUIDE * (c - u)
        lat = sch.step(npred, t, lat, return_dict=False)[0]
    return lat


@torch.no_grad()
def checks():
    guard("checks", 0.08)
    K = selected()
    an = SCHED[K]
    with Run("checks") as run_:
        pipe, R, _ = load()
        P = prompts()
        s = DEVP[0]
        emb = embed(pipe, P[s])
        ch = {"schedule": K, "anchors": an}
        h20, c = run(pipe, R, "H", emb, s)
        pipe.transformer = R.H
        direct = pipe(**emb, negative_prompt=None, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS,
                      generator=torch.Generator().manual_seed(s), output_type="latent").images
        pipe.transformer = R.Q
        dq = [pipe(**emb, negative_prompt=None, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS,
                   generator=torch.Generator().manual_seed(s), output_type="latent").images for _ in range(2)]
        pipe.transformer = R
        q20, _ = run(pipe, R, "Q", emb, s)
        ch["wrapper_H20_equals_unwrapped_bitwise"] = bool(torch.equal(h20, direct))
        ch["wrapper_Q20_vs_unwrapped_rel_l2"] = rel(q20, dq[0])
        ch["Q20_unwrapped_run_to_run_rel_l2"] = rel(dq[1], dq[0])
        xa, ca = run(pipe, R, "R", emb, s, anchors=range(STEPS))
        ch["anchor_every_call_equals_H20_bitwise"] = bool(torch.equal(xa, h20))
        ch["anchor_every_call_calls"] = ca
        Qreal = R.Q
        R.Q = R.H                                                             # identical stateless cheap/high
        D = []
        orig = R.forward

        def spy(*a, **k):
            o = orig(*a, **k)
            if R.d is not None:
                D.append(float(R.d.abs().max()))
            return o
        R.forward = spy
        xi, _ = run(pipe, R, "R", emb, s, anchors=an)
        R.forward = orig
        ch["identical_QH_zero_discrepancy"] = bool(max(D) == 0.0 and torch.equal(xi, h20))
        R.Q = FakeQ(R.H)                                                      # deterministic stand-in for routing logic
        xr0, _ = run(pipe, R, "R0", emb, s, anchors=an)
        xA, cA = run(pipe, R, "A", emb, s, anchors=an)
        ch["reuse_disabled_equals_A_bitwise_fakeQ"] = bool(torch.equal(xr0, xA))
        full = own_loop(pipe, R, "R", emb, s, an)
        ch["own_loop_equals_pipeline_bitwise_fakeQ"] = bool(torch.equal(full, run(pipe, R, "R", emb, s, an)[0]))
        rs = {}
        for st in (8, 9):
            snap = own_loop(pipe, R, "R", emb, s, an, stop=st)
            rs[st] = bool(torch.equal(own_loop(pipe, R, "R", emb, s, an, resume=snap), full))
        ch["resume_incl_scheduler_history_bitwise_fakeQ"] = rs
        R.Q = Qreal
        counts = {}
        for m in ("H", "Q", "A", "R", "S"):
            counts[m] = run(pipe, R, m, emb, s, anchors=an)[1]
        ch["calls"] = counts
        ch["no_offload_or_hooks"] = bool(getattr(pipe, "_offload_gpu_id", None) is None and not any(hasattr(m, "_hf_hook") for m in (R.H, R.Q, pipe.vae, pipe.text_encoder)))
        ch["pass"] = bool(ch["wrapper_H20_equals_unwrapped_bitwise"] and ch["anchor_every_call_equals_H20_bitwise"] and ch["identical_QH_zero_discrepancy"]
                          and ch["reuse_disabled_equals_A_bitwise_fakeQ"] and ch["own_loop_equals_pipeline_bitwise_fakeQ"] and all(rs.values())
                          and ch["no_offload_or_hooks"])
        ch["note"] = "Q paths cannot be bitwise (Q is not repeatable, determinism.json); routing logic verified with a deterministic stand-in (0.97 x H)"
        ch["gpu_h"] = run_.hours()
    jdump("checks.json", ch)
    print(json.dumps(ch, indent=1, default=str))


def to_u8(pipe, lat):
    im = pipe.vae.decode(lat.to(torch.bfloat16) / pipe.vae.config.scaling_factor, return_dict=False)[0]
    return ((im.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu()


def loop_time(pipe, R, mode, emb, seed, an, steps=STEPS, reps=3):
    run(pipe, R, mode, emb, seed, an, steps=steps)
    v = []
    for _ in range(reps):
        torch.cuda.synchronize()
        v.append(ev_time(lambda: run(pipe, R, mode, emb, seed, an, steps=steps)) / 1e3)
    return float(np.median(v)), v


@torch.no_grad()
def gen_arms(pipe, R, idx, arms, an, P, tag):
    lat, img, calls = {a: [] for a in arms}, {a: [] for a in arms}, {}
    for j, s in enumerate(idx):
        emb = embed(pipe, P[s])
        for a in arms:
            if a.startswith("H") and a[1:].isdigit() and a != "H20":
                x, c = run(pipe, R, "H", emb, s, steps=int(a[1:]))
            else:
                m = {"H20": "H", "Q20": "Q"}.get(a, a)
                x, c = run(pipe, R, m, emb, s, anchors=an)
            assert torch.isfinite(x).all(), (a, s)
            lat[a].append(x.float().cpu()); img[a].append(to_u8(pipe, x)); calls[a] = c
        print(f"  {tag} prompt {s} ({j + 1}/{len(idx)})", flush=True)
    lat = {a: torch.cat(v) for a, v in lat.items()}
    img = {a: torch.cat(v) for a, v in img.items()}
    torch.save({"lat": lat, "img": img, "idx": idx, "calls": calls}, os.path.join(CK, f"{tag}_outputs.pt"))
    return lat, calls


@torch.no_grad()
def dev():
    K = selected()
    an = SCHED[K]
    assert (jload("checks.json") or {}).get("pass"), "checks not passed"
    guard("dev", 0.15)
    with Run("dev") as run_:
        pipe, R, _ = load()
        P = prompts()
        arms = ["H20", "Q20", "A", "R", "S"]
        lat, calls = gen_arms(pipe, R, DEVP, arms, an, P, "dev")
        relm = {a: [float(((lat[a][i] - lat["H20"][i]).double().pow(2).sum() / lat["H20"][i].double().pow(2).sum())) for i in range(len(DEVP))]
                for a in arms if a != "H20"}
        out = {"schedule": K, "anchors": an, "calls": calls, "latent_rel_mse": relm, "mean": {a: float(np.mean(v)) for a, v in relm.items()},
               "median": {a: float(np.median(v)) for a, v in relm.items()}}
        # diagnostics at the same states (H20 trajectory): guided Q - H, Q's own repeat noise, persistence
        diag = {"lag_corr": {}, "lag_energy_ratio": {}, "held_remaining": [], "raw_energy": [], "q_noise_energy": []}
        E = []
        for s in DEVP:
            emb = embed(pipe, P[s])
            xs = {}

            def cb(p, i, t, kw):
                xs[i + 1] = kw["latents"].clone()
                return {}
            R.mode, R.anchors = "H", set(); R.reset()
            pipe(**emb, negative_prompt=None, height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=STEPS, generator=torch.Generator().manual_seed(s),
                 output_type="latent", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
            xs[0] = pipe.prepare_latents(1, 32, RES, RES, torch.float32, "cuda", torch.Generator().manual_seed(s))
            pipe.scheduler.set_timesteps(STEPS, device="cuda")
            ts = pipe.scheduler.timesteps
            pe = torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]]).to(torch.bfloat16)
            pm = torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])
            g = lambda o: (lambda u, c: u + GUIDE * (c - u))(*o.float().chunk(2))
            e, nz = [], []
            for i in range(STEPS):
                x = torch.cat([xs[i]] * 2).to(torch.bfloat16)
                k = dict(encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=ts[i].expand(2) * R.config.timestep_scale, return_dict=False)
                h = g(R.H(x, **k)[0]); q1 = g(R.Q(x, **k)[0]); q2 = g(R.Q(x, **k)[0])
                e.append((q1 - h).double().cpu()); nz.append(float((q2 - q1).double().pow(2).sum()) / 2)
            E.append(e)
            diag["raw_energy"].append([float(v.pow(2).sum()) for v in e]); diag["q_noise_energy"].append(nz)
        for L in (1, 2, 4, 8):
            num = sum(float((e[i] * e[i + L]).sum()) for e in E for i in range(STEPS - L))
            den = (sum(float(e[i].pow(2).sum()) for e in E for i in range(STEPS - L)) * sum(float(e[i + L].pow(2).sum()) for e in E for i in range(STEPS - L))) ** 0.5
            diag["lag_corr"][L] = num / den
            diag["lag_energy_ratio"][L] = sum(float((e[i + L] - e[i]).pow(2).sum()) for e in E for i in range(STEPS - L)) / sum(float(e[i + L].pow(2).sum()) for e in E for i in range(STEPS - L))
        held = [i for i in range(STEPS) if i not in an]
        anc_of = lambda i: max(a for a in an if a <= i)
        diag["held_remaining_fraction"] = sum(float((e[i] - e[anc_of(i)]).pow(2).sum()) for e in E for i in held) / sum(float(e[i].pow(2).sum()) for e in E for i in held)
        diag["q_noise_over_raw_energy"] = float(np.sum(diag["q_noise_energy"]) / np.sum(diag["raw_energy"]))
        diag["definitions"] = {"e_i": "guided Q - guided H at the H20 state x_i (scheduler-consumed flow prediction)", "q_noise": "|Q2 - Q1|^2 / 2 at the same state (Q run-to-run variance)",
                               "held_remaining_fraction": "sum_held |e_i - e_anchor(i)|^2 / sum_held |e_i|^2 at the selected schedule (pooled, same states)"}
        out["diagnostics"] = diag
        rt = {m: loop_time(pipe, R, mm, embed(pipe, P[DEVP[0]]), DEVP[0], an) for m, mm in (("H20", "H"), ("R", "R"), ("A", "A"))}
        out["loop_time_s"] = {m: {"median": v[0], "all": v[1]} for m, v in rt.items()}
        ratio = out["mean"]["R"] / out["mean"]["A"]
        out["rule"] = {"R_over_A_mean": ratio, "R_time_over_H20": rt["R"][0] / rt["H20"][0], "finite": True,
                       "pass": bool(ratio <= 0.90 and rt["R"][0] / rt["H20"][0] <= 0.90),
                       "text": "confirmation iff R mean endpoint rel. MSE <= 0.90 x A, finite, R warm loop median <= 0.90 x H20"}
        out["gpu_h"] = run_.hours()
    jdump("dev.json", out)
    print(json.dumps({k: out[k] for k in ("mean", "median", "rule", "loop_time_s")}, indent=1), json.dumps({k: v for k, v in out["diagnostics"].items() if k in ("lag_corr", "lag_energy_ratio", "held_remaining_fraction", "q_noise_over_raw_energy")}, indent=1))


@torch.no_grad()
def conf():
    K = selected()
    an = SCHED[K]
    d = jload("dev.json")
    assert d and d["rule"]["pass"], "development rule not met: confirmation not permitted"
    with Run("hk_select") as run_:
        pipe, R, _ = load()
        P = prompts()
        emb = embed(pipe, P[DEVP[0]])
        tR = loop_time(pipe, R, "R", emb, DEVP[0], an, reps=5)
        tk = {k: loop_time(pipe, R, "H", emb, DEVP[0], (), steps=k, reps=3)[0] for k in range(1, STEPS + 1)}
        k = max(kk for kk, v in tk.items() if v <= tR[0])
        per_prompt = sum(v for v in tk.values() if False) or (d["loop_time_s"]["H20"]["median"] * 2 + 2 * tR[0] + tk[k] + 3.0)
        sel = {"R_loop_s": tR, "H_k_loop_s": tk, "k": k, "rule": "largest k in 1..20 with warmed BF16 k-step loop <= R's loop (timing only)"}
    jdump("hk_select.json", sel)
    est = len(CONFP) * per_prompt / 3600 * 1.2
    guard("conf", est)
    with Run("conf") as run_:
        arms = ["H20", "Q20", "A", "R", "S", f"H{k}"]
        lat, calls = gen_arms(pipe, R, CONFP, arms, an, P, "conf")
        relm = {a: [float(((lat[a][i] - lat["H20"][i]).double().pow(2).sum() / lat["H20"][i].double().pow(2).sum())) for i in range(len(CONFP))]
                for a in arms if a != "H20"}
        out = {"schedule": K, "k": k, "calls": calls, "latent_rel_mse": relm, "mean": {a: float(np.mean(v)) for a, v in relm.items()},
               "estimate_h": est, "gpu_h": run_.hours()}
        torch.cuda.reset_peak_memory_stats()
        pl = {}
        for a, (m, an_, st) in {"H20": ("H", (), STEPS), "R": ("R", an, STEPS), f"H{k}": ("H", (), k)}.items():
            v = []
            for rep in range(3):
                R.mode, R.anchors = m, set(an_); R.reset()
                torch.cuda.synchronize(); t0 = time.perf_counter()
                pipe(prompt=P[DEVP[rep]], height=RES, width=RES, guidance_scale=GUIDE, num_inference_steps=st, generator=torch.Generator().manual_seed(DEVP[rep]))
                torch.cuda.synchronize(); v.append(time.perf_counter() - t0)
            pl[a] = {"median_s": float(np.median(v)), "all_s": v}
        out["loaded_pipeline_s"] = pl
        out["peak_reserved"] = torch.cuda.max_memory_reserved()
    jdump("conf.json", out)
    print(json.dumps({k_: out[k_] for k_ in ("k", "mean", "loaded_pipeline_s")}, indent=1))


STAGES = {"env": env, "smoke": smoke, "kernels": kernels, "timing": timing, "checks": checks, "dev": dev, "conf": conf}


if __name__ == "__main__":
    STAGES[sys.argv[1]]()
