"""sr1: SANA INT4 repeatability and persistence diagnostic (results/sana_repeat/PROTOCOL.md). Ledger tag sr1 (common handoff budget: h1031_budget).
  repro  : minimal reproducer at 4 prompts x 5 fixed states (saved): consecutive / interleaved / no-clone / direct-pipeline / routed / bare,
           pre-CFG rows and guided output, input checksums, RNG, per-layer isolation, BF16 pieces outside the fused stack
  fresh  : 2 saved states in a fresh process (run with CUDA_LAUNCH_BLOCKING=1), compared with repro
  traces : 8 prompts x 20 states x R = 8 repeats, saved traces; mean-discrepancy / variance / persistence / independence analysis
Run: /home/nk_env/bin/python scripts/sana_repeat.py repro|fresh|traces
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import h1031_budget as HB
import sana_a30 as S
from sana_a30_rep import traj_states

S.TAG = "sr1"
OUT = os.path.join(S.ROOT, "results", "sana_repeat")
CK = os.path.join(S.ROOT, "ckpt", "sana_repeat")
PROMPTS, STEPS_FIX = [352, 356, 361, 365], [0, 5, 10, 15, 19]
TR_PROMPTS, R = list(range(352, 360)), 8


def jdump(n, o):
    os.makedirs(OUT, exist_ok=True)
    json.dump(o, open(os.path.join(OUT, n), "w"), indent=1, default=str)


def guard(name, est):
    ok, info = HB.ok("sr1", est)
    print(f"[sr1 budget] {name}: {info} -> {'OK' if ok else 'STOP'}", flush=True)
    if not ok:
        jdump(f"{name}.json", {"status": "skipped (budget)", **info})
        sys.exit(4)


def csum(t):
    return hashlib.sha256(t.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes() if t.dtype != torch.bool else t.cpu().numpy().tobytes()).hexdigest()[:16]


def guided(o):
    u, c = o.float().chunk(2)
    return u + S.GUIDE * (c - u)


def spread(ys):
    Y = torch.stack([y.double() for y in ys])
    mu = Y.mean(0)
    return {"bitwise": all(torch.equal(ys[0], y) for y in ys[1:]), "max_rel_vs_first": max(S.rel(y, ys[0]) for y in ys[1:]),
            "mean_rel_vs_mean": float(np.mean([S.rel(y, mu) for y in Y])), "finite": bool(torch.isfinite(Y).all())}


def call_kwargs(st, R_):
    return torch.cat([st["x"]] * 2).to("cuda", torch.bfloat16), dict(encoder_hidden_states=st["pe"].cuda(), encoder_attention_mask=st["pm"].cuda(),
                                                                       timestep=st["t"].cuda().expand(2) * R_.config.timestep_scale, return_dict=False,
                                                                       attention_kwargs=None)


@torch.no_grad()
def build_states(pipe, R_):
    P = S.prompts()
    states = {}
    for s in PROMPTS:
        emb = S.embed(pipe, P[s])
        xs, ts = traj_states(pipe, R_, emb, s)
        pe = torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]]).to(torch.bfloat16)
        pm = torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])
        for i in STEPS_FIX:
            states[f"{s}@{i}"] = {"x": xs[i].cpu(), "t": ts[i].cpu(), "pe": pe.cpu(), "pm": pm.cpu(), "cfg_order": "uncond rows first", "prompt": s, "step": i}
    states["_rng"] = {"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}
    os.makedirs(CK, exist_ok=True)
    torch.save(states, os.path.join(CK, "states.pt"))
    return states


@torch.inference_mode()
def repro():
    guard("repro", 0.08)
    with S.Run("repro") as run_:
        pipe, R_, _ = S.load()
        Q, H = R_.Q, R_.H
        states = build_states(pipe, R_)
        keys = [k for k in states if not k.startswith("_")]
        out = {"states": {}, "nrep": 16}
        for n, k in enumerate(keys):
            st = states[k]
            x, kw = call_kwargs(st, R_)
            before = {"x": csum(x), "pe": csum(kw["encoder_hidden_states"]), "pm": csum(kw["encoder_attention_mask"]), "t": csum(kw["timestep"])}
            rng0 = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
            rec = {}
            for name, f in (("Q_bare", lambda: Q(x, **kw)[0]), ("H_bare", lambda: H(x, **kw)[0])):
                raw, clones = [], []
                for _ in range(16):
                    y = f()
                    torch.cuda.synchronize()
                    raw.append(y); clones.append(y.clone())
                rec[name] = {"uncond_row": spread([c[0] for c in clones]), "cond_row": spread([c[1] for c in clones]),
                             "guided": spread([guided(c) for c in clones]),
                             "no_clone_refs_unchanged": all(torch.equal(a, b) for a, b in zip(raw, clones))}
                if name == "Q_bare":
                    qg = torch.stack([guided(c).double() for c in clones])
                else:
                    hg = guided(clones[0]).double()
            rec["Qmean_vs_H_rel"] = S.rel(qg.mean(0), hg)
            # routed invocation
            ys = []
            for _ in range(16):
                R_.mode, R_.anchors = "Q", set(); R_.reset()
                ys.append(R_(x, **kw)[0].clone()); torch.cuda.synchronize()
            rec["Q_routed_guided"] = spread([guided(y) for y in ys])
            # interleaved with another fixed input
            st2 = states[keys[(n + 1) % len(keys)]]
            x2, kw2 = call_kwargs(st2, R_)
            ya = []
            for _ in range(16):
                ya.append(Q(x, **kw)[0].clone()); Q(x2, **kw2)
                torch.cuda.synchronize()
            rec["Q_interleaved_guided"] = spread([guided(y) for y in ya])
            after = {"x": csum(x), "pe": csum(kw["encoder_hidden_states"]), "pm": csum(kw["encoder_attention_mask"]), "t": csum(kw["timestep"])}
            rec["inputs_unchanged"] = before == after
            rec["rng_unchanged"] = bool(torch.equal(rng0[0], torch.get_rng_state()) and torch.equal(rng0[1], torch.cuda.get_rng_state()))
            out["states"][k] = rec
            print(f"  {k}: Q guided {rec['Q_bare']['guided']['max_rel_vs_first']:.4f} (uncond {rec['Q_bare']['uncond_row']['max_rel_vs_first']:.4f}, "
                  f"cond {rec['Q_bare']['cond_row']['max_rel_vs_first']:.4f}) routed {rec['Q_routed_guided']['max_rel_vs_first']:.4f} "
                  f"interleaved {rec['Q_interleaved_guided']['max_rel_vs_first']:.4f} H bitwise {rec['H_bare']['guided']['bitwise']} "
                  f"inputs {rec['inputs_unchanged']} rng {rec['rng_unchanged']} noclone {rec['Q_bare']['no_clone_refs_unchanged']}", flush=True)
        # official direct pipeline invocation at step 0 (the initial latent is the saved state 0), captured by a forward hook
        P = S.prompts()
        direct = {}
        cap = []
        h = Q.register_forward_hook(lambda m, a, o: cap.append(o[0].detach().clone()))

        def stop(p, i, t, kw_):
            p._interrupt = True
            return {}
        pipe.transformer = Q
        for s in PROMPTS:
            cap.clear()
            for _ in range(16):
                pipe(prompt=P[s], height=S.RES, width=S.RES, guidance_scale=S.GUIDE, num_inference_steps=S.STEPS,
                     generator=torch.Generator().manual_seed(s), output_type="latent", callback_on_step_end=stop)
                pipe._interrupt = False
            st = states[f"{s}@0"]
            x, kw = call_kwargs(st, R_)
            bare = Q(x, **kw)[0].clone()
            direct[s] = {"direct_guided": spread([guided(c) for c in cap]), "direct_mean_vs_bare_single_rel": S.rel(torch.stack([guided(c).double() for c in cap]).mean(0), guided(bare).double()),
                         "n": len(cap)}
        h.remove()
        pipe.transformer = R_
        out["direct_pipeline_step0"] = direct
        # BF16 pieces outside the fused stack: block-stack input (patch embed / time / caption) and proj_out on fixed inputs
        blk = Q.transformer_blocks[0]
        caps = []
        hpre = blk.register_forward_pre_hook(lambda m, a, k: caps.append(([v.clone() if torch.is_tensor(v) else v for v in a],
                                                                           {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in k.items()})), with_kwargs=True)
        pin = []
        hpo = Q.proj_out.register_forward_pre_hook(lambda m, a: pin.append(a[0].clone()))
        st = states[keys[2]]
        x, kw = call_kwargs(st, R_)
        for _ in range(8):
            Q(x, **kw)
        hpre.remove(); hpo.remove()
        tens = lambda c: [t for t in c[0] if torch.is_tensor(t)] + [v for v in c[1].values() if torch.is_tensor(v)]
        out["outside_stack"] = {"block_stack_inputs_bitwise_across_calls": all(all(torch.equal(a, b) for a, b in zip(tens(caps[0]), tens(c))) for c in caps[1:]),
                                "proj_out_on_fixed_input_bitwise": all(torch.equal(Q.proj_out(pin[0]), Q.proj_out(pin[0])) for _ in range(8)),
                                "proj_out_inputs_differ_across_calls": not all(torch.equal(pin[0], p) for p in pin[1:])}
        # per-layer isolation (supported forward_layer_at) at 2 states
        import inspect
        sig = inspect.signature(blk.forward)
        layers = {}
        for k in (keys[2], keys[10]):
            st = states[k]
            x, kw = call_kwargs(st, R_)
            caps.clear()
            hpre = blk.register_forward_pre_hook(lambda m, a, kk: caps.append(sig.bind(*a, **kk)), with_kwargs=True)
            Q(x, **kw)
            hpre.remove()
            b = caps[0]; b.apply_defaults()
            a = {kk: (v.clone() if torch.is_tensor(v) else v) for kk, v in b.arguments.items()}
            a.pop("skip_first_layer", None)
            h0 = a.pop("hidden_states")
            ref_in, hc = [], h0.clone()
            for idx in range(20):
                ref_in.append(hc.clone()); hc = blk.forward_layer_at(idx, hc, **a)
            iso = [spread([blk.forward_layer_at(idx, ref_in[idx].clone(), **a).float().clone() for _ in range(16)]) for idx in range(20)]
            layers[k] = {"first_nonbitwise": next((i for i, v in enumerate(iso) if not v["bitwise"]), None),
                         "isolated_max_rel": [v["max_rel_vs_first"] for v in iso]}
        out["layers"] = layers
        out["gpu_h"] = run_.hours()
    jdump("repro.json", out)
    print(json.dumps({k: v for k, v in out.items() if k != "states"}, indent=1, default=str)[:3000])


@torch.inference_mode()
def fresh():
    guard("fresh", 0.02)
    with S.Run("fresh") as run_:
        torch.manual_seed(0)
        pipe, R_, _ = S.load()
        states = torch.load(os.path.join(CK, "states.pt"))
        prev = json.load(open(os.path.join(OUT, "repro.json")))
        out = {"CUDA_LAUNCH_BLOCKING": os.environ.get("CUDA_LAUNCH_BLOCKING"), "states": {}}
        for k in ("352@10", "361@0"):
            x, kw = call_kwargs(states[k], R_)
            ys = [guided(R_.Q(x, **kw)[0].clone()) for _ in range(16)]
            hh = guided(R_.H(x, **kw)[0])
            out["states"][k] = {"Q_guided": spread(ys), "Qmean_vs_H_rel": S.rel(torch.stack([y.double() for y in ys]).mean(0), hh.double()),
                                "first_process": {"Q_guided_max_rel": prev["states"][k]["Q_bare"]["guided"]["max_rel_vs_first"],
                                                  "Qmean_vs_H_rel": prev["states"][k]["Qmean_vs_H_rel"]}}
        out["gpu_h"] = run_.hours()
    jdump("fresh.json", out)
    print(json.dumps(out, indent=1, default=str))


@torch.inference_mode()
def traces():
    guard("traces", 0.08)
    with S.Run("traces") as run_:
        pipe, R_, _ = S.load()
        P = S.prompts()
        E, Hn = [], []
        for j, s in enumerate(TR_PROMPTS):
            emb = S.embed(pipe, P[s])
            xs, ts = traj_states(pipe, R_, emb, s)
            pe = torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]]).to(torch.bfloat16)
            pm = torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])
            ep, hp = [], []
            for i in range(S.STEPS):
                x = torch.cat([xs[i]] * 2).to(torch.bfloat16)
                kw = dict(encoder_hidden_states=pe, encoder_attention_mask=pm, timestep=ts[i].expand(2) * R_.config.timestep_scale, return_dict=False)
                hg = guided(R_.H(x, **kw)[0])
                ep.append(torch.stack([(guided(R_.Q(x, **kw)[0]) - hg) for _ in range(R)]).half().cpu())
                hp.append(float(hg.double().pow(2).sum()))
            E.append(torch.stack(ep)); Hn.append(hp)
            print(f"  traces prompt {s} ({j + 1}/{len(TR_PROMPTS)})", flush=True)
        E = torch.stack(E)                                                    # [prompt, step, repeat, 1, 32, 32, 32]
        os.makedirs(CK, exist_ok=True)
        torch.save({"E": E, "H_energy": Hn, "prompts": TR_PROMPTS, "R": R, "note": "e = guided Q - guided H at the BF16 trajectory states, fp16"},
                   os.path.join(CK, "traces.pt"))
        out = analyze(E.double())
        out["gpu_h"] = run_.hours()
    jdump("traces.json", out)
    print(json.dumps({k: v for k, v in out.items() if k not in ("per_prompt",)}, indent=1, default=str))


def analyze(E):
    """E [P, T, R, ...] float64. All energies summed over elements; prompt-level bootstrap."""
    Pn, Tn, Rn = E.shape[:3]
    F = E.flatten(3)
    an = [0, 4, 8, 12, 16]
    held = [i for i in range(Tn) if i not in an]
    anc = lambda i: max(a for a in an if a <= i)
    mean = F.mean(2)                                                          # p-bar [P, T, N]
    V = (F - mean[:, :, None]).pow(2).sum(-1).sum(-1) / (Rn - 1)              # [P, T]
    raw = F.pow(2).sum(-1).mean(-1)                                           # mean over repeats of ||e||^2  [P, T]
    rd = (F[:, :, 0] - F[:, :, 1]).pow(2).sum(-1) / 2                         # repeat-difference estimate [P, T]
    pbar_e = mean.pow(2).sum(-1)                                              # ||p-bar||^2
    pcorr_e = pbar_e - V / Rn                                                 # finite-repeat-corrected ||p||^2
    h = Rn // 2
    m1, m2 = F[:, :, :h].mean(2), F[:, :, h:].mean(2)
    dot = lambda a, b: (a * b).sum(-1)
    per = []
    for p in range(Pn):
        r = {"V": V[p].tolist(), "raw": raw[p].tolist(), "repeat_diff": rd[p].tolist(), "pbar": pbar_e[p].tolist(), "pcorr": pcorr_e[p].tolist(),
             "lag": {}, "held": {}}
        for L in (1, 2, 4, 8):
            r["lag"][L] = {"split_cross": float(sum(0.5 * (dot(m1[p, i], m2[p, i + L]) + dot(m2[p, i], m1[p, i + L])) for i in range(Tn - L))),
                           "split_a": float(sum(dot(m1[p, i], m2[p, i]) for i in range(Tn - L))),
                           "split_b": float(sum(dot(m1[p, i + L], m2[p, i + L]) for i in range(Tn - L))),
                           "single_cross": float(sum(dot(F[p, i, 0], F[p, i + L, 1]) for i in range(Tn - L))),
                           "single_a": float(sum(dot(F[p, i, 0], F[p, i, 0]) for i in range(Tn - L))),
                           "single_b": float(sum(dot(F[p, i + L, 1], F[p, i + L, 1]) for i in range(Tn - L)))}
        # held subtraction on the mean: ||pbar_t - pbar_a||^2 - (V_t + V_a)/R (corrected); split-half cross estimator; single calls
        r["held"] = {"mean_raw": float(sum(dot(mean[p, i] - mean[p, anc(i)], mean[p, i] - mean[p, anc(i)]) for i in held)),
                     "mean_corr": float(sum(dot(mean[p, i] - mean[p, anc(i)], mean[p, i] - mean[p, anc(i)]) - (V[p, i] + V[p, anc(i)]) / Rn for i in held)),
                     "split": float(sum(dot(m1[p, i] - m1[p, anc(i)], m2[p, i] - m2[p, anc(i)]) for i in held)),
                     "p_energy_corr": float(sum(pcorr_e[p, i] for i in held)), "p_energy_split": float(sum(dot(m1[p, i], m2[p, i]) for i in held)),
                     "single": float(sum(dot(F[p, i, 0] - F[p, anc(i), 1], F[p, i, 0] - F[p, anc(i), 1]) for i in held)),
                     "single_energy": float(sum(dot(F[p, i, 0], F[p, i, 0]) for i in held))}
        per.append(r)
    # independence checks: repeat-order drift and successive-repeat deviation autocorrelation
    dev = F - mean[:, :, None]                                                # [P, T, R, N]
    idx = torch.arange(Rn, dtype=torch.float64) - (Rn - 1) / 2
    drift = float((dev * idx[None, None, :, None]).sum(2).pow(2).sum() / (idx.pow(2).sum() * dev.pow(2).sum() / Rn))
    ac1 = float((dev[:, :, 1:] * dev[:, :, :-1]).sum() / dev.pow(2).sum())
    q = torch.quantile(dev.flatten()[torch.randperm(dev.numel(), generator=torch.Generator().manual_seed(0))[:2_000_000]].float(),
                       torch.tensor([0.001, 0.01, 0.25, 0.5, 0.75, 0.99, 0.999])).tolist()

    def stats(ix):
        g = [per[i] for i in ix]
        sV, sP = sum(sum(r["V"]) for r in g), sum(sum(r["pcorr"]) for r in g)
        o = {"noise_share_V": sV / (sV + sP), "repeat_diff_over_V": sum(sum(r["repeat_diff"]) for r in g) / sV,
             "held_mean_corr_frac": sum(r["held"]["mean_corr"] for r in g) / sum(r["held"]["p_energy_corr"] for r in g),
             "held_split_frac": sum(r["held"]["split"] for r in g) / sum(r["held"]["p_energy_split"] for r in g),
             "held_single_frac": sum(r["held"]["single"] for r in g) / sum(r["held"]["single_energy"] for r in g)}
        for L in (1, 2, 4, 8):
            c = sum(r["lag"][L]["split_cross"] for r in g); a = sum(r["lag"][L]["split_a"] for r in g); b = sum(r["lag"][L]["split_b"] for r in g)
            o[f"p_corr_lag{L}"] = c / max(a * b, 1e-300) ** 0.5
            sc = sum(r["lag"][L]["single_cross"] for r in g); sa = sum(r["lag"][L]["single_a"] for r in g); sb = sum(r["lag"][L]["single_b"] for r in g)
            o[f"single_corr_lag{L}"] = sc / (sa * sb) ** 0.5
        return o
    est = stats(list(range(Pn)))
    gb = np.random.default_rng(0)
    boots = [stats(list(gb.integers(0, Pn, Pn))) for _ in range(2000)]
    cis = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in est}
    corr_ci = cis["held_mean_corr_frac"]
    return {"n_prompts": Pn, "R": Rn, "estimates": est, "ci95_prompt_bootstrap": cis,
            "independence": {"repeat_order_drift_ratio": drift, "successive_repeat_autocorr": ac1, "deviation_quantiles_0.1_1_25_50_75_99_99.9pct": q,
                             "reading": "drift ratio ~1 and autocorr ~ -1/(R-1) are expected for exchangeable repeats"},
            "increase_R": bool(corr_ci[0] < 1 < corr_ci[1] and (corr_ci[1] - corr_ci[0]) > 0.4), "per_prompt": per}


if __name__ == "__main__":
    {"repro": repro, "fresh": fresh, "traces": traces}[sys.argv[1]]()
