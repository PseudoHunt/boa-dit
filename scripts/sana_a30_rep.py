"""snk1b: INT4 repeatability follow-up (results/sana_a30/PROTOCOL.md, follow-up section). Ledger tag snk1b.
  loc   : repeat / mutation / RNG checks at fixed states; router vs bare module; per-layer isolation and chained divergence of the Nunchaku block stack
  split : split-half separation of the reproducible discrepancy p from call-specific repeat noise over all 20 states of 16 dev prompts
Run: /home/nk_env/bin/python scripts/sana_a30_rep.py loc|split
"""
import inspect
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sana_a30 as S

S.TAG = "snk1b"
S.CAP_H = 0.5
NREP = 16
LOC_PROMPTS, LOC_STEPS = [352, 356, 361, 365], [0, 10, 19]


def traj_states(pipe, R, emb, seed):
    xs = {}

    def cb(p, i, t, kw):
        xs[i + 1] = kw["latents"].clone()
        return {}
    R.mode, R.anchors = "H", set()
    R.reset()
    pipe(**emb, negative_prompt=None, height=S.RES, width=S.RES, guidance_scale=S.GUIDE, num_inference_steps=S.STEPS,
         generator=torch.Generator().manual_seed(seed), output_type="latent", callback_on_step_end=cb, callback_on_step_end_tensor_inputs=["latents"])
    xs[0] = pipe.prepare_latents(1, 32, S.RES, S.RES, torch.float32, "cuda", torch.Generator().manual_seed(seed))
    pipe.scheduler.set_timesteps(S.STEPS, device="cuda")
    return xs, pipe.scheduler.timesteps


def inputs(R, emb, x, t):
    pe = torch.cat([emb["negative_prompt_embeds"], emb["prompt_embeds"]]).to(torch.bfloat16)
    pm = torch.cat([emb["negative_prompt_attention_mask"], emb["prompt_attention_mask"]])
    return torch.cat([x] * 2).to(torch.bfloat16), dict(encoder_hidden_states=pe, encoder_attention_mask=pm,
                                                         timestep=t.expand(2) * R.config.timestep_scale, return_dict=False)


def spread(ys):
    """relative spread of repeated outputs: max rel L2 vs the first, mean rel L2 vs the mean, all-bitwise flag."""
    Y = torch.stack([y.double() for y in ys])
    mu = Y.mean(0)
    return {"bitwise": all(torch.equal(ys[0], y) for y in ys[1:]), "max_rel_vs_first": max(S.rel(y, ys[0]) for y in ys[1:]),
            "mean_rel_vs_mean": float(np.mean([S.rel(y, mu) for y in Y])), "frac_elements_differing": float((Y != Y[0]).any(0).float().mean())}


@torch.no_grad()
def loc():
    S.guard("loc", 0.06)
    with S.Run("loc") as run_:
        pipe, R, _ = S.load()
        P = S.prompts()
        out = {"states": {}, "nrep": NREP}
        for s in LOC_PROMPTS:
            emb = S.embed(pipe, P[s])
            xs, ts = traj_states(pipe, R, emb, s)
            for i in LOC_STEPS:
                x, kw = inputs(R, emb, xs[i], ts[i])
                ref_in = {"x": x.clone(), **{k: v.clone() for k, v in kw.items() if torch.is_tensor(v)}}
                rng = (torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone())
                rec = {}
                for name, f in (("Q_bare", lambda: R.Q(x, **kw)[0]), ("Q_router", None), ("H_bare", lambda: R.H(x, **kw)[0])):
                    ys = []
                    for _ in range(NREP):
                        if name == "Q_router":
                            R.mode, R.anchors = "Q", set()
                            R.reset()
                            y = R(x, **kw)[0]
                        else:
                            y = f()
                        torch.cuda.synchronize()
                        ys.append(y.detach().clone().float())
                    rec[name] = spread(ys)
                    if name == "Q_bare":
                        qmean = torch.stack(ys).mean(0)
                    if name == "H_bare":
                        rec["Q_mean_vs_H_rel"] = S.rel(qmean, ys[0])
                rec["inputs_unchanged"] = bool(torch.equal(ref_in["x"], x) and all(torch.equal(ref_in[k], kw[k]) for k in ref_in if k != "x"))
                rec["rng_unchanged"] = bool(torch.equal(rng[0], torch.get_rng_state()) and torch.equal(rng[1], torch.cuda.get_rng_state()))
                out["states"][f"{s}@{i}"] = rec
                print(f"  {s}@{i}: Q {rec['Q_bare']['max_rel_vs_first']:.4f} router {rec['Q_router']['max_rel_vs_first']:.4f} H {rec['H_bare']['bitwise']} "
                      f"inputs {rec['inputs_unchanged']} rng {rec['rng_unchanged']}", flush=True)
        # per-layer localization at two states
        blk = R.Q.transformer_blocks[0]
        sig = inspect.signature(blk.forward)
        cap = {}

        def pre(mod, args, kwargs):
            cap["b"] = sig.bind(*args, **kwargs)
            cap["b"].apply_defaults()
        layers = {}
        for s, i in ((352, 10), (361, 0)):
            emb = S.embed(pipe, P[s])
            xs, ts = traj_states(pipe, R, emb, s)
            x, kw = inputs(R, emb, xs[i], ts[i])
            h = blk.register_forward_pre_hook(pre, with_kwargs=True)
            full_out = R.Q(x, **kw)[0]
            h.remove()
            a = dict(cap["b"].arguments)
            a = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()}
            a.pop("skip_first_layer", None)
            stack_ref = blk(**{k: (v.clone() if torch.is_tensor(v) else v) for k, v in a.items()})
            h0 = a.pop("hidden_states")
            L = blk.m.numLayers() if hasattr(blk.m, "numLayers") else 20
            ref_in, hcur = [], h0.clone()
            for idx in range(L):
                ref_in.append(hcur.clone())
                hcur = blk.forward_layer_at(idx, hcur, **a)
            chain_vs_stack = S.rel(hcur.float(), stack_ref.float())
            iso = []
            for idx in range(L):
                ys = [blk.forward_layer_at(idx, ref_in[idx].clone(), **a).float().clone() for _ in range(NREP)]
                torch.cuda.synchronize()
                iso.append(spread(ys))
            chains = []
            for _ in range(NREP):
                hc, per = h0.clone(), []
                for idx in range(L):
                    hc = blk.forward_layer_at(idx, hc, **a)
                    per.append(hc.float().clone())
                chains.append(per)
            chained = [spread([c[idx] for c in chains]) for idx in range(L)]
            first = next((idx for idx, v in enumerate(iso) if not v["bitwise"]), None)
            layers[f"{s}@{i}"] = {"n_layers": L, "chain_vs_stack_rel": chain_vs_stack, "first_nonbitwise_layer_isolated": first,
                                  "isolated": iso, "chained": chained, "hidden_shape": list(h0.shape)}
            print(f"  layers {s}@{i}: first non-bitwise (isolated) {first}; isolated max-rel per layer "
                  f"{[round(v['max_rel_vs_first'], 4) for v in iso]}; chained {[round(v['max_rel_vs_first'], 4) for v in chained]}", flush=True)
        out["layers"] = layers
        out["gpu_h"] = run_.hours()
    S.jdump("rep_loc.json", out)


@torch.no_grad()
def split():
    S.guard("split", 0.20)
    with S.Run("split") as run_:
        pipe, R, _ = S.load()
        P = S.prompts()
        g = lambda o: (lambda u, c: u + S.GUIDE * (c - u))(*o.float().chunk(2))
        an = S.SCHED["K4"]
        held = [i for i in range(S.STEPS) if i not in an]
        anc = lambda i: max(a for a in an if a <= i)
        per = []
        for j, s in enumerate(S.DEVP):
            emb = S.embed(pipe, P[s])
            xs, ts = traj_states(pipe, R, emb, s)
            m1, m2, e1, V = [], [], [], []
            for i in range(S.STEPS):
                x, kw = inputs(R, emb, xs[i], ts[i])
                hh = g(R.H(x, **kw)[0])
                es = torch.stack([(g(R.Q(x, **kw)[0]) - hh).double() for _ in range(NREP)])
                m1.append(es[:NREP // 2].mean(0)); m2.append(es[NREP // 2:].mean(0)); e1.append(es[0])
                V.append(float((es - es.mean(0)).pow(2).sum() / (NREP - 1)))
            dot = lambda a, b: float((a * b).sum())
            r = {"prompt": s, "V": V,
                 "pp": [dot(m1[i], m2[i]) for i in range(S.STEPS)],                                     # unbiased ||p_i||^2
                 "single_raw": [dot(e1[i], e1[i]) for i in range(S.STEPS)],
                 "lag": {L: {"cross": sum(0.5 * (dot(m1[i], m2[i + L]) + dot(m2[i], m1[i + L])) for i in range(S.STEPS - L)),
                             "a": sum(dot(m1[i], m2[i]) for i in range(S.STEPS - L)), "b": sum(dot(m1[i + L], m2[i + L]) for i in range(S.STEPS - L)),
                             "single_cross": sum(dot(e1[i], e1[i + L]) for i in range(S.STEPS - L)),
                             "single_a": sum(dot(e1[i], e1[i]) for i in range(S.STEPS - L)), "single_b": sum(dot(e1[i + L], e1[i + L]) for i in range(S.STEPS - L))}
                         for L in (1, 2, 4, 8)},
                 "held_p_resid": sum(dot(m1[i] - m1[anc(i)], m2[i] - m2[anc(i)]) for i in held),
                 "held_p_energy": sum(dot(m1[i], m2[i]) for i in held),
                 "held_pred_single": sum(dot(m1[i] - m1[anc(i)], m2[i] - m2[anc(i)]) + V[i] + V[anc(i)] for i in held),
                 "held_meas_single": sum(dot(e1[i] - e1[anc(i)], e1[i] - e1[anc(i)]) for i in held),
                 "held_single_energy": sum(dot(e1[i], e1[i]) for i in held)}
            per.append(r)
            print(f"  prompt {s} ({j + 1}/16): noise share {sum(V) / (sum(V) + sum(r['pp'])):.3f} held p-resid {r['held_p_resid'] / r['held_p_energy']:.3f}", flush=True)

        def stats(idx):
            q = [per[k] for k in idx]
            sV, sP = sum(sum(r["V"]) for r in q), sum(sum(r["pp"]) for r in q)
            o = {"noise_share": sV / (sV + sP), "held_p_resid_frac": sum(r["held_p_resid"] for r in q) / sum(r["held_p_energy"] for r in q),
                 "held_single_pred_frac": sum(r["held_pred_single"] for r in q) / sum(r["held_single_energy"] for r in q),
                 "held_single_meas_frac": sum(r["held_meas_single"] for r in q) / sum(r["held_single_energy"] for r in q),
                 "raw_single_over_pred": sum(sum(r["single_raw"]) for r in q) / (sV + sP)}
            for L in (1, 2, 4, 8):
                c = sum(r["lag"][L]["cross"] for r in q); a = sum(r["lag"][L]["a"] for r in q); b = sum(r["lag"][L]["b"] for r in q)
                o[f"p_corr_lag{L}"] = c / max(a * b, 1e-300) ** 0.5
                sc = sum(r["lag"][L]["single_cross"] for r in q); sa = sum(r["lag"][L]["single_a"] for r in q); sb = sum(r["lag"][L]["single_b"] for r in q)
                o[f"single_corr_lag{L}"] = sc / (sa * sb) ** 0.5
            return o
        allk = list(range(len(per)))
        est = stats(allk)
        gb = np.random.default_rng(0)
        boots = [stats(list(gb.integers(0, len(per), len(per)))) for _ in range(2000)]
        cis = {k: [float(np.percentile([b[k] for b in boots], 2.5)), float(np.percentile([b[k] for b in boots], 97.5))] for k in est}
        out = {"n_prompts": len(per), "nrep": NREP, "estimates": est, "ci95_prompt_bootstrap": cis, "per_prompt": per,
               "definitions": {"p": "reproducible discrepancy E[guided Q] - guided H at the H20 state (split-half cross estimators, unbiased)",
                               "V": "within-state repeat variance (trace), 16 repeats", "noise_share": "sum V / (sum V + sum ||p||^2)",
                               "held_p_resid_frac": "K4: sum_held ||p_t - p_a||^2 / sum_held ||p_t||^2 (mean discrepancy, offline)",
                               "held_single_pred_frac": "(||p_t - p_a||^2 + V_t + V_a) / single-call held energy (independent-noise prediction)",
                               "held_single_meas_frac": "measured single-call ||e_t - e_a||^2 / ||e_t||^2"},
               "note": "offline diagnostic; averaging repeated calls is not proposed for deployment; assumes independent repeat errors for the V decomposition",
               "gpu_h": run_.hours()}
    S.jdump("rep_split.json", out)
    print(json.dumps({"estimates": est, "ci": cis}, indent=1))


if __name__ == "__main__":
    {"loc": loc, "split": split}[sys.argv[1]]()
