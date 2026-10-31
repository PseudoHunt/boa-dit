"""snk1 determinism diagnostic: repeated single Q/H calls on identical inputs; embedding path vs prompt path. Ledgered as snk1/det."""
import inspect, json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sana_a30 as S
with torch.no_grad(), S.Run("det") as r:
    pipe, R, _ = S.load()
    P = S.prompts()
    d = inspect.signature(type(pipe).__call__).parameters
    defaults = {k: d[k].default for k in ("complex_human_instruction", "max_sequence_length", "clean_caption")}
    pe, pm, ne, nm = pipe.encode_prompt(P[358], True, negative_prompt="", device="cuda")
    pe2, pm2, ne2, nm2 = pipe.encode_prompt(P[358], True, negative_prompt="", device="cuda", **defaults)
    out = {"encode_defaults_differ_from_call_defaults": not torch.equal(pe, pe2), "call_defaults": {k: str(v)[:80] for k, v in defaults.items()}}
    pipe.scheduler.set_timesteps(20, device="cuda")
    t = pipe.scheduler.timesteps[10].expand(2)
    x = torch.randn(2, 32, 32, 32, generator=torch.Generator().manual_seed(0)).to("cuda", torch.bfloat16)
    E = torch.cat([ne2, pe2]).to(torch.bfloat16); M = torch.cat([nm2, pm2])
    for name, m in (("H", R.H), ("Q", R.Q)):
        ys = [m(x, encoder_hidden_states=E, encoder_attention_mask=M, timestep=t, return_dict=False)[0].float() for _ in range(6)]
        out[name] = {"bitwise_all": all(torch.equal(ys[0], y) for y in ys[1:]), "max_rel_l2": max(S.rel(y, ys[0]) for y in ys[1:]),
                     "max_abs": max(float((y - ys[0]).abs().max()) for y in ys[1:]), "out_norm_rel_QvsH": None}
    yh = R.H(x, encoder_hidden_states=E, encoder_attention_mask=M, timestep=t, return_dict=False)[0].float()
    yq = R.Q(x, encoder_hidden_states=E, encoder_attention_mask=M, timestep=t, return_dict=False)[0].float()
    out["single_call_rel_l2_Q_vs_H"] = S.rel(yq, yh)
    out["gpu_h"] = r.hours()
S.jdump("determinism.json", out)
print(json.dumps(out, indent=1))
