"""snk1 determinism follow-up (bounded): Q repeat variance with the real padding mask vs an all-ones mask vs unpadded (truncated) tokens."""
import inspect, json, sys, os, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sana_a30 as S
with torch.no_grad(), S.Run("det2") as r:
    pipe, R, _ = S.load()
    P = S.prompts()
    d = inspect.signature(type(pipe).__call__).parameters
    kw = {k: d[k].default for k in ("complex_human_instruction", "max_sequence_length", "clean_caption")}
    pe, pm, ne, nm = pipe.encode_prompt(P[358], True, negative_prompt="", device="cuda", **kw)
    pipe.scheduler.set_timesteps(20, device="cuda")
    t = pipe.scheduler.timesteps[10].expand(2)
    x = torch.randn(2, 32, 32, 32, generator=torch.Generator().manual_seed(0)).to("cuda", torch.bfloat16)
    E = torch.cat([ne, pe]).to(torch.bfloat16); M = torch.cat([nm, pm])
    out = {"tokens": int(E.shape[1]), "valid_tokens": M.sum(1).tolist()}
    def rep(E_, M_, n=6):
        ys = [R.Q(x, encoder_hidden_states=E_, encoder_attention_mask=M_, timestep=t, return_dict=False)[0].float() for _ in range(n)]
        torch.cuda.synchronize()
        return {"bitwise": all(torch.equal(ys[0], y) for y in ys[1:]), "max_rel_l2": max(S.rel(y, ys[0]) for y in ys[1:])}
    out["real_mask"] = rep(E, M)
    out["ones_mask"] = rep(E, torch.ones_like(M))
    L = int(M.sum(1).max())
    out["truncated_to_max_valid"] = rep(E[:, :L].contiguous(), M[:, :L].contiguous())
    out["batch1_cond_only"] = None
    ys = [R.Q(x[:1], encoder_hidden_states=E[1:], encoder_attention_mask=M[1:], timestep=t[:1], return_dict=False)[0].float() for _ in range(6)]
    out["batch1_cond_only"] = {"bitwise": all(torch.equal(ys[0], y) for y in ys[1:]), "max_rel_l2": max(S.rel(y, ys[0]) for y in ys[1:])}
    out["gpu_h"] = r.hours()
S.jdump("determinism2.json", out)
print(json.dumps(out, indent=1))
