import json, torch
from boa_dit.common import Run
from boa_dit.local_refresh import run as R
from boa_dit.weight_anchor.run import guided
from boa_dit.model import NULL_CLASS
from boa_dit.sample import CTX
dev="cuda:0"
with Run("ler1","checks_c4_diagnosis",dev,{}) as run:
    fp=R.load_fp(dev); wref=R.ref_weights(fp,R.POOL,dev); del fp
    B=R.build(dev,"fb",wref,R.POOL); m,jq,Rf=B["m"],B["jq"],B["R"]
    n=4; xs=torch.randn(n,4,32,32,generator=torch.Generator().manual_seed(1)).to(dev); ts=torch.full((n,),500,device=dev,dtype=torch.long)
    lab=torch.as_tensor(R.labels_for_seeds(R.SEL[:4]),device=dev); yy=torch.cat([lab,torch.full_like(lab,NULL_CLASS)])
    Rf.configure(None); jq.cap_keys=set(R.POOL); jq.zero_state=True; CTX.step,CTX.n=20,n
    with torch.no_grad():
        guided(m,xs,ts,yy,n)
        out={}
        for k in R.POOL:
            h=jq.h[k].float(); lin=R.lin_of(m,k); fn=lin._forward_hooks[B["carry_hooks"][k].id]
            with torch.autocast("cuda",enabled=False):
                full=fn(lin,(h,),lin.forward(h))                 # module incl. bias, no subtraction
                helper=Rf.mh(k,h)
                nob=fn(lin,(h,),torch.nn.functional.linear(h,lin.weight))   # module GEMM without bias
            out[k]={"with_bias_compare_rel_l2":R.rl2(helper+lin.bias.float(),full),"subtract_bias_rel_l2":R.rl2(full-lin.bias.float(),helper),
                    "no_bias_gemm_rel_l2":R.rl2(nob,helper),"bias_over_output_norm":float(lin.bias.float().norm()*h[...,0].numel()**0.5/helper.norm())}
    out.update(run.record())
json.dump(out,open("results/local_error_refresh/checks_c4_diagnosis.json","w"),indent=1)
for k,v in out.items(): print(k,v)
