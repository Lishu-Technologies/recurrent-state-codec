"""Check the Mamba adapter against Transformers, and scoring against physical state writes."""
import argparse
import json
from pathlib import Path
import torch
from .mamba import Mamba130M, read_state
from recurrent_state_codec import BlockLayout, pack, unpack
from .cuda_scan import Scan


@torch.inference_mode()
def run(checkpoint,out):
    from transformers import MambaConfig, MambaForCausalLM
    torch.manual_seed(704)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    device="cuda"
    public=Mamba130M(checkpoint,device)
    params=torch.load(checkpoint,weights_only=True,map_location='cpu')
    params['backbone.embeddings.weight']=params.pop('backbone.embedding.weight')
    hf=MambaForCausalLM(MambaConfig(vocab_size=50280,hidden_size=768,num_hidden_layers=24,
                        residual_in_fp32=True,use_associative_scan=False)).eval()
    hf.load_state_dict(params,strict=True)
    hf.to(device)
    del params
    scan=Scan()
    ids=torch.randint(1,50000,(2,48),device=device)
    gold=hf(ids,use_cache=False).logits
    h,_=public.hidden(ids,'fp32')
    candidate=public.logits(h)
    torch.testing.assert_close(gold,candidate,atol=2e-3,rtol=2e-4)
    stats={'hf_vs_adapter_max_abs':float((gold-candidate).abs().max()),'positions':ids.numel(),'policies':{}}
    # With fixed coefficients, compare a real packed write on every step through a long switch.
    steps, width = 2112, 16
    u = torch.randn(2,steps,width,device=device) / 8
    dt = torch.rand_like(u) / 16
    bvec, cvec = torch.randn_like(u), torch.randn_like(u)
    decay = -torch.ones(width,16,device=device)
    fixed = BlockLayout.uniform(width,device)
    mixed = BlockLayout.balanced(width,17,device)
    fast_layer = scan(u,dt,bvec,cvec,decay,mixed.modes,2048,2)
    previous = pack(torch.zeros(2,width,16,device=device),fixed)
    outputs = []
    for t in range(steps):
        delta = dt[:,t,:,None]
        value = (delta*decay).exp()*unpack(previous)+(delta*u[:,t,:,None])*bvec[:,t,None,:]
        outputs.append(read_state(value,cvec[:,t,None,:]))
        previous = pack(value,fixed if t<2048 else mixed)
    reference_layer = torch.stack(outputs,1)
    torch.testing.assert_close(reference_layer,fast_layer,atol=1e-5,rtol=1e-5)
    stats['long_layer_max_abs'] = (reference_layer-fast_layer).abs().max().item()
    for k in ('fp32','fixed','mixed'):
        full,cache=public.hidden(ids,k,switch=16)
        fast,_=public.hidden(ids,k,switch=16,accelerator=scan)
        # Independent projection shapes and reduction orders can straddle a quantizer threshold.
        a,b=public.logits(full),public.logits(fast)
        rmse=(a-b).square().mean().sqrt().item()
        top_matches=int((a.argmax(-1)==b.argmax(-1)).sum())
        if rmse>0.01 or top_matches<ids.numel()-1:
            raise ValueError(f'{k}: accelerator parity outside tolerance, RMSE {rmse}, hits {top_matches}')
        head,part=public.hidden(ids[:,:16],k,switch=16)
        tail,_=public.hidden(ids[:,16:],k,cache=part,offset=16,switch=16)
        split=public.logits(torch.cat((head,tail),1))
        diff=(a-split).square().mean().sqrt().item()
        # A different GEMM shape can shift all vocabulary logits by a common amount.
        # Measure its impact on the distribution and also retain the raw difference.
        logp_diff = (a.log_softmax(-1)-split.log_softmax(-1)).square().mean().sqrt().item()
        chunk_top = int((a.argmax(-1)==split.argmax(-1)).sum())
        if logp_diff>0.02 or chunk_top<ids.numel()-2:
            raise ValueError(f'{k}: chunk log-probability RMSE {logp_diff}, top matches {chunk_top}')
        stats['policies'][k]=dict(scan_logit_rmse=rmse,scan_top_matches=top_matches,chunk_logit_rmse=diff,
                                 chunk_logprob_rmse=logp_diff,chunk_top_matches=chunk_top)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(stats,indent=2)+'\n',encoding='utf8')
    print(stats)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    run(args.checkpoint,args.output)
