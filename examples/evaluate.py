"""Score matched next-token positions and measure greedy decode with physical caches."""
import argparse
import hashlib
import json
import platform
import statistics
import struct
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from recurrent_state_codec import PackedState
from .assets import MODEL_SHA,TOKENIZER_SHA,digest,panel,fetch
from .mamba import Mamba130M


def sync(device):
    if device.type=='cuda':
        torch.cuda.synchronize(device)


def score(model, ids, policy, switch, scan):
    hidden,_ = model.hidden(ids[:,:-1],policy,switch=switch,accelerator=scan)
    losses,hits = [],[]
    for start in range(0,hidden.shape[1],64):
        end = min(start+64,hidden.shape[1])
        logits = model.logits(hidden[:,start:end])
        gold = ids[:,start+1:end+1]
        losses.append(F.cross_entropy(logits.flatten(0,1),gold.reshape(-1),reduction='none'))
        hits.append(logits.argmax(-1).flatten()==gold.flatten())
    return torch.cat(losses).double().cpu(),torch.cat(hits).cpu()


def benchmark_decode(model,ids,switch,steps):
    # Exercise both physical layouts in a small public eager benchmark.
    # Override the switch in this latency-only workload to keep setup short.
    prompt = ids[:,:32]
    stats = {p:[] for p in ('fp32','fixed','mixed')}
    memory = {}
    for repeat in range(4):
        keys=list(stats)
        for policy in keys[repeat%3:]+keys[:repeat%3]:
            hidden,cache=model.hidden(prompt,policy,switch=switch)
            token=model.logits(hidden[:,-1:]).argmax(-1)
            memory[policy]=sum(v.recurrent.payload_bytes if isinstance(v.recurrent,PackedState)
                               else v.recurrent.numel()*v.recurrent.element_size() for v in cache)
            sync(model.device)
            begin=time.perf_counter()
            for j in range(steps):
                hidden,cache=model.hidden(token,policy,cache=cache,offset=32+j,switch=switch)
                token=model.logits(hidden).argmax(-1)
            sync(model.device)
            if repeat:
                stats[policy].append(time.perf_counter()-begin)
    return dict(batch=1,prompt=32,switch=switch,steps=steps,wall_seconds=stats,
                tokens_per_second={k:steps/statistics.median(v) for k,v in stats.items()},ssm_payload_bytes=memory)


@torch.inference_mode()
def main(args):
    if args.length<1 or args.switch<0 or args.decode_steps<1:
        raise ValueError("length and decode steps must be positive; switch must be nonnegative")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    if args.fetch:
        fetch(args.assets)
    ckpt=args.assets/'pytorch_model.bin'
    if digest(ckpt)!=MODEL_SHA or digest(args.assets/'tokenizer.json')!=TOKENIZER_SHA:
        raise ValueError("unexpected model or tokenizer SHA256")
    model=Mamba130M(ckpt,args.device)
    scan=None
    if args.backend=='nvrtc':
        if model.device.type!='cuda':
            raise ValueError("NVRTC needs CUDA")
        from .cuda_scan import Scan
        scan=Scan()
    records=[]
    for name,domain,tokens in panel(args.assets,args.length):
        ids=torch.tensor([tokens],device=model.device)
        record=dict(document=name,domain=domain,token_sha256=hashlib.sha256(struct.pack(f'<{len(tokens)}I',*tokens)).hexdigest())
        reference=None
        for k in ('fp32','fixed','mixed'):
            loss,hit=score(model,ids,k,args.switch,scan)
            if not torch.isfinite(loss).all():
                raise ValueError(f"nonfinite loss in {name}, {k}")
            record[k]=dict(nll=loss.mean().item(),hits=int(hit.sum()))
            if k=='fixed':
                reference=(loss,hit)
            if k=='mixed':
                boundary=min(len(loss),args.switch)
                if not torch.equal(loss[:boundary],reference[0][:boundary]) or not torch.equal(hit[:boundary],reference[1][:boundary]):
                    raise RuntimeError("pre-switch comparisons diverged")
        print(name,record['fixed'],record['mixed'],flush=True)
        records.append(record)
    groups={}
    for kind in ('fp32','fixed','mixed'):
        groups[kind]={}
        for domain in ('fiction','technical'):
            s=[r[kind] for r in records if r['domain']==domain]
            groups[kind][domain]=dict(nll=statistics.mean(v['nll'] for v in s),hits=sum(v['hits'] for v in s))
    result=dict(model_sha256=MODEL_SHA,tokenizer_sha256=TOKENIZER_SHA,python=platform.python_version(),torch=str(torch.__version__),
                device=torch.cuda.get_device_name() if model.device.type=='cuda' else 'CPU',backend=args.backend,
                length=args.length,switch=args.switch,documents=records,summary=groups,
                decode=benchmark_decode(model,ids,16,args.decode_steps))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--fetch',action='store_true',help='fetch public benchmark inputs missing from the assets directory')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    parser.add_argument('--backend',choices=('torch','nvrtc'),default='nvrtc')
    parser.add_argument('--length',type=int,default=8192)
    parser.add_argument('--switch',type=int,default=2048)
    parser.add_argument('--decode-steps',type=int,default=8)
    main(parser.parse_args())
