"""Inference-only adapter for the public state-spaces/mamba-130m checkpoint."""
from dataclasses import dataclass
from pathlib import Path
import torch
from torch.nn import functional as F
from recurrent_state_codec import BlockLayout, PackedState, pack, unpack


@dataclass
class LayerState:
    history: torch.Tensor
    recurrent: torch.Tensor | PackedState


def rms(x, scale):
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-5) * scale


def read_state(state, coefficients):
    products = state * coefficients
    for distance in (8,4,2,1):
        products = products[..., :distance] + products[..., distance:]
    return products.squeeze(-1)


class Mamba130M:
    def __init__(self, checkpoint, device):
        self.device = torch.device(device)
        weights = torch.load(Path(checkpoint), weights_only=True, map_location="cpu")
        self.embedding = weights.pop("backbone.embedding.weight").to(device)
        self.head = weights.pop("lm_head.weight").to(device)
        self.norm = weights.pop("backbone.norm_f.weight").to(device)
        self.layers = []
        if self.embedding.shape != (50280,768):
            raise ValueError("This example supports the official Mamba-130M checkpoint")
        for i in range(24):
            prefix = f"backbone.layers.{i}."
            layer = {k[len(prefix):]: weights.pop(k).to(device) for k in list(weights) if k.startswith(prefix)}
            layer["decay"] = -layer.pop("mixer.A_log").exp()
            self.layers.append(layer)
        if weights:
            raise ValueError(f"unexpected weights: {list(weights)[:3]}")
        self.fixed = BlockLayout.uniform(1536, device)
        # Deliberately public, unfitted schedule. Six middle layers, seed fixed before scoring.
        self.layouts = [BlockLayout.balanced(1536, 42+i, device) if 12 <= i < 18 else self.fixed for i in range(24)]

    def hidden(self, ids, policy, *, cache=None, offset=0, switch=2048, accelerator=None):
        if policy not in ("fp32","fixed","mixed") or min(offset,switch)<0 or ids.shape[1]<1:
            raise ValueError("invalid policy, offset, switch or empty sequence")
        if (cache is None) != (offset == 0) or (cache is not None and len(cache)!=24):
            raise ValueError("provide all layer caches and their token offset together")
        if accelerator is not None and cache is not None:
            raise ValueError("accelerated scoring consumes one complete window")
        h = F.embedding(ids, self.embedding)
        next_cache = []
        for i,layer in enumerate(self.layers):
            previous = None if cache is None else cache[i]
            x,z = F.linear(rms(h,layer["norm.weight"]),layer["mixer.in_proj.weight"]).chunk(2,-1)
            x = x.transpose(1,2)
            history = x.new_zeros(*x.shape[:2],3) if previous is None else previous.history
            taps = torch.cat((history,x),dim=-1)
            u = F.silu(F.conv1d(taps,layer["mixer.conv1d.weight"],layer["mixer.conv1d.bias"],groups=x.shape[1])).transpose(1,2)
            dt,b,c = F.linear(u,layer["mixer.x_proj.weight"]).split((48,16,16),dim=-1)
            dt = F.softplus(F.linear(dt,layer["mixer.dt_proj.weight"],layer["mixer.dt_proj.bias"]))
            if accelerator is not None:
                mode = {"fp32":0,"fixed":1,"mixed":2}[policy]
                y = accelerator(u,dt,b,c,layer["decay"],self.layouts[i].modes,switch,mode)
            else:
                state = u.new_zeros(len(ids),1536,16) if previous is None else previous.recurrent
                readouts = []
                for t in range(ids.shape[1]):
                    dense = unpack(state) if isinstance(state,PackedState) else state
                    delta = dt[:,t,:,None]
                    dense = (delta*layer["decay"]).exp()*dense + (delta*u[:,t,:,None])*b[:,t,None,:]
                    coeff = c[:,t,None,:]
                    readout = (dense*coeff).sum(-1) if policy=='fp32' else read_state(dense,coeff)
                    readouts.append(readout)
                    if policy == "fp32":
                        state = dense
                    else:
                        layout = self.layouts[i] if policy=="mixed" and offset+t>=switch else self.fixed
                        state = pack(dense,layout)
                y = torch.stack(readouts,dim=1)
                next_cache.append(LayerState(taps[:,:,-3:].clone(),state))
            y = (y+u*layer["mixer.D"])*F.silu(z)
            h = h+F.linear(y,layer["mixer.out_proj.weight"])
        return rms(h,self.norm), next_cache

    def logits(self, hidden):
        return F.linear(hidden,self.head)
