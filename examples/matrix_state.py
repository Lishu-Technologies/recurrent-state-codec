"""A second adapter shape: store a small matrix-valued delta-rule state."""
import torch
from recurrent_state_codec import BlockLayout, pack, unpack


def delta_step(stored, key, value, decay):
    # One matrix per batch and head. Storage groups are independent of model axes.
    batch,heads,key_size=key.shape
    value_size=value.shape[-1]
    matrix=unpack(stored).reshape(batch,heads,key_size,value_size)*decay
    prediction=(matrix*key.unsqueeze(-1)).sum(dim=-2)
    matrix=matrix+key.unsqueeze(-1)*(value-prediction).unsqueeze(-2)
    return pack(matrix.reshape(batch,-1,16),stored.layout)


if __name__=='__main__':
    torch.manual_seed(0)
    state=torch.zeros(2,4,8,16)
    layout=BlockLayout.balanced(32)
    packed=pack(state.reshape(2,32,16),layout)
    packed=delta_step(packed,torch.randn(2,4,8),torch.randn(2,4,16),0.9)
    print(f'{packed.payload_bytes} payload bytes for two independent requests')
