"""Inference storage for tensors with trailing shape [blocks, 16]."""
from dataclasses import dataclass
import torch


@dataclass(frozen=True, eq=False)
class BlockLayout:
    """Shared block assignment. Construct once per layer and device."""

    modes: torch.Tensor
    order: torch.Tensor
    inverse: torch.Tensor
    residual_count: int
    quantized_count: int

    @classmethod
    def from_modes(cls, modes, device="cpu"):
        # 0 = plain INT8, 1 = INT8 plus one residual, 2 = FP16.
        source = torch.as_tensor(modes, device="cpu")
        if source.ndim != 1 or not len(source) or not ((source == 0) | (source == 1) | (source == 2)).all():
            raise ValueError("provide one mode (0, 1 or 2) per block")
        source = source.to(torch.uint8, copy=True)
        groups = [torch.where(source == mode)[0] for mode in (1, 0, 2)]
        order = torch.cat(groups)
        return cls(source.to(device), order.to(device), order.argsort().to(device),
                   len(groups[0]), len(groups[0]) + len(groups[1]))

    @classmethod
    def uniform(cls, blocks, device="cpu"):
        return cls.from_modes(torch.ones(blocks, dtype=torch.uint8), device)

    @classmethod
    def balanced(cls, blocks, seed=0, device="cpu"):
        """One eighth FP16, three eighths INT8, and one half INT8 with a residual."""
        if blocks <= 0 or blocks % 8:
            raise ValueError("equal-byte mixtures require a positive multiple of eight blocks")
        rng = torch.Generator(device="cpu").manual_seed(seed)
        permutation = torch.randperm(blocks, generator=rng)
        modes = torch.ones(blocks, dtype=torch.uint8)
        modes[permutation[:blocks//8]] = 2
        modes[permutation[blocks//8:blocks//2]] = 0
        return cls.from_modes(modes, device)

    @property
    def shared_bytes(self):
        return sum(t.numel()*t.element_size() for t in (self.modes, self.order, self.inverse))


@dataclass(frozen=True, eq=False)
class PackedState:
    layout: BlockLayout
    codes: torch.Tensor
    scales: torch.Tensor
    indices: torch.Tensor
    residuals: torch.Tensor
    halves: torch.Tensor

    @property
    def payload_bytes(self):
        fields = (self.codes, self.scales, self.indices, self.residuals, self.halves)
        return sum(t.numel()*t.element_size() for t in fields)


@torch.no_grad()
def pack(state, layout):
    """Encode finite FP32 states in FP16 range. Leading dimensions are preserved."""
    if state.ndim < 2 or state.shape[-2:] != (len(layout.modes), 16):
        raise ValueError("state must end in [number of layout blocks, 16]")
    if state.dtype != torch.float32 or state.device != layout.modes.device:
        raise ValueError("expected FP32 input on the layout device")
    sorted_state = state.index_select(-2, layout.order)
    nq, nr = layout.quantized_count, layout.residual_count
    quantized = sorted_state[..., :nq, :]
    peak = quantized.abs().amax(-1, keepdim=True)
    scales = (peak * (1.0 / 127.0)).clamp_min(torch.finfo(torch.float32).tiny)
    codes = (quantized / scales).round().clamp(-127, 127).to(torch.int8)
    error = quantized[..., :nr, :] - codes[..., :nr, :].float() * scales[..., :nr, :]
    indices = error.abs().argmax(-1, keepdim=True)
    residuals = error.gather(-1, indices).half()
    return PackedState(layout, codes, scales, indices.byte(), residuals, sorted_state[..., nq:, :].half())


@torch.no_grad()
def unpack(state):
    """Return a fresh FP32 tensor in the model's original block order."""
    dense = state.codes.float() * state.scales
    corrected = dense[..., :state.layout.residual_count, :]
    corrected.scatter_add_(-1, state.indices.long(), state.residuals.float())
    return torch.cat((dense, state.halves.float()), dim=-2).index_select(-2, state.layout.inverse)
