"""Custom activation functions."""
import torch
import torch.nn as nn

from vllm import activation_ops


class SiluAndMul(nn.Module):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        output_shape = (x.shape[:-1] + (d, ))
        out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
        activation_ops.silu_and_mul(out, x)
        return out


class NewGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        activation_ops.gelu_new(out, x)
        return out


class FastGELU(nn.Module):

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.empty_like(x)
        activation_ops.gelu_fast(out, x)
        return out

class SwiGLUActivation(nn.Module):
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 * nn.functional.silu(x2)

_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU(),
    "gelu_fast": FastGELU(),
    "gelu_new": NewGELU(),
    "gelu_pytorch_tanh": nn.GELU(approximate="tanh"),
    "relu": nn.ReLU(),
    "swiglu": SwiGLUActivation(),
}


def get_act_fn(act_fn: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn = act_fn.lower()
    if act_fn in _ACTIVATION_REGISTRY:
        return _ACTIVATION_REGISTRY[act_fn]
    raise ValueError(f"Activation function {act_fn!r} is not supported.")
