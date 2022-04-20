"""Gradient/Force clipping

Examples
--------


"""

__all__ = ["ClipGradient"]


import torch
from typing import Union
from functools import partial


class ClipGradient(torch.nn.Module):
    """A module that clips the gradients in the backward pass.

    Parameters
    ----------
    clip
        the max norm
    norm_dim
        the dimension of the space over which the norm is computed
        - `1` corresponds to clipping by value
        - `3` corresponds to clipping by atom
        - `-1` corresponds to clipping the norm of the whole tensor
    """

    def __init__(self, clip: Union[float, torch.Tensor], norm_dim: int = 1):
        super().__init__()
        self.register_buffer("clip", torch.as_tensor(clip))
        self.norm_dim = norm_dim

    def forward(self, x):
        x.register_hook(partial(ClipGradient.clip_tensor, clip=self.clip, last_dim=self.norm_dim))
        return x

    @staticmethod
    def clip_tensor(tensor, clip, last_dim):
        clip.to(tensor)
        original_shape = tensor.shape
        last_dim = (-1, ) if last_dim == -1 else (-1, last_dim)
        out = torch.nan_to_num(tensor, nan=0.0).flatten().reshape(*last_dim)
        norm = torch.linalg.norm(out.detach(), dim=-1, keepdim=True)
        factor = (clip.view(-1, *clip.shape) / norm.view(-1, *clip.shape)).view(-1)
        factor = torch.minimum(factor, torch.ones_like(factor))
        out = out.view(*last_dim) * factor.view(-1, 1)
        out = out.reshape(original_shape)
        return out
