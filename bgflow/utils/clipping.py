"""Gradient/Force clipping

Examples
--------


"""

__all__ = ["NoClipping", "ClipByValue", "ClipByAtom", "ClipBySample", "ClipByBatch"]


import torch.linalg
from typing import Union


class NoClipping:
    """Does nothing."""
    def __call__(self, grad):
        return grad


class ClipByValue:
    """Component-wise clipping."""
    def __init__(self, clip: Union[float, torch.Tensor]):
        self.clip = clip

    def __call__(self, grad: torch.Tensor):
        return torch.nan_to_num(grad, 0.0).clip(-self.clip, self.clip)


class ClipByAtom:
    """Clip atomic forces."""
    def __init__(self, clip: Union[float, torch.Tensor], n_dim: int = 3):
        self.clip = clip
        self.n_dim = n_dim

    def __call__(self, grad):
        original_shape = grad.shape
        grad = torch.nan_to_num(grad).reshape(-1, self.n_dim)
        norm = torch.linalg.norm(grad.detach(), dim=-1, keepdim=True)
        factor = torch.minimum(self.clip/norm, torch.ones_like(norm))
        grad = (grad * factor).reshape(original_shape)
        return grad


class ClipBySample:
    """Clip norm across event dims."""
    def __init__(self, clip: Union[float, torch.Tensor], n_event_dims):
        self.clip = clip
        self.n_event_dims = n_event_dims

    def __call__(self, grad):
        original_shape = grad.shape
        batch_shape = original_shape[:-self.n_event_dims]
        grad = torch.nan_to_num(grad).reshape(*batch_shape, -1)
        norm = torch.linalg.norm(grad.detach(), dim=-1, keepdim=True)
        factor = torch.minimum(self.clip/norm, torch.ones_like(norm))
        grad = (grad * factor).reshape(original_shape)
        return grad


class ClipByBatch:
    """Clip norm of the whole gradient tensor."""
    def __init__(self, clip: Union[float, torch.Tensor]):
        self.clip = clip

    def __call__(self, grad):
        original_shape = grad.shape
        grad = torch.nan_to_num(grad).reshape(-1)
        norm = torch.linalg.norm(grad.detach())
        factor = torch.minimum(self.clip/norm, torch.ones_like(norm))
        grad = (grad * factor).reshape(original_shape)
        return grad
