import torch
from math import pi

from .base import Energy


__all__ = ["MuellerPotential","WolfeQuappLike"]


class MuellerPotential(Energy):
    def __init__(self, dim=2, scale=0.15, yshift=0):
        super().__init__(dim)
        assert dim >= 2
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        self._scale = scale
        if not isinstance(yshift, torch.Tensor):
            yshift = torch.tensor(yshift)
        self._yshift = yshift

    def _energy(self, x):
        xx = x[:, [0]]
        yy = x[:, [1]] + self._yshift
        e1 = -200 * torch.exp(-(xx -1).pow(2) -10 * yy.pow(2))
        e1 += -100 * torch.exp(-xx.pow(2) -10 * (yy -0.5).pow(2))
        e1 += -170 * torch.exp(-6.5 * (0.5 + xx).pow(2) +11 * (xx +0.5) * (yy -1.5) -6.5 * (yy -1.5).pow(2))
        e1 += 15 * torch.exp(0.7 * (1 +xx).pow(2) +0.6 * (xx +1) * (yy -1) +0.7 * (yy -1).pow(2)) +146.7
        e1 *= self._scale
        v = x[:, 2:]
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2


class WolfeQuappLike(Energy):
    def __init__(self, dim=2, scale=2, theta=-0.15*pi):
        super().__init__(dim)
        assert dim >= 2
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        self._scale = scale
        if not isinstance(theta, torch.Tensor):
            theta = torch.tensor(theta)
        self._cos = torch.cos(theta)
        self._sin = torch.sin(theta)

    def _energy(self, x):
        xx = self._cos * x[:, [0]] - self._sin * x[:, [1]]
        yy = self._sin * x[:, [0]] + self._cos * x[:, [1]]
        e1 = xx.pow(4) + yy.pow(4) -2 * xx.pow(2) -4 * yy.pow(2) +2 * xx * yy +0.8 * xx +0.1 * yy +9.3
        e1 *= self._scale
        v = x[:, 2:]
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2
