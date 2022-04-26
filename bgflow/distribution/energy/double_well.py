import torch

from .base import Energy


__all__ = ["DoubleWellEnergy", "MultiDimensionalDoubleWell", "MuellerEnergy"]


class DoubleWellEnergy(Energy):
    def __init__(self, dim, a=0, b=-4.0, c=1.0):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c

    def _energy(self, x):
        d = x[..., [0]]
        v = x[..., 1:]
        e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2


class MultiDimensionalDoubleWell(Energy):
    def __init__(self, dim, a=0.0, b=-4.0, c=1.0, transformer=None):
        super().__init__(dim)
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c)
        self.register_buffer("_a", a)
        self.register_buffer("_b", b)
        self.register_buffer("_c", c)
        if transformer is not None:
            self.register_buffer("_transformer", transformer)
        else:
            self._transformer = None

    def _energy(self, x):
        if self._transformer is not None:
            x = torch.matmul(x, self._transformer)
        e1 = self._a * x + self._b * x.pow(2) + self._c * x.pow(4)
        return e1.sum(dim=1, keepdim=True)


class MuellerEnergy(Energy):
    def __init__(self, dim=2, scale=0.15):
        super().__init__(dim)
        assert dim >= 2
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)
        self._scale = scale

    def _energy(self, x):
        xx = x[..., [0]]
        yy = x[..., [1]]
        e1 = -200 * torch.exp(-(xx -1).pow(2) -10 * yy.pow(2))
        e2 = -100 * torch.exp(-xx.pow(2) -10 * (yy -0.5).pow(2))
        e3 = -170 * torch.exp(-6.5 * (0.5 + xx).pow(2) +11 * (xx +0.5) * (yy -1.5) -6.5 * (yy -1.5).pow(2))
        e4 = 15 * torch.exp(0.7 * (1 +xx).pow(2) +0.6 * (xx +1) * (yy -1) +0.7 * (yy -1).pow(2)) +146.7
        v = x[..., 2:]
        ev = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return self._scale * (e1 + e2 + e3 + e4) + ev

    @property
    def potential_str(self):
      pot_str = f'{self._scale:g}*(-200*exp(-(x-1)^2-10*y^2)-100*exp(-x^2-10*(y-0.5)^2)-170*exp(-6.5*(0.5+x)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)+15*exp(0.7*(1+x)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2)+146.7)'
      if self.dim == 3:
        pot_str += '+0.5*z^2'
      return pot_str
