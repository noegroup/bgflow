import torch

from .base import Energy
from numpy import pi as PI


__all__ = ["DoubleWellEnergy", "MultiDimensionalDoubleWell", "MuellerEnergy", "ModifiedWolfeQuapp"]


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
    def __init__(self, dim=2, scale1=0.15, scale2=15, beta=1.):
        super().__init__(dim)
        assert dim >= 2
        self._scale1 = scale1
        self._scale2 = scale2
        self._beta = beta

    def _energy(self, x):
        xx = x[..., [0]]
        yy = x[..., [1]]
        e1 = -200 * torch.exp(-(xx -1).pow(2) -10 * yy.pow(2))
        e2 = -100 * torch.exp(-xx.pow(2) -10 * (yy -0.5).pow(2))
        e3 = -170 * torch.exp(-6.5 * (0.5 + xx).pow(2) +11 * (xx +0.5) * (yy -1.5) -6.5 * (yy -1.5).pow(2))
        e4 = 15.0 * torch.exp(0.7 * (1 +xx).pow(2) +0.6 * (xx +1) * (yy -1) +0.7 * (yy -1).pow(2)) +146.7
        v = x[..., 2:]
        ev = self._scale2 * 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return self._beta * (self._scale1 * (e1 + e2 + e3 + e4) + ev)

    @property
    def potential_str(self):
      pot_str = f'{self._scale1:g}*(-200*exp(-(x-1)^2-10*y^2)-100*exp(-x^2-10*(y-0.5)^2)-170*exp(-6.5*(0.5+x)^2+11*(x+0.5)*(y-1.5)-6.5*(y-1.5)^2)+15*exp(0.7*(1+x)^2+0.6*(x+1)*(y-1)+0.7*(y-1)^2)+146.7)'
      if self.dim >= 3:
        pot_str += f'+{self._scale2:g}*0.5*z^2'
      return pot_str

class ModifiedWolfeQuapp(Energy):
    def __init__(self, dim=2, theta=-0.3*PI/2, scale1=2, scale2=15, beta=1.):
        super().__init__(dim)
        assert dim >= 2
        self._scale1 = scale1
        self._scale2 = scale2
        self._beta = beta
        self._c = torch.cos(torch.as_tensor(theta))
        self._s = torch.sin(torch.as_tensor(theta))

    def _energy(self, x):
        xx = self._c * x[..., [0]] - self._s * x[..., [1]]
        yy = self._s * x[..., [0]] + self._c * x[..., [1]]
        e4 = xx.pow(4) + yy.pow(4)
        e2 = -2 * xx.pow(2) - 4 * yy.pow(2) + 2 * xx * yy
        e1 = 0.8 * xx + 0.1 * yy + 9.28
        v = x[..., 2:]
        ev = self._scale2 * 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return self._beta * (self._scale1 * (e4 + e2 + e1) + ev)

    @property
    def potential_str(self):
      x_str = f'({self._c:g}*x-{self._s:g}*y)'
      y_str = f'({self._s:g}*x+{self._c:g}*y)'
      pot_str = f'{self._scale1:g}*({x_str}^4+{y_str}^4-2*{x_str}^2-4*{y_str}^2+2*{x_str}*{y_str}+0.8*{x_str}+0.1*{y_str}+9.28)'
      if self.dim >= 3:
        pot_str += f'+{self._scale2:g}*0.5*z^2'
      return pot_str
