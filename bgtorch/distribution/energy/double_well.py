from .base import Energy


class DoubleWellEnergy(Energy):
    def __init__(self, dim, a=0, b=-4., c=1.):
        super().__init__(dim)
        self._a = a
        self._b = b
        self._c = c
    
    def _energy(self, x):
        d = x[:, [0]]
        v = x[:, 1:]
        e1 = self._a * d + self._b * d.pow(2) + self._c * d.pow(4)
        e2 = 0.5 * v.pow(2).sum(dim=-1, keepdim=True)
        return e1 + e2