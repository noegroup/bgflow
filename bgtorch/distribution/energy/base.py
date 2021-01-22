import torch
import warnings


class Energy(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        if isinstance(dim, int):
            self._event_shape = torch.Size([dim])
        else:
            self._event_shape = torch.Size(dim)

    @property
    def dim(self):
        if len(self._event_shape) > 1:
            warnings.warn(
                "This Energy instance is defined on multidimensional events. "
                "Therefore, its Energy.dim is distributed over multiple tensor dimensions. "
                "Consider using Energy.event_shape instead.",
                UserWarning
            )
        return torch.prod(torch.tensor(self._event_shape, dtype=int))

    @property
    def event_shape(self):
        return self._event_shape

    def _energy(self, x):
        raise NotImplementedError()

    def energy(self, x, temperature=1.0):
        return self._energy(x) / temperature

    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x, create_graph=True, retain_graph=True)[0]

