import torch


class Energy(torch.nn.Module):
    
    def __init__(self, dim):
        super().__init__()
        self._dim = dim
        
    @property
    def dim(self):
        return self._dim
    
    def _energy(self, x):
        raise NotImplementedError()
        
    def energy(self, x, temperature=None):
        assert x.shape[-1] == self._dim, "`x` does not match `dim`"
        if temperature is None:
            temperature = 1.
        return self._energy(x) / temperature
    
    def force(self, x, temperature=None):
        x = x.requires_grad_(True)
        e = self.energy(x, temperature=temperature)
        return -torch.autograd.grad(e.sum(), x, create_graph=True, retain_graph=True)[0]
