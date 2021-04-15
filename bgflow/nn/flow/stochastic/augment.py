import torch

from bgflow.nn.flow.base import Flow


class StochasticAugmentation(Flow):
    def __init__(self, distribution):
        """
        Stochastic augmentation layer

        Adds additional coordinates to the state vector by sampling them from distribution.
        Pre-sampled momenta can be passed through the layer by the kwarg "momenta" and
        transformed momenta are returned if kwarg "return_momenta" is set True.
        If momenta are returned, their contribution is not added to the Jacobian.
        This behavior is required in some MCMC samplers.

        Parameters
        ----------
        distribution : Energy
            Energy object, needs sample and energy method.
        """
        super().__init__()
        self.distribution = distribution
        self._cached_momenta_forward = None
        self._cached_momenta_backward = None

    def _forward(self, q, **kwargs):
        batch_size = q.shape[0]
        temperature = kwargs.get("temperature", 1.0)
        cache_momenta = kwargs.get("cache_momenta", False)
        # Add option to pass pre-sampled momenta as key word argument
        p = kwargs.get("momenta", None)
        if p is None:
            p = self.distribution.sample(batch_size, temperature=temperature)
            dlogp = self.distribution.energy(p, temperature=temperature)
        else:
            dlogp = torch.zeros(p.shape[0], 1).to(p)
        if cache_momenta:
            self._cached_momenta_forward = p
        x = torch.cat([q, p], dim=1)
        return x, dlogp

    def _inverse(self, x, **kwargs):
        return_momenta = kwargs.get("return_momenta", False)
        cache_momenta = kwargs.get("cache_momenta", False)
        p = x[:, self.distribution.dim :]
        temperature = kwargs.get("temperature", 1.0)
        # Add option to return transformed momenta as key word argument.
        # Momenta will be returned in same tensor as configurations
        if cache_momenta:
            self._cached_momenta_backward = p
        if return_momenta:
            return x, torch.zeros(p.shape[0], 1).to(p)
        dlogp = self.distribution.energy(p, temperature=temperature)
        return x[:, : self.distribution.dim], -dlogp
