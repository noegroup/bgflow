import torch

from .distribution.energy import Energy
from .distribution.sampling import Sampler


def unnormalized_kl_div(prior, flow, target, n_samples, temperature=None):
    z = prior.sample(n_samples, temperature=temperature)
    x, dlogp = flow(z, temperature=temperature)
    return target.energy(x, temperature=temperature) - dlogp


def unormalized_nll(prior, flow, x, temperature=None):
    z, dlogp = flow(x, inverse=True, temperature=temperature)
    return prior.energy(z, temperature=temperature) - dlogp


def log_weights(x, prior, flow, target, temperature=None):
    z, dlogp = flow(x, inverse=True, temperature=temperature)
    return log_weights_given_latent(z, dlogp, prior, flow, target, temperature=temperature)


def log_weights_given_latent(x, z, dlogp, prior, flow, target, temperature=None):
    logw = prior.energy(z, temperature=temperature) + dlogp - target.energy(x, temperature=temperature)
    logw = logw - logw.max()
    logw = logw - torch.logsumexp(logw, dim=0)
    return logw.view(-1)


class BoltzmannGenerator(Energy, Sampler):
    
    def __init__(self, prior, flow, target):
        """ Constructs Boltzmann Generator, i.e. normalizing flow to sample target density

        Parameters
        ----------
        prior : object
            Prior distribution implementing the energy() and sample() functions
        flow : Flow object
            Flow that can be evaluated forward and reverse
        target : object
            Prior distribution implementing the energy() function

        """
        super().__init__(prior.dim)
        self._prior = prior
        self._flow = flow
        self._target = target
        
    @property
    def flow(self):
        return self._flow
    
    @property
    def prior(self):
        return self._prior
        
    def sample(self, n_samples, temperature=None, with_latent=False, with_dlogp=False, with_energy=False):
        z = self._prior.sample(n_samples, temperature=temperature)
        x, dlogp = self._flow(z)
        results = [x]
        if with_latent:
            results.append(z)
        if with_dlogp:
            results.append(dlogp)
        if with_energy:
            energy = self._prior(z) + dlogp
            results.append(energy)
        if len(results) > 1:
            return (*results,)
        else:
            return results[0]
    
    def energy(self, x, temperature=None):
        return unormalized_nll(self._prior, self._flow, x, temperature=temperature)
    
    def kldiv(self, n_samples, temperature=None):
        return unnormalized_kl_div(self._prior, self._flow, self._target, n_samples, temperature=temperature)
    
    def log_weights(self, x, temperature=None):
        return log_weights(x, self._prior, self._flow, self._target, temperature=temperature)
    
    def log_weights_given_latent(self, x, z, dlogp, temperature=None):
        return log_weights_given_latent(x, z, dlogp, self._prior, self._flow, self._target, temperature=None)