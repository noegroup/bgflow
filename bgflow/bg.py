import torch
from .distribution.energy import Energy
from .distribution.sampling import Sampler
from .utils.types import pack_tensor_in_tuple

__all__ = [
    "BoltzmannGenerator", "unnormalized_kl_div", "unormalized_nll",
    "sampling_efficiency", "effective_sample_size", "log_weights",
    "log_weights_given_latent"
]


def unnormalized_kl_div(prior, flow, target, n_samples, temperature=1.0):
    z = prior.sample(n_samples, temperature=temperature)
    z = pack_tensor_in_tuple(z)
    *x, dlogp = flow(*z, temperature=temperature)
    return target.energy(*x, temperature=temperature) - dlogp


def unormalized_nll(prior, flow, *x, temperature=1.0):
    *z, neg_dlogp = flow(*x, inverse=True, temperature=temperature)
    return prior.energy(*z, temperature=temperature) - neg_dlogp


def log_weights(*x, prior, flow, target, temperature=1.0, normalize=True):
    *z, neg_dlogp = flow(*x, inverse=True, temperature=temperature)
    return log_weights_given_latent(
        x, z, -neg_dlogp, prior, target, temperature=temperature, normalize=normalize
    )

def log_weights_from_samples(prior, flow, target, num_samples, batch_size, temperature=1.0, normalize=True):
    """sample a bunch of datapoints, and compute their log_weights"""

    z=[]
    x=[]
    dlogp=[]
    with torch.no_grad():
        for batch in range(num_samples//batch_size):
            z_batch = prior.sample(batch_size)
            z.append(z_batch)
            x_batch, dlogp_batch = flow(*z_batch)
            x.append(x_batch)
            dlogp.append(dlogp_batch)
        z_cat =tuple([torch.cat([z_t[el] for z_t in z], dim = 0) for el in range(len(z[0]))])
        x = torch.cat(x)
        dlogp = torch.cat(dlogp)


    return log_weights_given_latent(
        x, z_cat, dlogp, prior, target, temperature=temperature, normalize=normalize
    )


def log_weights_given_latent(x, z, dlogp, prior, target, temperature=1.0, normalize=True):
    x = pack_tensor_in_tuple(x)
    z = pack_tensor_in_tuple(z)
    logw = (
        prior.energy(*z, temperature=temperature)
        + dlogp
        - target.energy(*x, temperature=temperature)
    )
    if normalize:
        logw = logw - torch.logsumexp(logw, dim=0)
    return logw.view(-1)


def effective_sample_size(log_weights):
    """Kish effective sample size; log weights don't have to be normalized"""
    return torch.exp(2*torch.logsumexp(log_weights, dim=0) - torch.logsumexp(2*log_weights, dim=0))


def sampling_efficiency(log_weights):
    """Kish effective sample size / sample size; log weights don't have to be normalized"""
    return effective_sample_size(log_weights) / len(log_weights)


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
            Target distribution implementing the energy() function
        """
        super().__init__(
            target.event_shapes if target is not None else prior.event_shapes
        )
        self._prior = prior
        self._flow = flow
        self._target = target

    @property
    def flow(self):
        return self._flow

    @property
    def prior(self):
        return self._prior

    def sample(
        self,
        n_samples,
        temperature=1.0,
        with_latent=False,
        with_dlogp=False,
        with_energy=False,
        with_log_weights=False,
        with_weights=False,
    ):
        z = self._prior.sample(n_samples, temperature=temperature)
        z = pack_tensor_in_tuple(z)
        *x, dlogp = self._flow(*z, temperature=temperature)
        results = list(x)

        if with_latent:
            results.append(*z)
        if with_dlogp:
            results.append(dlogp)
        if with_energy or with_log_weights or with_weights:
            bg_energy = self._prior.energy(*z, temperature=temperature) + dlogp
            if with_energy:
                results.append(bg_energy)
            if with_log_weights or with_weights:
                target_energy = self._target.energy(*x, temperature=temperature)
                log_weights = bg_energy - target_energy
                if with_log_weights:
                    results.append(log_weights)
                if with_weights:
                    weights = torch.softmax(log_weights, dim=0).view(-1)
                    results.append(weights)
        if len(results) > 1:
            return (*results,)
        else:
            return results[0]

    def energy(self, *x, temperature=1.0):
        return unormalized_nll(self._prior, self._flow, *x, temperature=temperature)

    def kldiv(self, n_samples, temperature=1.0):
        return unnormalized_kl_div(
            self._prior, self._flow, self._target, n_samples, temperature=temperature
        )

    def log_weights(self, *x, temperature=1.0, normalize=True):
        return log_weights(
            *x,
            prior=self._prior,
            flow=self._flow,
            target=self._target,
            temperature=temperature,
            normalize=normalize
        )

    def log_weights_given_latent(self, x, z, dlogp, temperature=1.0, normalize=True):
        return log_weights_given_latent(
            x, z, dlogp, self._prior, self._target, temperature=temperature, normalize=normalize
        )

    def trigger(self, function_name):
        return self.flow.trigger(function_name)
