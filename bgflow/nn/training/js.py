

__all__ = ["jensen_shannon_divergence", "JensenShannonDivergence", "u_mixture", "logmeanexp", "logsumexp2"]


import torch
from typing import Union
import numpy as np
from bgflow.distribution.energy.base import Energy
from bgflow.bg import BoltzmannGenerator
from bgflow.utils.free_energy import bennett_acceptance_ratio
from bgflow.utils.types import pack_tensor_in_tuple


def jensen_shannon_divergence(
        ua_on_xa: torch.Tensor,
        ua_on_xb: torch.Tensor,
        ub_on_xa: torch.Tensor,
        ub_on_xb: torch.Tensor,
        free_energy_ua_to_ub: Union[torch.Tensor, float]
):
    """Jensen-Shannon divergence

    Parameters
    ----------
    ua_on_xa: torch.Tensor
        First energy function evaluated on samples from its corresponding distribution.
    ua_on_xb: torch.Tensor
        First energy function evaluated on samples from second distribution.
    ub_on_xa
        Second energy function evaluated on samples from first distribution.
    ub_on_xb: torch.Tensor
        Second energy function evaluated on samples from its corresponding distribution.
    free_energy_ua_to_ub:
        Free energy difference between the two energy functions.
        If :math:`F_a = -ln Z_a`, and math:`F_b = -ln Z_b`,
        the free energy is :math:`F_b - F_a`

    Returns
    -------
    JS-divergence : torch.Tensor
    """
    umix_on_xa = u_mixture(ua_on_xa, ub_on_xa - free_energy_ua_to_ub)
    umix_on_xb = u_mixture(ua_on_xb, ub_on_xb - free_energy_ua_to_ub)
    return 0.5*(- ua_on_xa.mean() - (ub_on_xb.mean() - free_energy_ua_to_ub) + umix_on_xb.mean() + umix_on_xa.mean())


class JensenShannonDivergence:
    """Jensen-Shannon divergence loss.

    Parameters
    ----------
    generator : BoltzmannGenerator
        The normalizing flow.
    target : Energy, Sampler, optional
        An (unnormalized) energy that we can sample from.
        This can for example be created by creating a `bgflow.CustomDistribution`
        of some Energy module with an iterative sampler, a dataset sampler, or
        a replay buffer sampler.
    target_free_energy : torch.Tensor
        Initial estimate for the absolute free energy of the target distribution.
    """
    def __init__(
            self,
            generator: BoltzmannGenerator,
            target: Energy = None,
            target_free_energy: torch.Tensor = 0.0
    ):
        self.generator = generator
        if target is None:
            target = self.generator._target
        if not hasattr(target, "sample"):
            raise ValueError("The target has to be an Energy and a Sampler. See documentation.")
        self.target = target
        self._target_free_energy = target_free_energy

    @property
    def target_free_energy(self):
        if self._target_free_energy is None:
            raise AttributeError(
                "JensenShannonDiv.target_free_energy not set. "
                "Call JensenShannonDiv.update_free_energy() first."
            )
        return self._target_free_energy

    @target_free_energy.setter
    def target_free_energy(self, delta_f):
        if isinstance(self._target_free_energy, torch.nn.Parameter):
            self._target_free_energy.data = delta_f
        else:
            self._target_free_energy = delta_f

    def __call__(self, n_samples_flow, n_samples_target=None, update_free_energy=False, **kwargs):
        if n_samples_target is None:
            n_samples_target = n_samples_flow
        *xa, ua_on_xa = self.generator.sample(n_samples_flow, with_energy=True)
        ub_on_xa = self.target.energy(*xa)
        xb = self.target.sample(n_samples_target)
        xb = pack_tensor_in_tuple(xb)
        ub_on_xb = self.target.energy(*xb)
        ua_on_xb = self.generator.energy(*xb)
        if update_free_energy:
            with torch.no_grad():
                free_energy, error_estimate = bennett_acceptance_ratio(
                    forward_work=(ub_on_xa - self.target_free_energy) - ua_on_xa,
                    reverse_work=ua_on_xb - (ub_on_xb - self.target_free_energy),
                    **kwargs,
                    compute_uncertainty=False
                )
                self.target_free_energy = self.target_free_energy + free_energy
        return jensen_shannon_divergence(
            ua_on_xa,
            ua_on_xb,
            ub_on_xa,
            ub_on_xb,
            self.target_free_energy
        )

    def update_free_energy(self, n_samples_flow, n_samples_target=None):
        """Update the target free energy through BAR."""
        self(n_samples_flow, n_samples_target, update_free_energy=True)


def logsumexp2(a: torch.Tensor, b: torch.Tensor):
    """Elementwise logsumexp between two tensors.

    Returns
    -------
    log(e^a + e^b): torch.Tensor
    """
    return torch.logsumexp(
        torch.stack([a[..., None], b[..., None]], dim=-1),
        dim=-1
    )[..., 0]


def u_mixture(u_p: torch.Tensor, u_q: torch.Tensor):
    """Energy of the mixture distribution between two *normalized* energies.
    """
    return - logsumexp2(-u_p, -u_q) + np.log(2)


def logmeanexp(a):
    return torch.logsumexp(a, dim=-1) - np.log(a.shape[-1])


def prob(u_p, u_m):
    return torch.exp(logmeanexp(- u_p + u_m))

