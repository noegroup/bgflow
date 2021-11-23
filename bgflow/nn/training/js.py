

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
        free_energy_ua_to_ub: Union[torch.Tensor, float],
        use_log_d_trick: bool = False
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
    free_energy_ua_to_ub: Union[torch.Tensor, float]
        Free energy difference between the two energy functions.
        If :math:`F_a = -ln Z_a`, and math:`F_b = -ln Z_b`,
        the free energy is :math:`F_b - F_a`
    use_log_d_trick: bool
        Whether to use the "logD" trick from the GAN paper instead of the classical JS divergence.

    Returns
    -------
    JS-divergence : torch.Tensor
    """
    umix_on_xa = u_mixture(ua_on_xa, ub_on_xa - free_energy_ua_to_ub)
    umix_on_xb = u_mixture(ua_on_xb, ub_on_xb - free_energy_ua_to_ub)
    expectation_xa = - ua_on_xa.mean() + umix_on_xa.mean()
    if use_log_d_trick:
        expectation_xb = - umix_on_xb.mean() + (ub_on_xa.mean() - free_energy_ua_to_ub)
    else:
        expectation_xb = - (ub_on_xb.mean() - free_energy_ua_to_ub) + umix_on_xb.mean()
    return 0.5*(expectation_xa + expectation_xb)


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

    Attributes
    ----------
    target_free_energy : Union[torch.Tensor, torch.nn.Parameter, float]
        The current estimate for the absolute free energy of the target distribution.
    result_dict : dict
        (only returned if `return_result_dict == True`) A dictionary containing the
        generated samples and computed energies. For example, the key "ugen_on_xref"
        denotes the energy of the generator evaluated on samples from the reference potential.
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

    def __call__(
            self,
            n_samples_flow: int,
            n_samples_target: int = None,
            update_free_energy: bool = False,
            return_result_dict: bool = False,
            **kwargs
    ):
        """

        Parameters
        ----------
        n_samples_flow : int
            Number of samples to draw from the generator.
        n_samples_target : int, optional
            Number of samples to draw from the target distribution; by default = n_samples_flow
        update_free_energy : bool, optional
            Whether to update the free energy using BAR.
        return_result_dict : bool, optional
            Whether to return the full result dictionary.
        kwargs : dict
            Any keyword arguments to `bennett_acceptance_ratio`.

        Returns
        -------
        js : torch.Tensor
            Jensen-Shannon divergence

        """
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
        js = jensen_shannon_divergence(
            ua_on_xa,
            ua_on_xb,
            ub_on_xa,
            ub_on_xb,
            self.target_free_energy
        )
        if return_result_dict:
            result_dict = {
                "xgen": xa,
                "xref": xb,
                "ugen_on_xgen": ua_on_xa,
                "uref_on_xref": ub_on_xb,
                "ugen_on_xref": ua_on_xb,
                "uref_on_xgen": ub_on_xa
            }
            return js, result_dict
        else:
            return js

    def update_free_energy(self, n_samples_flow, n_samples_target=None, **kwargs):
        """Update the target free energy through BAR."""
        self(n_samples_flow, n_samples_target, update_free_energy=True, **kwargs)


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

