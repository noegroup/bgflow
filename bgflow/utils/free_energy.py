
import io
import warnings
import torch
import numpy as np
from .types import as_numpy
from contextlib import redirect_stdout


__all__ = ["bennett_acceptance_ratio"]


def bennett_acceptance_ratio(
        forward_work: torch.Tensor,
        reverse_work: torch.Tensor,
        compute_uncertainty: bool = True,
        implementation: str = "torch",
        maximum_iterations: int = 500,
        relative_tolerance: float = 1e-12,
        warn: bool = False
):
    """Compute the free energy difference DF_{0 -> 1} between two thermodynamic ensembles
    with energies u0 and u1 using the Bennett acceptance ratio method.

    Parameters
    ----------
    forward_work : torch.Tensor
        The dimensionless energy difference u1(x_k) - u0(x_k) computed on samples x_k ~ e^-u0.
    reverse_work : torch.Tensor
        The dimensionless energy difference u0(x_k) - u1(x_k) computed on samples x_k ~ e^-u1.
    compute_uncertainty : bool
        Whether uncertainties on the free energy estimate are computed or not.
    implementation : str
        Either "torch" (the native python implementation) or "pymbar" (the original reference implementation)
    maximum_iterations : int
        Stopping criteria: maximum number of iterations.
    relative_tolerance : float
        Stopping criteria: relative difference between subsequent iterations.
    warn : bool
        Whether to raise a UserWarning when overlap is poor and free energies cannot be estimated.

    Returns
    -------
    delta_f : torch.Tensor
        The estimate for the free energy difference (shape [])
    uncertainty : torch.Tensor
        An estimate for the uncertainty of delta_f (shape []). None if no uncertainties are computed.

    Notes
    -----
    The free energy difference is the negative log ratio of the partitions functions.
    """
    implementations = {
        "torch": _bennett_acceptance_ratio_torch,
        "pymbar": _bennett_acceptance_ratio_pymbar
    }
    delta_f, uncertainty = implementations[implementation](
        forward_work,
        reverse_work,
        compute_uncertainty=compute_uncertainty,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance
    )
    if torch.isnan(delta_f) and warn:
        warnings.warn("BAR could not compute free energy differences due to poor overlap. Returns nan.", UserWarning)
    return delta_f, uncertainty


def _bennett_acceptance_ratio_pymbar(forward_work, reverse_work, compute_uncertainty=True, maximum_iterations=500, relative_tolerance=1e-12):
    """pymbar reference implementation"""
    import pymbar
    ctx = {"device": forward_work.device, "dtype": forward_work.dtype}
    f = io.StringIO()
    with redirect_stdout(f):
        result = pymbar.BAR(
            w_F=as_numpy(forward_work),
            w_R=as_numpy(reverse_work),
            return_dict=False,
            compute_uncertainty=compute_uncertainty,
            maximum_iterations=maximum_iterations,
            relative_tolerance=relative_tolerance
        )

    if "poor overlap" in f.getvalue() or (compute_uncertainty and np.isnan(result[1])):
        return torch.tensor(np.nan, **ctx), torch.tensor(np.nan, **ctx)
    if compute_uncertainty:
        return torch.tensor(result[0], **ctx), torch.tensor(result[1], **ctx)
    else:
        return torch.tensor(result, **ctx), None


def _bennett_acceptance_ratio_torch(forward_work, reverse_work, compute_uncertainty=True, maximum_iterations=500, relative_tolerance=1e-12):
    """native implementation in pytorch"""
    estimate, uncertainty = _bar(
        forward_work,
        reverse_work,
        compute_uncertainty=compute_uncertainty,
        maximum_iterations=maximum_iterations,
        relative_tolerance=relative_tolerance
    )
    return estimate, uncertainty


def _bar_zero(forward_work, reverse_work, delta_f):
    """The function that BAR needs to set to zero. Adapted from pymbar."""
    # compute log ratio of forward and reverse counts
    n_forward = forward_work.shape[0]
    n_reverse = reverse_work.shape[0]
    log_count = np.log(n_forward / n_reverse)

    exp_arg_forward = log_count + forward_work - delta_f
    max_arg_forward = torch.clamp(exp_arg_forward, min=0.0, max=1e10)
    log_f_forward = -max_arg_forward - torch.log(
        torch.exp(-max_arg_forward) + torch.exp(exp_arg_forward - max_arg_forward))
    log_numerator = torch.logsumexp(log_f_forward, dim=0)

    exp_arg_reverse = -(log_count - reverse_work - delta_f)
    max_arg_reverse = torch.clamp(exp_arg_reverse, min=0.0, max=1e10)
    log_f_reverse = -max_arg_reverse - torch.log(
        torch.exp(-max_arg_reverse) + torch.exp(exp_arg_reverse - max_arg_reverse))
    log_denominator = torch.logsumexp(log_f_reverse, dim=0)
    return log_numerator - log_denominator


def _one_sided_reweighting(work):
    n_work = work.shape[0]
    delta_f = -(torch.logsumexp(-work, dim=0) - np.log(n_work))
    return delta_f


def _bar(forward_work, reverse_work, compute_uncertainty=True, maximum_iterations=500, relative_tolerance=1e-8):
    """Bennett Acceptance Ratio; adapted from pymbar"""

    forward_work = forward_work.flatten()
    reverse_work = reverse_work.flatten()

    upper_bound = _one_sided_reweighting(forward_work)
    lower_bound = -_one_sided_reweighting(reverse_work)
    f_upper_bound = _bar_zero(forward_work, reverse_work, upper_bound)
    f_lower_bound = _bar_zero(forward_work, reverse_work, lower_bound)

    while f_upper_bound * f_lower_bound > 0:
        f_average = (upper_bound + lower_bound) / 2
        upper_bound = upper_bound - torch.clamp((upper_bound - f_average).abs(), min=0.1, max=1e10)
        lower_bound = lower_bound + torch.clamp((lower_bound - f_average).abs(), min=0.1, max=1e10)
        f_upper_bound = _bar_zero(forward_work, reverse_work, upper_bound)
        f_lower_bound = _bar_zero(forward_work, reverse_work, lower_bound)

    delta_f_old = np.infty
    for iterations in range(maximum_iterations):
        delta_f = upper_bound - f_upper_bound * (upper_bound - lower_bound) / (f_upper_bound - f_lower_bound)
        f_new = _bar_zero(forward_work, reverse_work, delta_f)
        if f_upper_bound * f_new < 0.0:
            lower_bound = delta_f
            f_lower_bound = f_new
        elif f_lower_bound * f_new <= 0:
            upper_bound = delta_f
            f_upper_bound = f_new
        else:
            return torch.tensor(np.nan).to(forward_work), torch.tensor(np.nan).to(forward_work)
            #raise Exception("Cannot determine Free energy")

        relative_change = (delta_f - delta_f_old).abs() / delta_f
        if relative_change < relative_tolerance:
            break

    if not compute_uncertainty:
        return delta_f, None

    # ==== COMPUTE UNCERTAINTY =====
    # Determine number of forward and reverse work values provided.
    n_forward = forward_work.shape[0]  # number of forward work values
    n_reverse = reverse_work.shape[0]  # number of reverse work values
    # Compute log ratio of forward and reverse counts.
    M = np.log(n_forward / n_reverse)
    C = M - delta_f
    exp_arg_f = (forward_work + C)
    max_arg_f = exp_arg_f.max()
    log_f_f = - torch.log(torch.exp(-max_arg_f) + torch.exp(exp_arg_f - max_arg_f))
    assert len(log_f_f.shape) == 1
    af_f = torch.exp(torch.logsumexp(log_f_f, dim=-1) - max_arg_f) / n_forward

    # fR = 1 / (1 + np.exp(w_R - C)), but we need to handle overflows
    exp_arg_r = reverse_work - C
    max_arg_r = exp_arg_r.max()
    log_fR = - torch.log(torch.exp(-max_arg_r) + torch.exp(exp_arg_r - max_arg_r))
    afR = torch.exp(torch.logsumexp(log_fR, dim=-1) - max_arg_r) / n_reverse

    afF2 = torch.exp(torch.logsumexp(2 * log_f_f, dim=-1) - 2 * max_arg_f) / n_forward
    afR2 = torch.exp(torch.logsumexp(2 * log_fR, dim=-1) - 2 * max_arg_r) / n_reverse

    nrat = (n_forward + n_reverse) / (n_forward * n_reverse)  # same for both methods

    variance = (afF2 / af_f ** 2) / n_forward + (afR2 / afR ** 2) / n_reverse - nrat
    d_delta_f = variance.sqrt()

    return delta_f, d_delta_f
