
__all__ = [""]

from functools import partial

import numpy as np
import torch
from torch.nn import functional as F

MIN_BIN_WIDTH = 1e-6
MIN_SLOPE = 1e-6


def piecewise_transform(inputs, bin_edges, bin_transform, **bin_parameters):
    """Apply piecewise 1D transforms to some input.
    The function `bin_transform` is applied to each input. The `bin_parameters`
    are keyword arguments to the `bin_transform`.

    Parameters
    ----------
    inputs: (*batch_dims, )
    bin_edges: (*batch_dims, n_bins + 1)
    bin_transform: callable: (*batch_dims, **parameters) -> (*batch_dims)
        each parameter has shape (*batch_dims, n_bins)
    bin_parameters: torch.Tensor
        additional keyword arguments for the bin_transform.
        Each tensor has to have shape `(*batch_dims, n_bins)`.
    """
    bin_indices = bucketize_batch(inputs, bin_edges)
    assert (bin_indices >= 0).all()
    assert (bin_indices < bin_edges.shape[-1] - 1).all()
    input_parameters = {
        name: pick_along_last_dim(parameter, bin_indices)
        for name, parameter in bin_parameters.items()
    }
    output, logdet = bin_transform(
        inputs,
        **input_parameters
    )
    return output, logdet


def piecewise_linear(unnormalized_widths, energies, temperature=1.0, left=0.0, right=1.0, low=0.0, high=1.0):
    widths, cumwidths = normalize_bins(unnormalized_widths, low=left, high=right)
    heights, cumheights = energies2heights(energies, widths, temperature=temperature, low=low, high=high)

    _forward = partial(piecewise_transform,
                       bin_edges=cumwidths,
                       bin_transform=linear_piece,
                       slope=heights / widths,
                       x0=cumwidths[..., :-1],
                       y0=cumheights[..., :-1]
                       )
    _inverse = partial(piecewise_transform,
                       bin_edges=cumheights,
                       bin_transform=linear_piece,
                       slope=widths / heights,
                       x0=cumheights[..., :-1],
                       y0=cumwidths[..., :-1]
                       )
    return _forward, _inverse


def piecewise_rational_quadratic(
        unnormalized_widths, energies, unnormalized_slopes,
        temperature=1.0, left=0.0, right=1.0, low=0.0, high=1.0
):
    widths, cumwidths = normalize_bins(unnormalized_widths)
    heights, cumheights = energies2heights(energies, widths, temperature=temperature)
    mean_slope = (high-low)/(right-left)
    slopes = normalize_slopes(unnormalized_slopes, mean_slope=mean_slope, temperature=temperature)
    slopes_left, slopes_right = slopes[..., :-1], slopes[..., 1:]

    _forward = partial(piecewise_transform,
                       bin_edges=cumwidths,
                       bin_transform=rational_quadratic_piece,
                       width=widths,
                       cumwidth=cumwidths,
                       height=heights,
                       cumheight=cumheights,
                       slope_left=slopes_left,
                       slope_right=slopes_right
                       )
    _inverse = partial(piecewise_transform,
                       bin_edges=cumwidths,
                       bin_transform=inverse_rational_quadratic_piece,
                       width=widths,
                       cumwidth=cumwidths,
                       height=heights,
                       cumheight=cumheights,
                       slope_left=slopes_left,
                       slope_right=slopes_right
                       )
    return _forward, _inverse


# === Tools ===

def pick_along_last_dim(tensor, indices):
    """Select value .
    Parameters
    ----------
    tensor: (*batch_dims, n_bins) or (*broadcastable_dims, n_bins) or (n_bins, )
    indices: (*batch_dims, )
    """
    n_bins = tensor.shape[-1]
    tensor = tensor.expand(*indices.shape, n_bins)
    indices = indices[..., None]
    return tensor.gather(-1, indices)[..., 0]


def bucketize_batch(inputs: torch.Tensor, bin_edges: torch.Tensor, eps: float = 1e-6):
    edges = bin_edges.clone()
    edges[..., -1] += eps
    edges[..., 0] -= eps
    return (edges <= inputs[..., None]).sum(dim=-1) - 1


# === Linear piece ===

def linear_piece(inputs, slope, x0, y0):
    return slope * (inputs-x0) + y0, slope.abs().log()


# === Rational quadratic piece ===

def rational_quadratic_piece(
        inputs, width, cumwidth, height, cumheight, slope_left, slope_right
):
    delta = height / width
    theta = (inputs - cumwidth) / width
    theta_one_minus_theta = theta * (1 - theta)

    numerator = height * (
            delta * theta.pow(2) + slope_left * theta_one_minus_theta
    )

    denominator, logabsdet = _rqs_denom_dlogp(delta, slope_left, slope_right, theta)

    outputs = cumheight + numerator / denominator
    return outputs, logabsdet


def inverse_rational_quadratic_piece(
        inputs, width, cumwidth, height, cumheight, slope_left, slope_right
):
    delta = height / width
    a = (inputs - cumheight) * (
            slope_left + slope_right - 2 * delta
    ) + height * (delta - slope_left)
    b = height * slope_left - (inputs - cumheight) * (
            slope_left + slope_right - 2 * delta
    )
    c = - delta * (inputs - cumheight)

    discriminant = b.pow(2) - 4 * a * c
    assert (discriminant >= 0).all()

    root = (2 * c) / (-b - torch.sqrt(discriminant))
    # root = (- b + torch.sqrt(discriminant)) / (2 * a)
    outputs = root * width + cumwidth

    _, logabsdet = _rqs_denom_dlogp(delta, slope_left, slope_right, theta=root)

    return outputs, -logabsdet


def _rqs_denom_dlogp(delta, slope_left, slope_right, theta):
    """helper function to avoid duplicate code between forward and inverse"""
    theta_one_minus_theta = theta * (1 - theta)
    denominator = delta + (
            (slope_left + slope_right - 2 * delta)
            * theta_one_minus_theta
    )
    derivative_numerator = delta ** 2 * (
            slope_right * theta.pow(2)
            + 2 * delta * theta_one_minus_theta
            + slope_left * (1 - theta).pow(2)
    )
    logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)
    return denominator, logabsdet


# ===  Normalization ===

def normalize_bins(unconstrained, low=0.0, high=1.0, min_bin_width=MIN_BIN_WIDTH):
    """Turn an unconstrained tensor into bins that span an interval [low, high].

    Parameters
    ----------
    unconstrained : torch.Tensor
        tensor of shape (*batchdims, n_bins)
    low : float, optional
    high : float, optional
    min_bin_width : float, optional

    Returns
    -------
    bin_widths : torch.Tensor
        of shape (*batchdims, n_bins)
    bin_edges :
        of shape (*batchdims, n_bins + 1)
    """
    widths = F.softmax(unconstrained, dim=-1)
    return scale_and_cumsum(widths, low=low, high=high, min_bin_width=min_bin_width)


def energies2heights(energies, widths, temperature=1.0, low=0.0, high=1.0, min_bin_width=MIN_BIN_WIDTH):
    """Normalization method #2: Turn an unconstrained energy tensor into bin heights.
    The advantage is that we can apply a temperature parameter to the implied Boltzmann CDF.

    Parameters
    ----------
    energies : torch.Tensor
        tensor of shape (*batchdims, n_bins)
    widths : torch.Tensor
        NORMALIZED widths of shape (*batchdims, n_bins)
    temperature : float

    Returns
    -------
    bin_widths : torch.Tensor
        of shape (*batchdims, n_bins)
    bin_edges :
        of shape (*batchdims, n_bins + 1)
    """
    # F = \int_low^x f = \int_low^x e^(-u/T) / int_low^high e^(-u/T)
    #   = \sum_{i<k} e^(-u/T) * dx / \sum_{i} e^(-u/T) * dx
    heights = F.softmax(-energies/temperature + torch.log(widths), dim=-1)
    return scale_and_cumsum(heights, low=low, high=high, min_bin_width=min_bin_width)


def normalize_slopes(unconstrained, mean_slope=1.0, temperature=1.0, min_slope=MIN_SLOPE):
    """Render an unconstrained slope tensor positive.

    Notes
    -----
    This is defined such that if `unconstrained = zeros` the output is the mean slope.
    """
    slopes = F.softplus(unconstrained, beta=np.log(2) / (1 - min_slope))
    slopes = (min_slope + slopes) * mean_slope
    slopes = slopes ** (1 / temperature)
    assert slopes.min() >= min_slope
    return slopes


def scale_and_cumsum(widths, low=0.0, high=1.0, min_bin_width=MIN_BIN_WIDTH):
    """Scale bin widths that sum to 1 so that they fill the interval [low, high]"""
    num_bins = widths.shape[-1]
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (high - low) * cumwidths + low
    cumwidths[..., 0] = low
    cumwidths[..., -1] = high
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]
    return widths, cumwidths
