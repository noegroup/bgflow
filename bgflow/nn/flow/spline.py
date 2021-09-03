import torch
from bgflow.nn.flow.base import Flow
import torch.nn.functional as F


class PeriodicTabulatedTransform(Flow):
    def __init__(
            self,
            support_points: torch.Tensor,
            support_values: torch.Tensor,
            slopes: torch.Tensor,
    ):
        """
        Parameters
        ----------
        support_points: torch.Tensor
            ascending support points on x axis; shape (n_degrees_of_freedom, n_bin_edges)
        support_values: torch.Tensor
            ascending support points on y axis; shape (n_degrees_of_freedom, n_bin_edges)
        slopes: torch.Tensor
            slopes at support points, shape (n_degrees_of_freedom, n_bin_edges)
        """
        super().__init__()
        self.register_buffer("_support_points", support_points)
        self.register_buffer("_support_values", support_values)
        self.register_buffer("_slopes", torch.clamp(slopes, 1e-6, 1e6))
        widths = support_points[..., 1:] - support_points[..., :-1]
        assert torch.all(widths >= 0.0), ValueError("support points must be ascending in last dimension")
        heights = support_values[..., 1:] - support_values[..., :-1]
        assert torch.all(heights >= 0.0), ValueError("support values must be ascending in last dimension")

    def _forward(self, x: torch.Tensor):
        # shift into primary interval
        left = torch.min(self._support_points, dim=-1)[0]
        right = torch.max(self._support_points, dim=-1)[0]
        # x = torch.remainder(x - left, right - left) + left
        assert (x >= left).all()
        assert (x <= right).all()
        # evaluate spline
        y, dlogp = rq_spline(x, self._support_points, self._support_values, self._slopes)
        return y.clamp(self._support_values.min(), self._support_values.max()), dlogp.sum(dim=-1, keepdim=True)

    def _inverse(self, x: torch.Tensor, *args, **kwargs):
        # shift into primary interval
        bottom = torch.min(self._support_values, dim=-1)[0]
        top = torch.max(self._support_values, dim=-1)[0]
        # x = torch.remainder(x - bottom, top - bottom) + bottom
        assert (x >= bottom).all()
        assert (x <= top).all()
        # evaluate spline
        y, dlogp = rq_spline(x, self._support_points, self._support_values, self._slopes, inverse=True)
        return y.clamp(self._support_points.min(), self._support_points.max()), dlogp.sum(dim=-1, keepdim=True)


DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


def rq_spline(
        inputs,
        supportx,
        supporty,
        derivatives,
        inverse=False,
        min_bin_width=1e-4,#DEFAULT_MIN_BIN_WIDTH,
        min_bin_height=1e-4,#=1e-8,#DEFAULT_MIN_BIN_HEIGHT,
        min_derivative=1e-4,#DEFAULT_MIN_DERIVATIVE,
):
    """Rational Quadratic Spline
    Parameters
    ----------
    inputs : torch.Tensor
        input tensor; shape (..., n_distributions), where ... represents batch dimensions
        The input for each distribution has to be within the range of its support points.
    supportx : torch.Tensor
        support points on the x axis; shape (n_distributions, n_points_per_distribution)
        The points for each distribution have to be monotonically increasing.
    supporty : torch.Tensor
        support values on the y axis; shape (n_distributions, n_points_per_distribution)
        The points for each distribution have to be monotonically increasing.
    derivatives : torch.Tensor
        derivatives at the support points (have to be strictly positive)
    Returns
    -------
    outputs : torch.Tensor
        The transformed input. Same shape as input.
    logdet : torch.Tensor
        Elementwise (!) logarithmic determinant of the jacobian, log |det J(x)|. Same shape as input.
    References
    ----------
    [1] C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, Neural Spline Flows, (2019). http://arxiv.org/abs/1906.04032
    """
    assert torch.all(derivatives > 0)
    assert torch.all(supportx[..., :-1] <= supportx[..., 1:])
    assert torch.all(supporty[..., :-1] <= supporty[..., 1:])
    if inverse:
        assert torch.all(inputs >= supporty.min(dim=-1)[0])
        assert torch.all(inputs <= supporty.max(dim=-1)[0])
    else:
        assert torch.all(inputs >= supportx.min(dim=-1)[0])
        assert torch.all(inputs <= supportx.max(dim=-1)[0])

    num_bins = supportx.shape[-1] - 1
    widths = supportx[..., 1:] - supportx[..., :-1]
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    supportx = (
        supportx.min(dim=-1, keepdim=True)[0]
        + F.pad(torch.cumsum(widths, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    )

    heights = supporty[..., 1:] - supporty[..., :-1]
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    supporty = (
        supporty.min(dim=-1, keepdim=True)[0]
        + F.pad(torch.cumsum(heights, dim=-1), pad=(1, 0), mode="constant", value=0.0)
    )

    derivatives = min_derivative + derivatives

    if inverse:
        bin_idx = searchsorted(supporty, inputs)
    else:
        bin_idx = searchsorted(supportx, inputs)
    bin_idx[bin_idx == num_bins] = 0  #

    input_supportx = select_item(supportx, bin_idx)
    input_bin_widths = select_item(widths, bin_idx)

    input_supporty = select_item(supporty, bin_idx)
    delta = heights / widths
    input_delta = select_item(delta, bin_idx)

    input_derivatives = select_item(derivatives, bin_idx)
    input_derivatives_plus_one = select_item(derivatives, bin_idx + 1)

    input_heights = select_item(heights, bin_idx)

    if inverse:
        a = (((inputs - input_supporty) * (input_derivatives
                                           + input_derivatives_plus_one
                                           - 2 * input_delta)
              + input_heights * (input_delta - input_derivatives)))
        b = (input_heights * input_derivatives
             - (inputs - input_supporty) * (input_derivatives
                                            + input_derivatives_plus_one
                                            - 2 * input_delta))
        c = - input_delta * (inputs - input_supporty)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_supportx

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * root.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - root).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_supportx) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2)
                                     + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + ((input_derivatives + input_derivatives_plus_one - 2 * input_delta)
                                     * theta_one_minus_theta)
        outputs = input_supporty + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (input_derivatives_plus_one * theta.pow(2)
                                                     + 2 * input_delta * theta_one_minus_theta
                                                     + input_derivatives * (1 - theta).pow(2))
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(
        inputs[..., None] >= bin_locations,
        dim=-1
    ) - 1


def select_item(items, index):
    return items[..., index].diagonal(dim1=0, dim2=-1)