import torch
from typing import NamedTuple

from .base import Transformer

__all__ = [
    "ConditionalSplineTransformer",
    "DomainExtension",
]


class DomainExtension(NamedTuple):
    tails: str = "linear"
    tail_bound: float = 1.0


class ConditionalSplineTransformer(Transformer):
    def __init__(
        self,
        params_net: torch.nn.Module,
        is_circular: bool = False,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
        domain_extension: DomainExtension = None,
    ):
        """
        Spline transformer transforming variables in [left, right) into variables in [bottom, top).

        Uses `n_bins` spline nodes to define a inverse CDF transform on the interval.

        Uses bayesiains/nflows under the hood.

        Parameters
        ----------
        params_net: torch.nn.Module
            Computes the transformation parameters for `y` conditioned
            on `x`. Input dimension must be `x.shape[-1]`. Output
            dimension must be
                - `y.shape[-1] * n_bins * 3` if splines are circular,
                - `y.shape[-1] * (n_bins * 3 + 1)` if not
                - `y.shape[-1] * n_bins * 3 + n_circular if some transformed variables are circular
            The integer `n_bins` is inferred implicitly from the network output shape.
        is_circular : bool or torch.Tensor
            If True, the boundaries are treated as periodic boundaries, i.e. the pdf is enforced to be continuous.
            If is_circular is a boolean tensor, only the indices at which the tensor is True are treated as periodic
            (tensor shape has to be (y.shape[-1], ) ).

        Raises
        ------
        RuntimeError
            If the `params_net` output does not have the correct shape.

        Notes
        -----
        This transform requires the nflows package to be installed.
        It is available from PyPI and can be installed with `pip install nflows`.

        References
        ----------
        C. Durkan, A. Bekasov, I. Murray, G. Papamakarios, Neural Spline Flows, NeurIPS 2019,
        https://arxiv.org/abs/1906.04032.
        """
        super().__init__()
        self._params_net = params_net
        self._is_circular = torch.as_tensor(is_circular, dtype=torch.bool)
        self._left = left
        self._right = right
        self._bottom = bottom
        self._top = top
        self._domain_extension = domain_extension

    def _compute_params(self, x, y_dim):
        """Compute widths, heights, and slopes from x through the params_net.

        Parameters
        ----------
        x : torch.Tensor
            Conditioner input.
        y_dim : int
            Number of transformed degrees of freedom.

        Returns
        -------
        widths : torch.Tensor
            unnormalized bin widths for the monotonic spline interpolation
            shape ( ... , y_dim, n_bins), where ... represents batch dims
        heights : torch.Tensor
            unnormalized bin heights for the monotonic spline interpolation
            shape ( ... , y_dim, n_bins)
        slopes : torch.Tensor
            unnormalized slopes for the monotonic spline interpolation
            shape (... , y_dim, n_bins + 1)
        """
        params = self._params_net(x)
        # assume that all but the last dim of the params tensor are batch dims
        batch_shape = params.shape[:-1]
        n_bins = params.shape[-1] // (y_dim * 3)
        widths, heights, slopes, noncircular_slopes = torch.split(
            params,
            [
                n_bins * y_dim,
                n_bins * y_dim,
                n_bins * y_dim,
                self._n_noncircular(y_dim),
            ],
            dim=-1,
        )
        widths = widths.reshape(*batch_shape, y_dim, n_bins)
        heights = heights.reshape(*batch_shape, y_dim, n_bins)
        slopes = slopes.reshape(*batch_shape, y_dim, n_bins)
        noncircular_slopes = noncircular_slopes.reshape(*batch_shape, -1)
        # make periodic
        slopes = torch.cat([slopes, slopes[..., [0]]], dim=-1)
        # make noncircular indices non-periodic
        slopes[..., self._noncircular_indices(y_dim), -1] = noncircular_slopes
        return widths, heights, slopes

    def forward(self, x, y, *args, inverse=False, **ignored_kwargs):
        from nflows.transforms.splines import (
            rational_quadratic_spline,
            unconstrained_rational_quadratic_spline,
        )

        widths, heights, slopes = self._compute_params(x, y.shape[-1])

        kwargs = {
            "unnormalized_widths": widths,
            "unnormalized_heights": heights,
            "unnormalized_derivatives": slopes,
            "inverse": inverse,
        }

        if self._domain_extension is None:
            kwargs = {
                **kwargs,
                "left": self._left,
                "right": self._right,
                "top": self._top,
                "bottom": self._bottom,
            }
            z, dlogp = rational_quadratic_spline(y, **kwargs)
        else:
            kwargs = {
                **kwargs,
                "tails": self._domain_extension.tails,
                "tail_bound": self._domain_extension.tail_bound,
            }
            z, dlogp = unconstrained_rational_quadratic_spline(y, **kwargs)

        return z, dlogp.sum(dim=-1, keepdim=True)

    def _n_noncircular(self, y_dim):
        if self._is_circular.all():
            return 0
        elif not self._is_circular.any():
            return y_dim
        else:
            return self._is_circular.sum()

    def _noncircular_indices(self, y_dim):
        if self._is_circular.all():
            return slice(0)
        elif not self._is_circular.any():
            return slice(None)
        else:
            return torch.logical_not(self._is_circular)
