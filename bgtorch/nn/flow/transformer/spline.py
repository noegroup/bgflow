import torch

from .base import Transformer

__all__ = [
    "ConditionalSplineTransformer",
]


class ConditionalSplineTransformer(Transformer):
    def __init__(
        self,
        params_net: torch.nn.Module,
        is_circular: bool = False,
        left: float = 0.0,
        right: float = 1.0,
        bottom: float = 0.0,
        top: float = 1.0,
    ):
        """
        Spline transformer for variables defined in [0, 1].

        Uses bayesiains/nflows under the hood.

        Parameters:
        -----------
        params_net: torch.nn.Module
            Computes the transformation parameters for `y` conditioned
            on `x`. Input dimension must be `x.shape[-1]`. Output
            dimension must be `y.shape[-1] * 3` if splines are circular
            and `y.shape[-1] * 3 + 1` if splines are defined on [0, 1]
        """
        super().__init__()
        self._params_net = params_net
        self._is_circular = is_circular
        self._left = left
        self._right = right
        self._bottom = bottom
        self._top = top

    def _compute_params(self, x, y_dim):
        params = self._params_net(x)
        params = params.view(x.shape[0], y_dim, -1)
        if self._is_circular:
            widths, heights, slopes = torch.chunk(params, 3, dim=-1)
            slopes = torch.cat([slopes, slopes[..., [0]]], dim=-1)
        else:
            widths, heights = torch.chunk(params[..., : 2 * y_dim], 2, dim=-1)
            slopes = params[..., 2 * y_dim :]
        return widths, heights, slopes

    def _forward(self, x, y, *args, **kwargs):
        from nflows.transforms.splines import rational_quadratic_spline

        widths, heights, slopes = self._compute_params(x, y.shape[-1])
        z, dlogp = rational_quadratic_spline(
            y,
            widths,
            heights,
            slopes,
            inverse=True,
            left=self._left,
            right=self._right,
            top=self._top,
            bottom=self._bottom,
        )
        return z, dlogp.sum(dim=-1, keepdim=True)

    def _inverse(self, x, y, *args, **kwargs):
        from nflows.transforms.splines import rational_quadratic_spline

        widths, heights, slopes = self._compute_params(x, y.shape[-1])
        z, dlogp = rational_quadratic_spline(
            y,
            widths,
            heights,
            slopes,
            inverse=False,
            left=self._left,
            right=self._right,
            top=self._top,
            bottom=self._bottom,
        )
        return z, dlogp.sum(dim=-1, keepdim=True)
