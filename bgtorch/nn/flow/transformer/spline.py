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
        Spline transformer transforming variables in [left, right) into variables in [bottom, top).

        Uses `n_bins` spline nodes to define a inverse CDF transform on the interval.

        Uses bayesiains/nflows under the hood.

        Parameters
        ----------
        params_net: torch.nn.Module
            Computes the transformation parameters for `y` conditioned
            on `x`. Input dimension must be `x.shape[-1]`. Output
            dimension must be `y.shape[-1] * n_bins * 3` if splines are circular
            and `y.shape[-1] * (n_bins * 3 + 1)` if not.
            The integer `n_bins` is inferred implicitly from the network output shape.
        is_circular : bool
            If True, the boundaries are treated as periodic boundaries, i.e. the pdf is enforced to be continuous.

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
        self._is_circular = is_circular
        self._left = left
        self._right = right
        self._bottom = bottom
        self._top = top

    def _compute_params(self, x, y_dim):
        params = self._params_net(x)
        params = params.view(x.shape[0], y_dim, -1)
        n_bins = params.shape[-1] // 3
        if self._is_circular:
            widths, heights, slopes = torch.split(params, [n_bins, n_bins, n_bins], dim=-1)
            slopes = torch.cat([slopes, slopes[..., [0]]], dim=-1)
        else:
            widths, heights, slopes = torch.split(params, [n_bins, n_bins, n_bins+1], dim=-1)
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
