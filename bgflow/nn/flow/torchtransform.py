
import torch
from .base import Flow


__all__ = ["TorchTransform"]


class TorchTransform(Flow):
    """Wrap a torch.distributions.Transform as a Flow instance

    Parameters
    ----------
    transform : torch.distributions.Transform
        The transform instance that should be wrapped as a Flow instance.
    reinterpreted_batch_ndims : int, optional
        Number of batch dimensions to be reinterpreted as event dimensions.
        If >0, this transform is wrapped in an torch.distributions.IndependentTransform instance.
    """

    def __init__(self, transform, reinterpreted_batch_ndims=0):
        super().__init__()
        if reinterpreted_batch_ndims > 0:
            transform = torch.distributions.IndependentTransform(transform, reinterpreted_batch_ndims)
        self._delegate_transform = transform

    def _forward(self, x, **kwargs):
        y = self._delegate_transform(x)
        dlogp = self._delegate_transform.log_abs_det_jacobian(x, y)
        return y, dlogp[..., None]

    def _inverse(self, y, **kwargs):
        x = self._delegate_transform.inv(y)
        dlogp = - self._delegate_transform.log_abs_det_jacobian(x, y)
        return x, dlogp[..., None]
