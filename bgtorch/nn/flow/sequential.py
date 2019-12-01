import torch

from .base import Flow
from ...utils.types import is_list_or_tuple


class SequentialFlow(Flow):
    def __init__(self, blocks):
        """
        Represents a diffeomorphism that can be computed
        as a discrete finite stack of layers.
        
        Returns the transformed variable and the log determinant
        of the Jacobian matrix.
            
        Parameters
        ----------
        blocks : Tuple / List of flow blocks
        """
        super().__init__()
        self._blocks = torch.nn.ModuleList(blocks)

    def forward(self, x, inverse=False, **kwargs):
        """
        Transforms the input along the diffeomorphism and returns
        the transformed variable together with the volume change.
            
        Parameters
        ----------
        x : PyTorch Floating Tensor.
            Input variable to be transformed. 
            Tensor of shape `[..., n_dimensions]`.
        inverse: boolean.
            Indicates whether forward or inverse transformation shall be performed.
            If `True` computes the inverse transformation.
        
        Returns
        -------
        z: PyTorch Floating Tensor.
            Transformed variable. 
            Tensor of shape `[..., n_dimensions]`.
        dlogp : PyTorch Floating Tensor.
            Total volume change as a result of the transformation.
            Corresponds to the log determinant of the Jacobian matrix.
        """
        dlogp = torch.zeros(*x.shape[:-1], 1).to(x)
        blocks = self._blocks
        if inverse:
            blocks = reversed(blocks)
        if not is_list_or_tuple(x):
            x = [x]
        for block in blocks:
            *x, ddlogp = block(*x, inverse=inverse, **kwargs)
            dlogp += ddlogp
        return (*x, dlogp)
        