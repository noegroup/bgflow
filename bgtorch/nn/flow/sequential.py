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

    def forward(self, *xs, inverse=False, **kwargs):
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
        n_batch = xs[0].shape[0]
        dlogp = torch.zeros(n_batch, 1).to(xs[0])
        blocks = self._blocks
        if inverse:
            blocks = reversed(blocks)
        for block in blocks:
            *xs, ddlogp = block(*xs, inverse=inverse, **kwargs)
            dlogp += ddlogp
        return (*xs, dlogp)

    def trigger(self, function_name):
        """Evaluate functions for all blocks that have a function with that name and return a list of the results."""
        return [
            getattr(block, function_name)()
            for block in self._blocks
            if hasattr(block, function_name) and callable(getattr(block, function_name))
        ]
