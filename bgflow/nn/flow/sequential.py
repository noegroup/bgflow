import logging
import numpy as np
import torch

from .base import Flow

logger = logging.getLogger('bgflow')


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
        dlogp = 0.0
        blocks = self._blocks
        if inverse:
            blocks = reversed(blocks)
        for i, block in enumerate(blocks):
            logger.debug(f"Input shapes {[x.shape for x in xs]}")
            *xs, ddlogp = block(*xs, inverse=inverse, **kwargs)
            logger.debug(f"Flow block {i} (inverse={inverse}):  {block}")
            logger.debug(f"Output shapes {[x.shape for x in xs]}")
            dlogp += ddlogp
        return (*xs, dlogp)

    def _forward(self, *args, **kwargs):
        return self.forward(*args, **kwargs, inverse=False)

    def _inverse(self, *args, **kwargs):
        return self.forward(*args, **kwargs, inverse=True)

    def trigger(self, function_name):
        """
        Evaluate functions for all blocks that have a function with that name and return a tensor of the stacked results.
        """
        results = [
            getattr(block, function_name)()
            for block in self._blocks
            if hasattr(block, function_name) and callable(getattr(block, function_name))
        ]
        if len(results) > 0 and all(res is not None for res in results):
            return torch.stack(results)
        else:
            return torch.zeros(0)

    def __iter__(self):
        return iter(self._blocks)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self._blocks[index]
        else:
            indices = np.arange(len(self))[index]
            return SequentialFlow([self._blocks[i] for i in indices])

    def __len__(self):
        return len(self._blocks)
