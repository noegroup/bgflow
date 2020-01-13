import torch

# TODO: write docstrings


class SimpleDynamics(torch.nn.Module):
    
    def __init__(self, dynamics_function, divergence_estimator):
        super().__init__()
        self._dynamics_function = dynamics_function
        self._divergence_estimator = divergence_estimator
        
    def forward(self, *xs, compute_divergence=True):
        if compute_divergence:
            *dxs, divergence = self._divergence_estimator(
                self._dynamics_function, *xs
            )
        else:
            dxs = self._dynamics_function(xs)
            divergence = None
        return (*dxs, divergence)