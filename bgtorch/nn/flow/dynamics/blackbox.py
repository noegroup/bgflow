import torch

# TODO: write docstrings


class BlackBoxDynamics(torch.nn.Module):
    
    def __init__(self, dynamics_function, divergence_estimator, compute_divergence=True):
        super().__init__()
        self._dynamics_function = dynamics_function
        self._divergence_estimator = divergence_estimator
        self._compute_divergence = compute_divergence

    def forward(self, t, *xs):
        if self._compute_divergence:
            *dxs, divergence = self._divergence_estimator(
                self._dynamics_function, t, *xs
            )
        else:
            dxs = self._dynamics_function(t, xs)
            divergence = None
        return (*dxs, divergence)