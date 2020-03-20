from bgtorch.utils.autograd import brute_force_jacobian_trace
import torch

# TODO: write docstrings

class BruteForceEstimator(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, dynamics, t, xs):
        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            dxs = dynamics(t, xs)

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            divergence = brute_force_jacobian_trace(dxs, xs)

        return dxs, -divergence.view(-1, 1)