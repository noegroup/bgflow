import torch

# TODO: write docstrings


class InversedDynamics(torch.nn.Module):
    def __init__(self, dynamics, t_max=1.0):
        super().__init__()
        self._dynamics = dynamics
        self._t_max = t_max

    def forward(self, t, state):
        *dxs, trace = self._dynamics(self._t_max - t, state)
        return [-dx for dx in dxs] + [-trace]

    #TODO correct sign? + rename!