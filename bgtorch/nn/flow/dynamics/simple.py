import torch

# TODO: write docstrings

class TimeIndependentDynamics(torch.nn.Module):
    def __init__(self, dynamics):
        super().__init__()
        self._dynamics = dynamics

    def forward(self,  t, xs):
        dxs = self._dynamics(xs)
        return dxs