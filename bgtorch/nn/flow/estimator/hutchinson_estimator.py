import torch

# TODO: write docstrings

class HutchinsonEstimator(torch.nn.Module):
    def __init__(self, rademacher=True):
        super().__init__()
        self._rademacher = rademacher
        self._reset_noise = True

    def reset_noise(self, reset_noise=True):
        self._reset_noise = reset_noise

    def forward(self, dynamics, t, xs):
        with torch.set_grad_enabled(True):
            xs.requires_grad_(True)
            dxs = dynamics(t, xs)

            assert len(dxs.shape) == 2, "`dxs` must have shape [n_btach, system_dim]"
            system_dim = dxs.shape[-1]

            if self._reset_noise == True:
                self._reset_noise = False
                if self._rademacher == True:
                    self._noise = torch.randint(low=0, high=2, size=xs.shape).to(xs) * 2 - 1
                else:
                    self._noise = torch.randn_like(xs)

            noise_ddxs = torch.autograd.grad(dxs, xs, self._noise, create_graph=True)[0]
            divergence = torch.sum((noise_ddxs * self._noise).view(-1, system_dim), 1, keepdim=True)

        return dxs, -divergence 