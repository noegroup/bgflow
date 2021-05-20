import torch
import numpy as np


__all__ = [
    "NonCompactAffineSigmoidComponents"
]


_logsigmoid = torch.nn.functional.logsigmoid


def _logit_deriv(x):
    return 1. / x + 1. / (1. - x)


def _log_logit_deriv(x):
    return torch.log(1. / x + 1. / (1. - x))


def non_compact_affine_sigmoid_transform(
    x,
    mu,
    log_sigma,
    min_density=torch.tensor(1e-5),
):        
    sigma = log_sigma.exp()
    z = x.logit() * sigma + mu
    y = z.sigmoid() * (1. - min_density) + min_density * x
    
    log_j = (1. - min_density).log() + _logsigmoid(z) + _logsigmoid(-z) + log_sigma + _log_logit_deriv(x)
    log_j_ = torch.where(torch.isnan(log_j), min_density.log(), log_j)
    log_pdf = torch.logsumexp(torch.stack([log_j_, min_density.expand_as(log_j).log()], dim=0), dim=0)
    
    return y, log_pdf


def rescaled_non_compact_affine_sigmoid_transform(
    x,
    mu,
    log_sigma,
    min_density=torch.tensor(1e-5),
    domain=torch.tensor([0., 1.]),
):
    
    a_in, b_in = domain.chunk(2, dim=-1)
    
    x = (x - a_in) / (b_in - a_in)
    x0 = (torch.zeros_like(x) - a_in) / (b_in - a_in)
    x1 = (torch.ones_like(x) - a_in) / (b_in - a_in)
    
    x_ = torch.stack([x, x0, x1], dim=0)
    (y, y0, y1), (log_pdf, _, _) = non_compact_affine_sigmoid_transform(x_, mu, log_sigma, min_density)
    
    y = (y - y0) / (y1 - y0)
    
    log_pdf = log_pdf - (y1 - y0).log() - (b_in - a_in).log()
    
    return y, log_pdf


class NonCompactAffineSigmoidComponents(torch.nn.Module):
    
    def __init__(
        self,
        compute_params: torch.nn.Module,
        min_density=torch.tensor(1e-4),
        log_sigma_bound=torch.tensor(4.),
        domain=torch.tensor([0., 1.]),
    ):
        super().__init__()
        self._param_net = compute_params
        self.register_buffer("_log_sigma_bound", log_sigma_bound)
        self.register_buffer("_min_density_lower_bound", min_density)
        self.register_buffer("_domain", domain)
        
        
    def _compute_params(self, cond, out):
        params = self._param_net(cond)
        mu, log_sigma, min_density = params.chunk(3, dim=-1)  
        
        mu = mu.view(*out.shape, -1)#.softmax(dim=-1).cumsum(dim=-1)
            
        log_sigma = log_sigma.view(*out.shape, -1)    
#         log_sigma = log_sigma.tanh() * self._log_sigma_bound
        
        # min density in [lb, 1]
        lower_bound = self._min_density_lower_bound.expand_as(min_density)
        lower_bound = lower_bound + min_density.sigmoid() * (1. - self._min_density_lower_bound)
     
        lower_bound = lower_bound.view(*out.shape, -1)
        
        return mu, log_sigma, lower_bound
    
    def forward(self, cond, out, *args, **kwargs):        
        mu, log_sigma, min_density = self._compute_params(cond=cond, out=out)
        out = out.unsqueeze(-1)        
        return rescaled_non_compact_affine_sigmoid_transform(
            out, mu, log_sigma, min_density, self._domain
        )