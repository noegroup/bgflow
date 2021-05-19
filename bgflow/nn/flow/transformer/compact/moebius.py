import numpy as np
import torch

__all__ = ["MoebiusComponents"]


def moebius_transform(z, w):
    """compute moebius transform of z along w 
       returns value together with corresponding jacobian"""
    wnorm = 1 - w.pow(2).sum(dim=-1, keepdim=True)
    diffnorm = (z - w).norm(dim=-1, keepdim=True)
    trafo = wnorm / diffnorm.pow(2) * (z - w) - w
    outer = torch.einsum("...i, ...j -> ...ij", z - w, z - w)
    jac = (-2 * wnorm / diffnorm.pow(4)).unsqueeze(-1) * outer 
    jac += ( wnorm / diffnorm.pow(2) ).unsqueeze(-1) * torch.eye(2).expand_as(jac).to(z)
    return trafo, jac


def uniform_to_complex(x):
    """map number in [0,1) to complex number / S^1
       returns value together with corresponding jacobian"""
    assert torch.all((x >= 0.) & (x <= 1.))
    x = (2 * x - 1) * np.pi
    jac = 2 * np.pi * torch.stack([-x.sin(), x.cos()], dim=-1)
    return torch.stack([x.cos(), x.sin()], dim=-1), jac


def complex_to_uniform(z):
    """map complex number / S^1 to number in [0, 1)
       returns value together with corresponding jacobian"""
    assert torch.allclose(z.norm(dim=-1), torch.ones_like(z.norm(dim=-1)), rtol=1e-4, atol=1e-4)    
    jac = torch.stack([
          -z[..., 1],
          z[..., 0]
    ], dim=-1)
    jac = 1 / (z[..., 0].pow(2) + z[..., 1].pow(2)).unsqueeze(-1) * jac
    jac = 1 / (2 * np.pi) * jac
    
    x = torch.atan2(z[..., 1], z[..., 0]) % (2 * np.pi)
    x = x / (2 * np.pi ) # - 0.5
    return x, jac


def complex_multiply(a, b):
    """multiplies complex number a with complex number b (rotation)
       returns value together with corresponding jacobian"""
    prod = torch.stack([
        a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1],
        a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0],
    ], dim=-1)
    jac = torch.stack([
        torch.stack([
            b[..., 0], b[..., 1]
        ], dim=-1),
        torch.stack([
            -b[..., 1], b[..., 0]
        ], dim=-1),
    ], dim=-1)    
    return prod, jac.transpose(-1, -2)


def complex_conjugate(a):
    """computes complex conjugate (complex inverse on circle)"""
    return torch.stack([
        a[..., 0],
        -a[..., 1]
    ], dim=-1)


def full_moebius(x, w):
    """applies moebius trafo along w onto x and computes corresponding log det jac"""
    z, jac_in = uniform_to_complex(x)
    z_, jac_moeb = moebius_transform(z, w)
    z0, _ = uniform_to_complex(torch.zeros_like(x))
    z0_, _  = moebius_transform(z0, w)
    z0_ = complex_conjugate(z0_)
    z_, jac_rot = complex_multiply(z_, z0_)
    x_, jac_out = complex_to_uniform(z_)
    x_ = torch.where(x == 1, x, x_)
    
    # TODO: inline computation of jacobian?
    return x_, torch.einsum("...i, ...ij, ...jk, ...k", jac_in, jac_moeb, jac_rot, jac_out).log()



class MoebiusComponents(torch.nn.Module):
    
    def __init__(
        self,
        compute_params: torch.nn.Module,
        w_bound=torch.tensor(0.99),
    ):
        """
            compute_params: predicts directional and radial components of w, output should be [*shape, 3] dimensional
            w_bound: bound avoiding singular projections
        """
        super().__init__()
        self._param_net = compute_params
        self.register_buffer("_w_bound", w_bound)
        
    def _compute_params(self, cond, out):
        w = self._param_net(cond).view(*out.shape, -1, 2)
        
        w = w * self._w_bound / ( 1 + w.norm(dim=-1, keepdim=True))
        
        return w
    
    def forward(self, cond, out, *args, **kwargs):  
        w = self._compute_params(cond=cond, out=out)
        out = out.unsqueeze(-1).expand_as(w[..., 0])
        return full_moebius(out, w)