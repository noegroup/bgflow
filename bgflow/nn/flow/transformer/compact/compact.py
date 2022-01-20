from functools import partial
from typing import Callable

import torch
import numpy as np

from bgflow.nn.flow.transformer.base import Transformer


__all__ =[
    "SmoothRamp",
    "PowerRamp",
    "AffineSigmoidComponents",
    "AffineSigmoidComponentInitGrid",
    "MixtureCDFTransformer",
    "ConstrainedBoundaryCDFTransformer",
    "SmoothRampWithTrainableExponent"
]

# TODO: docstrings and tests!


def polynomial_ramp(x, power=3):
    rx = torch.relu(x)
    ramp = rx.pow(power)
    grad = power * rx.pow(power - 1)
    return ramp, grad


_SMOOTH_RAMP_INFLECTION_ALPHA_2 = 0.3052033


def smooth_ramp_pow2(x, alpha, unimodal=True, eps=1e-8):
    if unimodal:
        alpha = alpha + _SMOOTH_RAMP_INFLECTION_ALPHA_2
    
    nonzero = x > 0
    
    xinv = x.clamp_min(eps).reciprocal()
#     xinv = x.reciprocal()
    arg = xinv.pow(2).neg()
    ramp = ((1 + arg) * alpha).exp()
#     ramp = torch.where(
#         nonzero,
#         ramp,
#         torch.zeros_like(ramp)
#     )
    
    grad_arg = xinv.pow(3) * 2
    grad = ramp * grad_arg * alpha
#     grad = torch.where(
#         nonzero,
#         grad,
#         torch.zeros_like(ramp)
#     )
    
    return ramp, grad


_SMOOTH_RAMP_INFLECTION_ALPHA_1 = 0.8656721


def smooth_ramp_pow1(x, alpha, unimodal=True, eps=1e-8):
    if unimodal:
        alpha = alpha + _SMOOTH_RAMP_INFLECTION_ALPHA_1
        
    nonzero = x > 0
    
    xinv = x.clamp_min(eps).reciprocal()
#     xinv = x.reciprocal()
    arg = xinv.neg()
    ramp = ((1 + arg) * alpha).exp()
#     ramp = torch.where(
#         nonzero,
#         ramp,
#         torch.zeros_like(ramp)
#     )
    
    grad_arg = xinv.pow(2)
    grad = ramp * grad_arg * alpha
#     grad = torch.where(
#         nonzero,
#         grad,
#         torch.zeros_like(ramp)
#     )
    
    return ramp, grad



def smooth_ramp_pow_beta(x, alpha, beta, eps=1e-8):

    
    xinv = x.clamp_min(eps).reciprocal()
#     xinv = x.reciprocal()
    arg = xinv.pow(beta).neg()
    ramp = ((1 + arg) * alpha).exp()
#     ramp = torch.where(
#         nonzero,
#         ramp,
#         torch.zeros_like(ramp)
#     )
    
    grad_arg = xinv.pow(beta + 1) * beta
    grad = ramp * grad_arg * alpha
#     grad = torch.where(
#         nonzero,
#         grad,
#         torch.zeros_like(ramp)
#     )
    
    return ramp, grad


def generalized_sigmoid_transform(x, ramp):  
    (fx, fx_), (dfx, dfx_) = ramp(torch.stack([x, 1.0 - x], dim=0))
    denom_recip = (fx + fx_).reciprocal()
    cdf = fx * denom_recip 
    numer = (dfx * fx_ + fx * dfx_) 
    pdf = numer * denom_recip.pow(2)   

    return cdf, pdf


def affine_sigmoid_transform(x, ramp, mu, log_sigma, min_density, periodic=True):
    sigma = log_sigma.exp()
    inv_sigma = (-log_sigma).exp()
    
    mu = (mu - 0.5 * inv_sigma)
    if periodic:
        mu = mu % 1
    
    diff = x - mu
    
    zs = torch.stack([diff, -mu, 1 - mu], dim=0)
    if periodic:
        zs = zs % 1
    
    zs = zs * sigma

    (y, y0, y1), (pdf, *_) = generalized_sigmoid_transform(
        zs,
        ramp=ramp
    )
    pdf = pdf * sigma
    
    y = y - y0 
    
    if periodic:
        offset = diff.floor()
        offset += torch.where(mu != 0., torch.ones_like(offset), torch.zeros_like(offset))
        y = y + offset
    else:
        scale = 1. / (y1 - y0)
        y = y * scale
        pdf = pdf * scale
        
    # regularize cdf and compute log density
    y = y * (1. - min_density) + min_density * x
    log_pdf = torch.log(pdf * (1. - min_density) + min_density)
    
    return y, log_pdf


def mixture_cdf_transform(cdfs, log_pdfs, log_weights):        
    
    weights = log_weights.exp()
    
    y = (cdfs * weights).sum(dim=-1)
    
    log_pdf = torch.logsumexp(
        log_pdfs + log_weights,
        dim=-1
    )
    
    return y, log_pdf


class ConditionalRamp(torch.nn.Module):
    
    def forward(self, y, cond):
        raise NotImplementedError()


class PowerRamp(ConditionalRamp):
    
    def __init__(self, power: torch.Tensor):
        super().__init__()
        self.register_buffer("_power", power)
    
    def forward(self, y, cond):
        return polynomial_ramp(y, self._power)


class SmoothRamp(ConditionalRamp):
    """Smooth ramp function with variable slope alpha."""
    
    def __init__(
        self,
        compute_alpha: Callable,
        unimodal: bool=True,
        eps: torch.Tensor=torch.tensor(1e-8),
        max_alpha: torch.Tensor=torch.tensor(10.),
        ramp_type: str="type1"
    ):
        super().__init__()
        
        assert ramp_type in ["type1", "type2"]
        if ramp_type == "type1":
            self._ramp_fn = smooth_ramp_pow1
        elif ramp_type == "type2":
            self._ramp_fn = smooth_ramp_pow2
        
        self._compute_alpha = compute_alpha
        self._unimodal = unimodal
        self.register_buffer("_eps", eps)
        self.register_buffer("_max_alpha", max_alpha)

    def forward(self, out, cond):
        alpha = self._compute_alpha(cond).sigmoid() * self._max_alpha
        alpha = alpha.view(1, 1, *out.shape[2:])
        return self._ramp_fn(out, alpha, unimodal=self._unimodal, eps=self._eps)
    
    
class SmoothRampWithTrainableExponent(ConditionalRamp):
    
    def __init__(
        self,
        compute_params: Callable,
        unimodal: bool=True,
        eps: torch.Tensor=torch.tensor(1e-8),
        max_alpha: torch.Tensor=torch.tensor(10.),
        max_beta: torch.Tensor=torch.tensor(2.),
        min_beta: torch.Tensor=torch.tensor(0.5),
    ):
        super().__init__()        
        self._compute_params = compute_params
        self._unimodal = unimodal
        self.register_buffer("_eps", eps)
        self.register_buffer("_max_alpha", max_alpha)
        self.register_buffer("_max_beta", max_beta)
        self.register_buffer("_min_beta", min_beta)
        
    
    def forward(self, out, cond):
        alpha, beta = self._compute_params(cond).view(1, 1, *out.shape[2:], 2).chunk(2, dim=-1)
        alpha = alpha[..., 0]
        beta = beta[..., 0]
        alpha = alpha.sigmoid() * self._max_alpha
        beta = beta.sigmoid() * (self._max_beta - self._min_beta) + self._min_beta
        return smooth_ramp_pow_beta(out, alpha, beta, eps=self._eps)


class AffineSigmoidComponents(torch.nn.Module):

    def __init__(
        self,
        conditional_ramp: ConditionalRamp,
        compute_params: torch.nn.Module,
        min_density=torch.tensor(1e-4),
        log_sigma_bound=torch.tensor(4.),
        periodic=True,
        zero_boundary_left=False,
        zero_boundary_right=False
    ):
        super().__init__()
        self._conditional_ramp = conditional_ramp
        self._param_net = compute_params
        self.register_buffer("_min_density_lower_bound", min_density)
        self.register_buffer("_log_sigma_bound", log_sigma_bound)
        
        self._periodic = periodic
        self._zero_boundary_left = zero_boundary_left
        self._zero_boundary_right = zero_boundary_right

    def _compute_params(self, cond, out):
        params = self._param_net(cond)
        mu, log_sigma, min_density = params.chunk(3, dim=-1)  
        
        if self._zero_boundary_right:
            # avoid that any mu is placed on 1
            mu = mu = mu.view(*out.shape, -1)
            mu = torch.cat([
                mu,
                mu[..., [-1]]
            ], dim=-1)
            mu = mu.view(*out.shape, -1).softmax(dim=-1).cumsum(dim=-1)
            mu = mu[..., :-1]
        else:
            mu = mu.view(*out.shape, -1).softmax(dim=-1).cumsum(dim=-1)

        log_sigma = log_sigma.view(*out.shape, -1)
        if self._periodic or self._zero_boundary_right or self._zero_boundary_left:
            log_sigma = -self._log_sigma_bound
            log_sigma = log_sigma.sigmoid() * self._log_sigma_bound
            min_value = torch.tensor(1.).to(out)
            if self._zero_boundary_left:
                min_value = torch.minimum(mu, min_value)
            if self._zero_boundary_right:
                min_value = torch.minimum(1. - mu, min_value)
            log_sigma = log_sigma - min_value.log()
        else:
            log_sigma = log_sigma.tanh() * self._log_sigma_bound
        
        # min density in [lb, 1]
        lower_bound = self._min_density_lower_bound.expand_as(min_density)
        if not (self._zero_boundary_right or self._zero_boundary_left):
            lower_bound = lower_bound + min_density.sigmoid() * (1. - self._min_density_lower_bound)

        lower_bound = lower_bound.view(*out.shape, -1)
        return mu, log_sigma, lower_bound
    
    def forward(self, cond, out, *args, **kwargs):

        mu, log_sigma, min_density = self._compute_params(cond=cond, out=out)

        # condition ramp with inputs (necessary for smooth ramps with trainable alpha)
        ramp = partial(self._conditional_ramp, cond=cond)
        
        out = out.unsqueeze(-1)
        
        return affine_sigmoid_transform(
            out, ramp, mu, log_sigma, min_density, periodic=self._periodic
        )


class AffineSigmoidComponentInitGrid(torch.nn.Module):
    
    def __init__(self, components: AffineSigmoidComponents):
        super().__init__()
        self._components = components
        
    def forward(self, cond, inp):
        mu, _, _ = self._components._compute_params(cond, inp)
        if torch.any(mu.min(dim=-1).values > 0):
            mu = torch.cat([
                torch.zeros_like(mu[..., [0]]),
                mu
            ], dim=-1)
        if torch.any(mu.max(dim=-1).values < 1):
            mu = torch.cat([
                mu,
                torch.ones_like(mu[..., [-1]])
            ], dim=-1)
        return mu.permute(-1, *np.arange(len(mu.shape) - 1)).contiguous()
    
    
class MixtureCDFTransformer(Transformer):
    """
    """
    def __init__(
        self,
        compute_components: Callable,
        compute_weights: Callable=None,
    ):
        super().__init__()
        self._compute_components = compute_components
        self._compute_weights = compute_weights

    def _forward(self, cond, out, log_weights=None, *args, elementwise_jacobian=False, **kwargs):
        cdfs, log_pdfs = self._compute_components(cond, out)
        if log_weights is None and self._compute_weights is not None:
            log_weights = self._compute_weights(cond).view(*cdfs.shape).log_softmax(dim=-1)
        else:
            log_weights = torch.zeros(*cdfs.shape).log_softmax(dim=-1)
        out, dlogp = mixture_cdf_transform(cdfs, log_pdfs, log_weights)
        if not elementwise_jacobian:
            dlogp = dlogp.sum(dim=-1, keepdim=True)
        return out, dlogp
    
    def _inverse(self, x, y, *args, **kwargs):
        raise NotImplementedError("No analytic inverse")
        

class ConstrainedSigmoidComponents(torch.nn.Module):
    def __init__(self, compute_params: Callable, smoothness_type: str="type1"):
        super().__init__()
        assert smoothness_type in ["type1", "type2"]
        if smoothness_type == "type1":
            self._k = 1
            self._alpha_offset = _SMOOTH_RAMP_INFLECTION_ALPHA_1
            self._ramp_fn = smooth_ramp_pow1
        elif smoothness_type == "type2":
            self._k = 2
            self._alpha_offset = _SMOOTH_RAMP_INFLECTION_ALPHA_2
            self._ramp_fn = smooth_ramp_pow2
        self._param_net = compute_params
            
    def _compute_params(self, cond, out):
        params = self._param_net(cond)
        mu, log_sigma, log_pdf_constraint = params.view(*out.shape, -1).chunk(3, dim=-1)
        log_sigma = torch.nn.functional.softplus(log_sigma)
        return mu, log_sigma, log_pdf_constraint 

    def forward(self, cond, out):
        mu, log_sigma, log_pdf_constraint = self._compute_params(cond, out)
        alpha = (
            log_pdf_constraint.exp() /  (torch.tensor(2.).to(out).pow(self._k + 1) * self._k * log_sigma.exp())
        ) - self._alpha_offset
        ramp = partial(self._ramp_fn , alpha=alpha, unimodal=True)
        
        min_density = torch.zeros_like(out).unsqueeze(-1)
        return affine_sigmoid_transform(
            out.unsqueeze(-1), ramp, mu, log_sigma, min_density, periodic=False
        )
     

class ConstantConstrainedSigmoidComponents(ConstrainedSigmoidComponents):

    def __init__(self, mu=torch.tensor([0.]), log_pdf_constraint=torch.tensor([np.log(4.)]), smoothness_type: str="type1"):
        super().__init__(compute_params=None, smoothness_type=smoothness_type)
        self._mu = mu
        self._log_sigma = torch.nn.Parameter(torch.zeros_like(mu))
        self._log_pdf_constraint = log_pdf_constraint
            
    def _compute_params(self, _, out):
        mu = self._mu.expand(*out.shape, self._mu.shape[-1])
        log_sigma = torch.nn.functional.softplus(self._log_sigma).expand(*out.shape, self._log_sigma.shape[-1])
        log_pdf_constraint = self._log_pdf_constraint.expand(*out.shape, self._log_pdf_constraint.shape[-1])
        return mu, log_sigma, log_pdf_constraint
    
    
class UniformIntervalConstrainedSigmoidComponents(ConstrainedSigmoidComponents):
    
    def __init__(self, compute_constraints, left_constraint=False, right_constraint=False, smoothness_type: str="type1"):
        super().__init__(compute_params=None, smoothness_type=smoothness_type)
        self._n_constraints = sum([left_constraint, right_constraint])
        self._log_sigma = torch.nn.Parameter(torch.zeros(self._n_constraints))
        self._left_constraint = left_constraint
        self._right_constraint = right_constraint
        self._compute_constraints = compute_constraints
            
    def _compute_params(self, cond, out):
        mu = []
        if self._left_constraint:
            mu.append(torch.zeros_like(out))
        if self._right_constraint:
            mu.append(torch.ones_like(out))
        mu = torch.stack(mu, dim=-1)
        log_sigma = torch.nn.functional.softplus(self._log_sigma).expand(*out.shape, self._n_constraints)
        log_pdf_constraint = self._compute_constraints(cond).view(*out.shape, self._n_constraints) + np.log(self._n_constraints + 1)
        return mu, log_sigma, log_pdf_constraint
    

class ConstrainedBoundaryCDFTransformer(Transformer):
    
    def __init__(self, transformer: Transformer, compute_constraints, left_constraint=True, right_constraint=True, smoothness_type: str="type1"):
        super().__init__()
        self._transformer = transformer
        self._mixture = MixtureCDFTransformer(
            compute_components=self._compute_components
        )
        self._constrained_cdfs = UniformIntervalConstrainedSigmoidComponents(
            compute_constraints=compute_constraints,
            left_constraint=left_constraint,
            right_constraint=right_constraint,
            smoothness_type=smoothness_type
        )
        
    def _compute_components(self, cond, out):
        boundary_cdfs, boundary_pdfs = self._constrained_cdfs(cond, out)
        center_cdf, center_pdf = self._transformer(cond, out)
        cdfs = torch.cat([boundary_cdfs, center_cdf.unsqueeze(-1)], dim=-1)
        pdfs = torch.cat([boundary_pdfs, center_pdf.unsqueeze(-1)], dim=-1)
        return cdfs, pdfs

    def _forward(self, cond, out):
        return self._mixture(cond, out)
