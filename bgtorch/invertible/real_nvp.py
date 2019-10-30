import torch
import numpy as np

from .splines import _spline


class AffineTransformer(torch.nn.Module):
    
    def __init__(self, masked_transform, dt=1.):
        super().__init__()
        self._masked_transform = masked_transform
        assert dt > 0
        self._dt = dt
        
    def forward(self, x, y, inverse=False, dt=None):
        if dt is None:
            dt = self._dt
        assert dt > 0
        
        n_dim_left = x.shape[-1]
        n_dim_right = y.shape[-1]
        
        mu_log_sigma = self._masked_transform(x)
        if (mu_log_sigma.shape[-1] != 2 * n_dim_right):
            raise ValueError("`masked_transform` has wrong output dimension: expected {} but was {}".format(
                2 * n_dim_right, mu_log_sigma.shape[-1]))
            
        mu = mu_log_sigma[..., :n_dim_right]
        log_sigma = mu_log_sigma[..., n_dim_right:]
            
        if not inverse:
            sigma = torch.exp(log_sigma)
            dlogp = log_sigma.sum(dim=-1, keepdim=True)
            
            y = sigma * y
            y = y + mu * dt
        else:
            sigma_inv = torch.exp(-log_sigma)
            dlogp = -log_sigma.sum(dim=-1, keepdim=True)
            
            y = y - mu * dt
            y = sigma_inv * y
            
        return y, dlogp


class SplineTransformer(torch.nn.Module):
    
    def __init__(self, masked_conditioner):
        super().__init__()
        self._masked_conditioner = masked_conditioner
        
    def forward(self, x, y, inverse=False, dt=None):
        n_batch = x.shape[0]
        n_dim = y.shape[1]
        pdf = self._masked_conditioner(x).view(n_batch, n_dim, -1)
        y, dlogp = _spline(y, pdf, inverse=inverse)
        print(y.shape, dlogp.shape)
        return y, dlogp
        
        
class RealNVP(torch.nn.Module):
    
    def __init__(self, transformer, dt=1.):
        """
        Coupling layer based on the RealNVP design.
        
        Parameters:
        -----------
        masked_transform: PyTorch Module
            Function mapping the left part of the input `x_left` to 
                1. a translation vector `t`
                2. a scaling vector `s`
            If `x = (x_left, x_right)` with 
                `x_left : [..., n_dims_left]`
                `x_right : [..., n_dims_right]`
            then `masked_transform` must map from
                `[..., n_dim_left]`
            to
                `[..., 2 * n_dim_right]`.
        dt : Float > 0
            Integration time step (if RealNVP is reinterpreted as a symplectic integrator).
            If K RealNVP layers are used in a flow, it should be set to `dt=1./K`.
        """
        super().__init__()
        
        self._transformer = transformer
        
        assert dt > 0
        self._dt = dt
        
    def forward(self, x_left, x_right, inverse=False, dt=None):
        """
            Transforms the input tuple to the output tuple.
            
            Parameters:
            -----------
            x_left : PyTorch Tensor
                Left part of the input.
                Tensor of shape `[..., n_dims_left]`.
            x_right : PyTorch Tensor
                Right part of the input.
                Tensor of shape `[..., n_dims_right]`.
            inverse : Boolean.
                If set to `True` computes the inverse transform.;
            dt : Float
                Integration time step (if RealNVP is reinterpreted as a symplectic integrator).
                If K RealNVP layers are used in a flow, it should be set to `dt=1./K`.
                
            Returns:
            --------
            z : Out
        """
        if dt is None:
            dt = self._dt
        assert dt > 0
        
        x_right, dlogp = self._transformer(x_left, x_right, dt=dt, inverse=inverse)
            
        return x_left, x_right, dlogp


class _RealNVP_old(torch.nn.Module):
    
    def __init__(self, masked_transform, dt=1.):
        """
        Coupling layer based on the RealNVP design.
        
        Parameters:
        -----------
        masked_transform: PyTorch Module
            Function mapping the left part of the input `x_left` to 
                2. a scaling vector `s`
            If `x = (x_left, x_right)` with 
                `x_left : [..., n_dims_left]`
                `x_right : [..., n_dims_right]`
            then `masked_transform` must map from
                `[..., n_dim_left]`
            to
                `[..., 2 * n_dim_right]`.
        dt : Float > 0
            Integration time step (if RealNVP is reinterpreted as a symplectic integrator).
            If K RealNVP layers are used in a flow, it should be set to `dt=1./K`.
        """
        super().__init__()
        
        self._masked_transform = masked_transform
        
        assert dt > 0
        self._dt = dt
        
    def forward(self, x_left, x_right, inverse=False, dt=None):
        """
            Transforms the input tuple to the output tuple.
            
            Parameters:
            -----------
            x_left : PyTorch Tensor
                Left part of the input.
                Tensor of shape `[..., n_dims_left]`.
            x_right : PyTorch Tensor
                Right part of the input.
                Tensor of shape `[..., n_dims_right]`.
            inverse : Boolean.
                If set to `True` computes the inverse transform.;
            dt : Float
                Integration time step (if RealNVP is reinterpreted as a symplectic integrator).
                If K RealNVP layers are used in a flow, it should be set to `dt=1./K`.
                
            Returns:
            --------
            z : Out
        """
        if dt is None:
            dt = self._dt
        assert dt > 0
        
        n_dim_left = x_left.shape[-1]
        n_dim_right = x_right.shape[-1]
        
        mu_log_sigma = self._masked_transform(x_left)
        if (mu_log_sigma.shape[-1] != 2 * n_dim_right):
            raise ValueError("`masked_transform` has wrong output dimension: expected {} but was {}".format(
                2 * n_dim_right, mu_log_sigma.shape[-1]))
            
        mu = mu_log_sigma[..., :n_dim_right]
        log_sigma = mu_log_sigma[..., n_dim_right:]
            
        if not inverse:
            sigma = torch.exp(dt * log_sigma)
            dlogp = (dt * log_sigma).sum(dim=-1, keepdim=True)
            
            x_right = sigma * x_right
            x_right = x_right + mu * dt
        else:
            sigma_inv = torch.exp(-dt * log_sigma)
            dlogp = (-dt * log_sigma).sum(dim=-1, keepdim=True)
            
            x_right = x_right - mu * dt
            x_right = sigma_inv * x_right
            
        return x_left, x_right, dlogp