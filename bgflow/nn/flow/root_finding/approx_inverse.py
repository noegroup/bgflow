import numpy as np
import torch
    
from bgflow.nn.flow.base import Flow
from bgflow.nn.flow.transformer import Transformer
from bgflow.nn.flow.root_finding.bisection import (
    find_interval
)

__all__ = [
#     "DifferentiableApproximateInverse",
#     "WrapFlowWithInverse",
#     "TransformerToFlowAdapter",
#     "WrapTransformerWithInverse",
    "WrapCDFTransformerWithInverse",
    "GridInversion"
]


class DifferentiableApproximateInverse(torch.autograd.Function):
    """First-order-differentiation of an elementwise bijection under black-box inversion.
    """
    
    @staticmethod
    def forward(ctx, root_finder, bijection, y, *params):
        """
        Inverse pass

        Parameters
        ----------
        ctx : object
            context object to stash information for the backward call
        root_finder : Callable
            Root finding method, which takes two parameters (residue, x0)
        bijection : Flow
            a flow object, whose forward function returns (y, dlogp)
        y : torch.Tensor
            the input to the inverse
        params : tuple
            further arguments for the ctx.save_for_backward call

        Returns
        -------
        x : torch.Tensor
            the root of the bijection, bijection^-1(y)
        log_dx_dy : torch.Tensor
            the log derivative

        """

        def residual(x):
            y_, log_dy_dx = bijection(x)
            return y_ - y, log_dy_dx
        
        with torch.no_grad():
            x, log_dx_dy = root_finder(residual, x0=y)
            
        ctx.save_for_backward(x, *params)
        ctx.bijection = bijection
        
        return x, log_dx_dy
    
    @staticmethod
    def backward(ctx, grad_out_x, grad_out_dlogp):
        """
        Backpropagation through the inverse

        Parameters
        ----------
        ctx : object
            context object to stash information for the backward call
        grad_out_x
        grad_out_dlogp

        Returns
        -------
        A tuple:
            - None
            - None
            - grad_in_y : torch.Tensor
            -

        """
        x, *params = ctx.saved_tensors        
 
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            y, dlogp = ctx.bijection(x, diagonal_jacobian=True)

            force = torch.autograd.grad(-dlogp.sum(), x, create_graph=True)[0]
            
            grad_out_x = (-dlogp).exp() * grad_out_x
            grad_out_dlogp = (-dlogp).exp() * grad_out_dlogp
            
            grad_in_y = grad_out_x + force * grad_out_dlogp
            
            grad_in_params = torch.autograd.grad(
                [
                    y,
                    y, 
                    dlogp.exp()
                ], 
                params, 
                grad_outputs=[
                    -grad_out_x,
                    -force * grad_out_dlogp,
                    -grad_out_dlogp
                ], 
                create_graph=True
            )
        
        return (None, None, grad_in_y, *grad_in_params)
    

class WrapFlowWithInverse(Flow):
    """A flow, where the inverse is computed in approximate fashion by a root finder.
    """
    
    def __init__(self, flow, root_finder):
        super().__init__()
        self._flow = flow
        self._root_finder = root_finder
        
    def _forward(self, x, *args, **kwargs):
        return self._flow(x, *args, **kwargs)
    
    def _inverse(self, y, *args, elementwise_jacobian=False, **kwargs):
        x, dlogp = DifferentiableApproximateInverse.apply(
            self._root_finder,
            self,
            y,
            *self.parameters()
        )
        if not elementwise_jacobian:
            dlogp = dlogp.sum(-1, keepdim=True)
        return x, dlogp


    
class TransformerToFlowAdapter(Flow):
    """Transformer with constant conditioner.
    """
    
    def __init__(self, transformer, cond):
        super().__init__()
        self._transformer = transformer
        self._cond = cond

    def _forward(self, out, *args, **kwargs):
        return self._transformer(self._cond, out, *args, **kwargs)
    
    def _inverse(self, out, *args, **kwargs):
        return self._transformer(self._cond, out, inverse=True, *args, **kwargs)

    
class WrapTransformerWithInverse(Transformer):
    """Transformer with approximate black-box inversion (and variable conditioning).
    """
    
    def __init__(self, transformer, root_finder):
        super().__init__()
        self._transformer = transformer
        self._root_finder = root_finder

    def _forward(self, cond, out, *args, **kwargs):
        return (self._transformer(cond, out, *args, **kwargs))
    
    def _inverse(self, cond, out, *args, elementwise_jacobian=False, **kwargs):
        flow = TransformerToFlowAdapter(self._transformer, cond=cond)
        x, dlogp = DifferentiableApproximateInverse.apply(
            self._root_finder,
            flow,
            out,
            *flow.parameters()
        )
        if not elementwise_jacobian:
            dlogp = dlogp.sum(-1, keepdim=True)
        return x, dlogp
    

class TransformerApproximateInverse(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, oracle, bijection, cond, out, *params):
        
        with torch.no_grad():
            inp, dlogp = oracle(cond, out)
            
        ctx.save_for_backward(cond, inp, *params)
        ctx.bijection = bijection
        
        return inp, dlogp
    
    @staticmethod
    def backward(ctx, grad_out_x, grad_out_dlogp):        
        cond, x, *params = ctx.saved_tensors        
 
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            y, dlogp = ctx.bijection(cond, x)

            force = torch.autograd.grad(-dlogp.sum(), x, create_graph=True)[0]
            
            grad_out_x = (-dlogp).exp() * grad_out_x
            grad_out_dlogp = (-dlogp).exp() * grad_out_dlogp
            
            grad_in_y = grad_out_x + force * grad_out_dlogp
            
            grad_in_params = torch.autograd.grad(
                [
                    y,
                    y, 
                    dlogp.exp()
                ], 
                params, 
                grad_outputs=[
                    -grad_out_x,
                    -force * grad_out_dlogp,
                    -grad_out_dlogp
                ], 
                create_graph=True
            )
        
        return (None, None, None, grad_in_y, *grad_in_params)
    
    
class WrapCDFTransformerWithInverse(Transformer):
    
    def __init__(self, transformer, oracle):
        super().__init__()
        self._transformer = transformer
        self._oracle = oracle
    
    def _forward(self, *args, **kwargs):
        return self._transformer(*args, **kwargs)
    
    def _inverse(self, cond, out, *args, **kwargs):
        return TransformerApproximateInverse.apply(
            self._oracle,
            self._transformer,
            cond,
            out,
            *self._transformer.parameters()
        )
    
    
class GridInversion(Transformer):
    
    def __init__(
        self,
        transformer,
        compute_init_grid,
        verbose=False,
        abs_tol=1e-6,
        newton_threshold=1e-4,
        max_iters=100,
        raise_exception=False
    ):
        super().__init__()
        self._transformer = transformer
        self._compute_init_grid = compute_init_grid
        self._verbose = verbose
        self._abs_tol = abs_tol
        self._newton_threshold = newton_threshold
        self._max_iters = max_iters
        self._raise_exception = raise_exception
    
    def forward(self, cond, out, *args, **kwargs):
        
        def _residual(inp):

            n_extra_dims = len(inp.shape) - len(cond.shape)
            extra_dims = inp.shape[:n_extra_dims]
            out_pred, dlogp = self._transformer(
                cond.view(*np.ones(n_extra_dims, dtype=int), *cond.shape).expand(*extra_dims, *cond.shape),
                inp,
                *args,
                **kwargs
            )
            return out_pred - out, dlogp
        
        init_grid = self._compute_init_grid(cond, out)
        left, right, fleft, dfleft = find_interval(
            _residual,
            init_grid,
            verbose=self._verbose,
            threshold=self._abs_tol,
            max_iters=self._max_iters,
            raise_exception=self._raise_exception
        )
        
        return left, -dfleft
