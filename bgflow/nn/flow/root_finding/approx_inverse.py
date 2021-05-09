import torch
    
from bgflow.nn.flow.base import Flow
from bgflow.nn.flow.transformer import Transformer

__all__ = [
    "DifferentiableApproximateInverse",
    "WrapFlowWithInverse",
    "TransformerToFlowAdapter",
    "WrapTransformerWithInverse"
]


class DifferentiableApproximateInverse(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, root_finder, bijection, y, *params):
        
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
        x, *params = ctx.saved_tensors        
 
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            y, dlogp = ctx.bijection(x)

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
        
        return None, None, grad_in_y, *grad_in_params
    

class WrapFlowWithInverse(Flow):
    
    def __init__(self, flow, root_finder):
        super().__init__()
        self._flow = flow
        self._root_finder = root_finder
        
    def _forward(self, x, *args, **kwargs):
        return self._flow(x, *args, **kwargs)
    
    def _inverse(self, y, *args, **kwargs):
        return DifferentiableApproximateInverse.apply(
            self._root_finder,
            self,
            y,
            *self.parameters()
        )

    
class TransformerToFlowAdapter(Flow):
    
    def __init__(self, transformer, cond):
        super().__init__()
        self._transformer = transformer
        self._cond = cond
    
    def _forward(self, out, *args, **kwargs):
        return self._transformer(self._cond, out, *args, **kwargs)
    
    def _inverse(self, out, *args, **kwargs):
        return self._transformer(self._cond, out, inverse=True, *args, **kwargs)
    
    
class WrapTransformerWithInverse(Transformer):
    
    def __init__(self, transformer, root_finder):
        super().__init__()
        self._transformer = transformer
        self._root_finder = root_finder
        
    def _forward(self, cond, out, *args, **kwargs):
        return self._transformer(cond, out, *args, **kwargs)
    
    def _inverse(self, cond, out, *args, **kwargs):
        flow = TransformerToFlowAdapter(self._transformer, cond=cond)
        return DifferentiableApproximateInverse.apply(
            self._root_finder,
            flow,
            out,
            *flow.parameters()
        )