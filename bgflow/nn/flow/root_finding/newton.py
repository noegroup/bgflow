import torch
import numpy as np


__all__ = [
    "NewtonRaphsonRootFinder"
]


# TODO this is not really working so far!
def newton_raphson_inverse(
    f,
    x0,
    abs_tol=1e-6,
    max_iters=100,
    verbose=False,
    max_step=0.1,
    min_step=0.,
    step_shrink=0.9,
    log_derivative=True,
    raise_exception=True
):
    x = x0
    last_val = None    
    max_step = torch.ones_like(x) * max_step
    
    for i in range(max_iters):
        fy, dfy = f(x)
        if last_val is not None:
            max_step=torch.where(
                fy.abs() > last_val.abs(),
                (max_step * step_shrink).clamp_min(min_step),
                max_step
            )
            max_step = (max_step * step_shrink).clamp_min(min_step)
        if verbose:
            if verbose  <= 1:
                print(f"iteration: {i}, error: {(fy).abs().max().item()}")
            else:
                print(f"iteration: {i}, x: {x.detach().cpu().numpy()}, f(x): {fy.detach().cpu().numpy()}")
        if fy.abs().max().item() < abs_tol:
            return x, -dfy
        else:
            dfy = (-dfy).exp()
            step = fy * dfy.expand_as(x) 
            step = torch.where(
                step.abs() > max_step,
                step.sign() * max_step,
                step
            )
            x = x - step
            x = x.clamp(0, 1)
        
        last_val = fy
    if raise_exception:
        raise ValueError(f"Root finding did not converge: error={fy.abs().max().item()}")
    else:
        return x, -dfy
    
    
class NewtonRaphsonRootFinder(torch.nn.Module):
    
    def __init__(
        self,
        abs_tol=torch.tensor(1e-6),
        max_iters=100,
        max_step=torch.tensor(0.1),
        step_shrink=torch.tensor(0.9),
        verbose=False,
        raise_exception=True
    ):
        super().__init__()
        self._max_iters = max_iters
        self.register_buffer("_abs_tol", abs_tol)
        self.register_buffer("_max_step", max_step)
        self.register_buffer("_step_shrink", step_shrink)
        self._verbose=verbose
        self._raise_exception=raise_exception
        
    def forward(self, callback, x0):
        return newton_raphson_inverse(
            callback,
            x0,
            abs_tol=self._abs_tol,
            max_iters=self._max_iters,
            max_step=self._max_step,
            step_shrink=self._step_shrink,
            verbose=self._verbose,
            raise_exception=self._raise_exception
        )