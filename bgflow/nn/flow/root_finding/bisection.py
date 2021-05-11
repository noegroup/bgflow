import torch
import numpy as np


__all__ = [
    "BisectionRootFinder"
]


def bisection_inverse(
    f, 
    x0=None,
    left=None, 
    right=None,
    abs_tol=1e-5, 
    max_iters=100, 
    verbose=False, 
    raise_exception=True, 
    *args, 
    **kwargs
):
    if verbose:
        print(f"Starting bisection search with abs_tol={abs_tol:.4} and {max_iters} iterations.")
    assert (left is not None) or (x0 is not None), "either left or x0 has to be given"
    assert (right is not None) or (x0 is not None), "either right or x0 has to be given"
    if left is None:
        left = torch.zeros_like(x0)
    if right is None:
        right = torch.ones_like(x0)
    for i in range(max_iters):
        x = (left + right) / 2.
        fy, dfy = f(x, *args, **kwargs)
        if verbose:
            print(f"iteration: {i}/{max_iters}, error: {fy.abs().max().detach().cpu().numpy():.5}")
        if torch.all(fy.abs() < abs_tol):
            return x, -dfy
        gt = fy >= abs_tol
        lt = fy <= -abs_tol
        right[gt] = x[gt]
        left[lt] = x[lt]        
    if raise_exception:
        raise ValueError(f"Root finding did not converge: error={fy.abs().max().detach().cpu().item():.5}")
    else:
        return x, -dfy


class BisectionRootFinder(torch.nn.Module):

    def __init__(
        self,
        abs_tol=torch.tensor(1e-5),
        max_iters=100,
        verbose=False,
        raise_exception=True
    ):
        super().__init__()
        self._max_iters = max_iters
        self.register_buffer("_abs_tol", abs_tol)
        self._verbose=verbose
        self._raise_exception=raise_exception
    
    def forward(self, callback, x0):
        left = torch.zeros_like(x0)
        right = torch.ones_like(x0)
        return bisection_inverse(
            callback,
            left=left,
            right=right,
            abs_tol=self._abs_tol,
            max_iters=self._max_iters,
            verbose=self._verbose,
            raise_exception=self._raise_exception
        )