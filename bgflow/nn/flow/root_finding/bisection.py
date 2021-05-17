import warnings

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
    
    
def filter_grid(f, grid):
    fgrid, dfgrid = f(grid)
    lidx = ((fgrid < 0).sum(dim=0, keepdim=True) - 1).clamp_min(0)
    ridx = (lidx + 1).clamp_max(len(grid) - 1)
    left = grid.gather(0, lidx)
    right = grid.gather(0, ridx)
    fleft = fgrid.gather(0, lidx)
    lidx[-1] = 0
    dfleft = dfgrid.gather(0, lidx)
    return (
        torch.linspace(0, 1, len(grid), dtype=grid.dtype, device=grid.device).view(-1, 1, 1) * (right - left) + left,
        (left[0], right[0], fleft[0], dfleft[0]), 
    )


def find_interval(f, grid, threshold=1e-4, max_iters=100, verbose=False, raise_exception=False):
    if verbose:
        print(f"starting grid search with max_iters={max_iters}, threshold={threshold:.4}")
    converged = False
    for it in range(max_iters):
        (grid, (left, right, fleft, dfleft)) = filter_grid(f, grid)
        if torch.all(grid[-1] - grid[0] < threshold):
            converged=True
            break
        if verbose:
            print(f"it: {it}, interval: {(grid[-1] - grid[0]).abs().max().item():.4}, err: {fleft.abs().max().item():.4}")
    if not converged:
        msg = f"interval finding did not converge: err={(right - left).abs().max().item():.5}"
        if raise_exception:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
    return left, right, fleft, dfleft


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