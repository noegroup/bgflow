import warnings

import torch
import numpy as np


__all__ = [
    "stable_newton_raphson"
]


def stable_newton_raphson(
    f,
    left, 
    right,
    fleft=None,
    dfleft=None,
    abs_tol=1e-5,
    max_iters=20,
    verbose=False,
    raise_exception=False
):    
    if verbose:
        print(f"starting stable newton raphson with max_iters={max_iters}, abs_tol={abs_tol:.4}")
    if fleft is None or dfleft is None:
        fleft, dfleft = f(left)    
    x = left
    fx = fleft
    dfx = dfleft
    
    converged = False
    
    for it in range(max_iters):
        
        
        
        step = fx / dfx.exp()
        x_cand = x - step
        x = torch.where(
            (x_cand > left) & (x_cand < right),
            x_cand,
            (left + right) / 2
        )        
        fx, dfx = f(x)        
        err = fx.abs()
        
        if verbose:
            print(f"it: {it}, err: {err.max().item():.4}")
            
        converged = torch.all(
            ((right - left) < abs_tol) | (fx.abs() < abs_tol) | (x == left) | (x == right)
        )

        if converged:
            break

        lt = fx < 0
        gt = fx > 0
        
        left[lt] = x[lt]        
        right[gt] = x[gt]
            
    if not converged:
        msg = f"root finding did not converge: err={err.max().item():.5}"
        if raise_exception:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
        
    return x, -dfx
