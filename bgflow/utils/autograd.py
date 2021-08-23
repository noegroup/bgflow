import torch

import einops
from collections import namedtuple
from typing import Iterable


__all__ = [
    "brute_force_jacobian_trace", "brute_force_jacobian",
    "batch_jacobian", "get_jacobian", "requires_grad"
]


def brute_force_jacobian_trace(y, x):
    """
    Computes the trace of the jacobian matrix `dy/dx` by
    brute-forcing over the components of `x`. This requires
    `O(D)` backward passes, where D is the dimension of `x`.

    Parameters
    ----------
    y: PyTorch tensor
        Result of a computation depending on `x`.
        Tensor of shape `[n_batch, n_dimensions_in]`
    x: PyTorch tensor
        Argument to the computation yielding `y`.
        Tensor of shape `[n_batch, n_dimensions_out]`

    Returns
    -------
    trace: PyTorch tensor
        Trace of Jacobian matrix `dy/dx`.
        Tensor of shape `[n_batch, 1]`.

    Examples
    --------
    TODO
    """
    system_dim = x.shape[-1]
    trace = 0.0
    for i in range(system_dim):
        dyi_dx = torch.autograd.grad(
            y[:, i], x, torch.ones_like(y[:, i]), create_graph=True, retain_graph=True
        )[0]
        trace = trace + dyi_dx[:, i]
    return trace.contiguous()


def brute_force_jacobian(y, x):
    """
    Compute the jacobian matrix `dy/dx` by
    brute-forcing over the components of `x`. This requires
    `O(D)` backward passes, where D is the dimension of `x`.

    Parameters
    ----------
    y: PyTorch tensor
        Result of a computation depending on `x`
        Tensor of shape `[n_batch, n_dimensions_in]`
    x: PyTorch tensor
        Argument to the computation yielding `y`
        Tensor of shape `[n_batch, n_dimensions_out]`

    Returns
    -------
    trace: PyTorch tensor
        Jacobian matrix `dy/dx`.
        Tensor of shape `[n_batch, n_dimensions_in, n_dimensions_out]`.

    Examples
    --------
    TODO
    """
    output_dim = y.shape[-1]
    rows = []
    for i in range(output_dim):
        row = torch.autograd.grad(
            y[..., i],
            x,
            torch.ones_like(y[..., i]),
            create_graph=True,
            retain_graph=True,
        )[0]
        rows.append(row)
    jac = torch.stack(rows, dim=-2)
    return jac


def batch_jacobian(y, x):
    """
    Compute the Jacobian matrix in batch form.
    Return (B, D_y, D_x)
    """
    import numpy as np

    batch = y.shape[0]
    single_y_size = np.prod(y.shape[1:])
    y = y.view(batch, -1)
    vector = torch.ones(batch).to(y)

    # Compute Jacobian row by row.
    # dy_i / dx -> dy / dx
    # (B, D) -> (B, 1, D) -> (B, D, D)
    jac = [
        torch.autograd.grad(
            y[:, i], x, grad_outputs=vector, retain_graph=True, create_graph=True
        )[0].view(batch, -1)
        for i in range(single_y_size)
    ]
    jac = torch.stack(jac, dim=1)

    return jac


Jacobian = namedtuple("Jacobian", ["y", "jac"])


def get_jacobian(fun, x):
    """
    Computes the jacobian of `fun` wrt to `x`.

    This evaluates the jacobian via batch parallelization in one step having armotized cost of O(1) rather O(d).

    However, comes at a memory cost of O(d) and thus will only work for low d (d < 1000s).

    Parameters:
    -----------
    fun: callable
        function from which the jacobian is to be computed
    x: torch.Tensor
        input tensor for which the jacobian is to be computed

    Returns:
    --------
    jac: Jacobian
        Named tuple containing members `y` and `jac`.
        `y` is the function value  of `fun` evaluated at `x`
        `jac` is the jacobian matrix of `fun` evaluated at `x`
    """
    shape = x.shape[:-1]
    d = x.shape[-1]
    x = x.view(-1, d)
    n = x.shape[0]
    z = einops.repeat(x, "n j -> (n i) j", i=d)
    z.requires_grad_(True)
    y = fun(z)
    out_grad = torch.eye(d, device=x.device, dtype=x.dtype).tile(n, 1)
    j = torch.autograd.grad(y, z, out_grad, create_graph=True, retain_graph=True)[0].view(*shape, d, d)
    return Jacobian(
        y=einops.rearrange(y, "(n i) j -> n i j", i=d)[:, 0, :].view(*shape, -1),
        jac=j
    )


class requires_grad(torch.enable_grad):
    """
    This environment guarantees, that all `nodes` have gradients within the scope
    and can be called used as arguments for PyTorch's autograd engine.

    It furthermore takes care for cleaning up, namely setting all nodes back to
    their original gradient state.

    Parameters:
    -----------
    nodes: Iterable[torch.Tensor]
        iterable of PyTorch nodes that should be wrapped by this environment
    """

    def __init__(self, *nodes: Iterable[torch.Tensor]):
        self._nodes = nodes
        self._grad_state = None

    def __enter__(self):
        super().__enter__()
        self._grad_states = []
        for node in self._nodes:
            self._grad_states.append(node.requires_grad)
            node.requires_grad_(True)

    def __exit__(self, *args, **kwargs):
        for node, state in zip(self._nodes, self._grad_states):
            node.requires_grad_(state)
        super().__exit__(*args, **kwargs)
