import torch


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
    input_dim = x.shape[-1]
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
