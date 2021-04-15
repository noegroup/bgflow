import torch
from .shape import tile


def distance_vectors(x, remove_diagonal=True):
    """
    Computes the matrix `r` of all distance vectors between
    given input points where

        ``r_{ij} = x_{i} - x{j}``

    Parameters
    ----------
    x : PyTorch tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - x_i` should be included in the result

    Returns
    -------
    r : PyTorch tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    Examples
    --------
    TODO
    """
    r = tile(x.unsqueeze(2), 2, x.shape[1])
    r = r - r.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distance_vectors_v2(x, y, remove_diagonal=True):
    """
    Computes the matrix `r` of all distance vectors between
    given input points x and y where

        ``r_{ij} = x_{i} - y{j}``

    Parameters
    ----------
    x : PyTorch tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    y : PyTorch tensor
        Tensor of shape `[n_batch, n_particles, n_dimensions]`
        containing input points.
    remove_diagonal : boolean
        Flag indicating whether the all-zero distance vectors
        `x_i - y_i` should be included in the result

    Returns
    -------
    r : PyTorch tensor
        Matrix of all distance vectors r.
        If `remove_diagonal=True` this is a tensor of shape
            `[n_batch, n_particles, n_particles, n_dimensions]`.
        Otherwise this is a tensor of shape
            `[n_batch, n_particles, n_particles - 1, n_dimensions]`.

    Examples
    --------
    TODO
    """
    r1 = tile(x.unsqueeze(2), 2, x.shape[1])
    r2 = tile(y.unsqueeze(2), 2, y.shape[1])
    r = r1 - r2.permute([0, 2, 1, 3])
    if remove_diagonal:
        r = r[:, torch.eye(x.shape[1], x.shape[1]) == 0].view(
            -1, x.shape[1], x.shape[1] - 1, x.shape[2]
        )
    return r


def distances_from_vectors(r, eps=1e-6):
    """
    Computes the all-distance matrix from given distance vectors.
    
    Parameters
    ----------
    r : PyTorch tensor
        Matrix of all distance vectors r.
        Tensor of shape `[n_batch, n_particles, n_other_particles, n_dimensions]`
    eps : Small real number.
        Regularizer to avoid division by zero.
    
    Returns
    -------
    d : PyTorch tensor
        All-distance matrix d.
        Tensor of shape `[n_batch, n_particles, n_other_particles]`.
    """
    return (r.pow(2).sum(dim=-1) + eps).sqrt()
