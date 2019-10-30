import torch
import numpy as np


# TODO: write docstrings


def _compute_bin_filter(x, grid):
    '''
        x: [n_batch, n_dim]
        grid: [n_batch, n_dim, n_grid]
        
        returns
            bin_filter: [n_batch, n_dim, n_grid + 2]
    '''
    
    x = x.unsqueeze(-1)
#     grid = grid.unsqueeze(0)
    
    is_in = (x >= grid[..., :-1]) & (x < grid[..., 1:])
    is_left = x < grid[..., [0]]
    is_right = x >= grid[..., [-1]]
    
    bin_filter = torch.cat([is_left, is_in, is_right], dim=-1).to(x)
    
    return bin_filter


def _compute_dx(x, grid):
    '''
        x: [n_batch, n_dim]
        grid: [n_batch, n_dim, n_grid]
        
        returns
            dx: [n_batch, n_dim, n_grid]
    '''
    
    x = x.unsqueeze(-1)
#     grid = grid.unsqueeze(0)
    
    dx = x - grid
    dx = torch.cat([
        dx[..., [0]],
        dx
    ], dim=-1)
    
    return dx


def _compute_slope(cdf, grid, eps=1e-7):
    n_batch = cdf.shape[0]
    n_dim = cdf.shape[1]
    slope = (cdf[..., 1:] - cdf[..., :-1]) / (grid[..., 1:] - grid[..., :-1] + eps)
    slope = torch.cat([
        torch.ones(n_batch, n_dim, 1), 
        slope,
        torch.ones(n_batch, n_dim, 1)
    ], dim=-1)
    return slope


def _compute_transform(dx, cdf, slope, bin_filter):
    n_batch = cdf.shape[0]
    n_dim = cdf.shape[1]
    cdf = torch.cat([
        torch.zeros(n_batch, n_dim, 1),
        cdf
    ], dim=-1)
    
    y = ((cdf + dx * slope) * bin_filter).sum(dim=-1)
    logdet = (slope * bin_filter).sum(dim=-1).log().sum(dim=-1, keepdim=True)
    
    return y, logdet


def _compute_cdf(pdf):
    n_batch = pdf.shape[0]
    n_dim = pdf.shape[1]
    cdf = torch.cumsum(pdf, dim=-1)
    cdf = torch.cat([
        torch.zeros(n_batch, n_dim, 1),
        cdf
    ], dim=-1)
    return cdf


def _spline(x, unnormed_pdf, grid=None, inverse=False):
    if len(unnormed_pdf.shape) < 3:
        unnormed_pdf = unnormed_pdf.unsqueeze(0)

    n_batch = unnormed_pdf.shape[0]
    n_dim = unnormed_pdf.shape[1]
    n_grid = unnormed_pdf.shape[2]
    
    pdf = torch.softmax(unnormed_pdf, dim=-1)
    
    cdf = _compute_cdf(pdf)
    
    if grid is None:
        grid = torch.linspace(0, 1, n_grid+1).repeat(n_batch, n_dim, 1)
    
    if inverse:
        cdf, grid = grid, cdf
    
    bin_filter = _compute_bin_filter(x, grid)
    dx = _compute_dx(x, grid)
    slope = _compute_slope(cdf, grid)
    
    y, logdet = _compute_transform(dx, cdf, slope, bin_filter)
    
    return y, logdet


class LinearSplineLayer(torch.nn.Module):
    
    def __init__(self, n_points, n_dims, min_val=-1., max_val=1.):
        super().__init__()
        self._n_points = n_points
        self._n_dims = n_dims

        self._pdf = torch.nn.Parameter(
            torch.Tensor(1, n_dims, self._n_points).zero_()
        )
        
        self._min_val = min_val
        self._max_val = max_val
    
    def forward(self, x, inverse=False):
        n_dim = x.shape[-1]
        x = (x - self._min_val) / (self._max_val - self._min_val)
        y, logdet = _spline(x, self._pdf, grid=None, inverse=inverse)
        y = y * (self._max_val - self._min_val) + self._min_val
        return y, logdet
