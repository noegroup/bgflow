from typing import List
import torch
import numpy as np

from .base import Flow
from ...utils.tensorops import log_dot_exp


def _diag_mask(d: int, a: int, b: int):
    """ Computes the diagonal mask for a BNAF

        Arguments:
            d:  data dimension
            a:  input block factor
            b:  output block factor

        Returns:
            m:  the binary mask
                np.array of shape [d * a, d * b]
    """
    assert d >= 1
    assert a >= 1
    assert b >= 1
    m = np.zeros((a * d, b * d,)).astype(bool)
    # TODO vectorize `for` loop
    for i in range(d):
        m[a * i : a * (i + 1), b * i : b * (i + 1)] = True
    return m


def _off_diag_mask(d: int, a: int, b: int):
    """ Computes the off diagonal mask for a BNAF

        Arguments:
            d:  data dimension
            a:  input block factor
            b:  output block factor

        Returns:
            m:  the binary mask
                np.array of shape [d * a, d * b]
    """
    assert d >= 1
    assert a >= 1
    assert b >= 1
    m = np.zeros((a * d, b * d,)).astype(bool)

    # TODO vectorize `for` loop
    for i in range(a * d):
        for j in range(b * d):
            l = i // a
            k = j // b
            if l < k:
                m[i, j] = True
    return m


def _leaky_relu_gate(x: torch.Tensor, a, b, inverse=False):
    """ Invertible point-wise nonlinearity implemented by leaky relu

        Arguments:
            x: input tensor
            a: positive scaling factor (0 < a)
            b: negative scaling factor (0 < b)

        Returns:
            y:      transformed data
                    torch.Tensor
            dlogp:  log of diagonal jacobian
                    torch.Tensor
    """
    cond = x >= 0
    if not inverse:
        y = cond * x * a + ~cond * x * b
        dlogp = torch.log(cond * a + ~cond * b)
        return y, dlogp
    else:
        y = cond * x / a + ~cond * x / b
        dlogp = torch.log(cond / a + ~cond / b)
        return y, dlogp


def _tanh_gate(x: torch.Tensor, alpha, beta, inverse=False):
    """ Invertible point-wise nonlinearity implemented by mixture of linear and tanh

        Arguments:
            x:      input tensor
            alpha:  bandwidth factor for tanh (0 < alpha)
            beta:   mixing coefficient interpolating between linear and tanh (0 < beta < 1)

        Returns:
            y:      transformed data
                    torch.Tensor
            dlogp:  log of diagonal jacobian
                    torch.Tensor
    """
    if not inverse:
        dlogp = torch.log(
            beta + (1.0 - beta) * alpha * (1.0 - torch.tanh(alpha * x).pow(2))
        )
        y = beta * x + (1.0 - beta) * torch.tanh(alpha * x)
        return y, dlogp
    else:
        raise NotImplementedError()


class _LinearBlockTransformation(torch.nn.Module):
    """ Linear block-wise layer of a BNAF """

    def __init__(self, dim: int, a: int = 1, b: int = 1):
        """ Block-wise linear layer

            Arguments:
                dim: dimension of original data
                a: input block factor (a >= 1)
                b: output block factor (b >= 1)

        """
        super().__init__()
        assert dim >= 1
        assert a >= 1
        assert b >= 1
        self._dim = dim
        self._a = a
        self._b = b
        self._diag_mask = torch.Tensor(_diag_mask(dim, a, b)).bool()
        self._off_diag_mask = torch.Tensor(_off_diag_mask(dim, a, b)).bool()
        self.register_buffer("diag_mask", self._diag_mask)
        self.register_buffer("off_diag_mask", self._off_diag_mask)

        # TODO weight initialization should be optimized
        weight = torch.Tensor(a * dim, b * dim).normal_() / np.sqrt(a * dim + b * dim)
        weight[self._diag_mask] = weight[self._diag_mask].abs().log()
        self._weight = torch.nn.Parameter(weight)
        log_diag = torch.Tensor(1, b * dim).uniform_().log()
        self._log_diag = torch.nn.Parameter(log_diag)
        self._bias = torch.nn.Parameter(torch.Tensor(1, b * dim).zero_())

    @property
    def _weight_and_log_diag(self):
        # log_diag_blocks = self._weight[self._diag_mask].view(
        # 1, self._dim, self._a, self._b
        # )
        diag = self._weight.exp() * self._diag_mask
        offdiag = self._weight * self._off_diag_mask
        weight = diag + offdiag

        # perform weight normalization
        weight_norm = torch.norm(weight, dim=-1, keepdim=True)
        weight = weight / weight_norm
        weight = self._log_diag.exp() * weight

        # accumulate diagonal log blocks
        log_diag_blocks = self._log_diag + self._weight - weight_norm.log()
        log_diag_blocks = log_diag_blocks[self._diag_mask].view(
            1, self._dim, self._a, self._b
        )
        return weight, log_diag_blocks

    def forward(self, x: torch.Tensor, accum_blocks: torch.Tensor = None):
        """ Apply this layer to the input and accumulate the block diagonals

            Arguments:
                x:              input tensor
                accum_blocks:   accumulated log block diagonals from former layers

            Returns:
                x:              transformed tensor
                accum_blocks:   accumulated log block diagonals after applying this layer

        """
        weight, log_diag_blocks = self._weight_and_log_diag
        x = x @ weight + self._bias
        log_diag_blocks = log_diag_blocks.repeat(x.shape[0], 1, 1, 1)
        if accum_blocks is None:
            accum_blocks = log_diag_blocks
        else:
            accum_blocks = log_dot_exp(accum_blocks, log_diag_blocks)
        return x, accum_blocks


class _NonlinearBlockTransformation(torch.nn.Module):
    """ Nonlinear diagonal block-wise layer of a BNAF """

    def __init__(self, dim: int, b: int, alpha: float = 1.0):
        """ Nonlinearity that acts as a diagonal (element-wise) transformation.

            The nonlinearity is a gated tanh, where the gate and the bandwidth
            of the tanh are parameters of the layer:

                y = beta * tanh(alpha * x) + (1-beta) * x

                where

                0 < alpha
                0 < beta < 1

            Arguments:
                dim:    dimension of the data
                b:      output factor of previous block

                alpha:  initial bandwidth of tanh gate
        """
        super().__init__()
        assert dim >= 1
        assert b >= 1
        self._dim = dim
        self._b = b
        self._log_alpha = torch.nn.Parameter(torch.zeros(1, dim * b) + np.log(alpha))
        self._log_beta = torch.nn.Parameter(torch.zeros(1, dim * b))

    def forward(self, x, accum_blocks):
        alpha = self._log_alpha.exp()
        beta = self._log_beta.sigmoid()
        x, log_diag = _tanh_gate(x, alpha, beta)
        log_diag = log_diag.view(
            x.shape[0], accum_blocks.shape[1], 1, accum_blocks.shape[3]
        )
        accum_blocks = accum_blocks + log_diag
        return x, accum_blocks


class BNARFlow(Flow):
    def __init__(self, dim: int, block_sizes: List[int]):
        """ A block neural autoregressive flow

            Arguments:
                dim:            input dimension
                block_sizes:    list of block factors determining hidden layer sizes (> 0)
        """
        super().__init__()
        assert all(a > 0 for a in block_sizes)
        self._dim = dim
        layers = []
        for i, (a, b) in enumerate(zip(block_sizes[:-1], block_sizes[1:])):
            layers.append(_LinearBlockTransformation(dim, a, b))
            if i < len(block_sizes) - 2 and i > 0:
                layers.append(_NonlinearBlockTransformation(dim, b))
        self._layers = torch.nn.ModuleList(layers)
        self._alpha = torch.nn.Parameter(torch.Tensor(1, 1).zero_())

    def _forward(self, x, *args, **kwargs):
        accum_blocks = None
        for layer in self._layers:
            x, accum_blocks = layer(x, accum_blocks)
        return x, accum_blocks.squeeze().sum(dim=-1, keepdim=True)
