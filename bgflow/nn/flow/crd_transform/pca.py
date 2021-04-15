import numpy as np
import torch
from bgflow.nn.flow.base import Flow


__all__ = ["WhitenFlow"]


def _pca(X0, keepdims=None):
    """Implements PCA in Numpy.

    This is not written for training in torch because derivatives of eig are not implemented

    """
    if keepdims is None:
        keepdims = X0.shape[1]

    # pca
    X0mean = X0.mean(axis=0)
    X0meanfree = X0 - X0mean
    C = np.matmul(X0meanfree.T, X0meanfree) / (X0meanfree.shape[0] - 1.0)
    eigval, eigvec = np.linalg.eigh(C)

    # sort in descending order and keep only the wanted eigenpairs
    I = np.argsort(eigval)[::-1]
    I = I[:keepdims]
    eigval = eigval[I]
    std = np.sqrt(eigval)
    eigvec = eigvec[:, I]

    # whiten and unwhiten matrices
    Twhiten = np.matmul(eigvec, np.diag(1.0 / std))
    Tblacken = np.matmul(np.diag(std), eigvec.T)
    return X0mean, Twhiten, Tblacken, std


class WhitenFlow(Flow):
    def __init__(self, X0, keepdims=None, whiten_inverse=True):
        """Performs static whitening of the data given PCA of X0

        Parameters:
        -----------
        X0 : array
            Initial Data on which PCA will be computed.
        keepdims : int or None
            Number of dimensions to keep. By default, all dimensions will be kept
        whiten_inverse : bool
            Whitens when calling inverse (default). Otherwise when calling forward

        """
        super().__init__()
        if keepdims is None:
            keepdims = X0.shape[1]
        self.dim = X0.shape[1]
        self.keepdims = keepdims
        self.whiten_inverse = whiten_inverse

        X0_np = X0.detach().cpu().numpy()
        X0mean, Twhiten, Tblacken, std = _pca(X0_np, keepdims=keepdims)
        # self.X0mean = torch.tensor(X0mean)
        self.register_buffer("X0mean", torch.tensor(X0mean).to(X0))
        # self.Twhiten = torch.tensor(Twhiten)
        self.register_buffer("Twhiten", torch.tensor(Twhiten).to(X0))
        # self.Tblacken = torch.tensor(Tblacken)
        self.register_buffer("Tblacken", torch.tensor(Tblacken).to(X0))
        # self.std = torch.tensor(std)
        self.register_buffer("std", torch.tensor(std).to(X0))
        if torch.any(self.std <= 0):
            raise ValueError(
                "Cannot construct whiten layer because trying to keep nonpositive eigenvalues."
            )
        self.jacobian_xz = -torch.sum(torch.log(self.std))

    def _whiten(self, x):
        # Whiten
        output_z = torch.matmul(x - self.X0mean, self.Twhiten)
        # if self.keepdims < self.dim:
        #    junk_dims = self.dim - self.keepdims
        #    output_z = torch.cat([output_z, torch.Tensor(x.shape[0], junk_dims).normal_()], dim=1)
        # Jacobian
        dlogp = self.jacobian_xz * torch.ones((x.shape[0], 1)).to(x)

        return output_z, dlogp

    def _blacken(self, x):
        # if we have reduced the dimension, we ignore the last dimensions from the z-direction.
        # if self.keepdims < self.dim:
        #    x = x[:, 0:self.keepdims]
        output_x = torch.matmul(x, self.Tblacken) + self.X0mean
        # Jacobian
        dlogp = -self.jacobian_xz * torch.ones((x.shape[0], 1)).to(x)

        return output_x, dlogp

    def _forward(self, x, *args, **kwargs):
        if self.whiten_inverse:
            y, dlogp = self._blacken(x)
        else:
            y, dlogp = self._whiten(x)
        return y, dlogp

    def _inverse(self, x, *args, **kwargs):
        if self.whiten_inverse:
            y, dlogp = self._whiten(x)
        else:
            y, dlogp = self._blacken(x)
        return y, dlogp
