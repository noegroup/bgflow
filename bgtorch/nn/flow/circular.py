import torch
import numpy as np

from .base import Flow


# TODO UNIT TESTS!


def _relu_3_ramp(x):
    """ Relu(x) ** 3 ramp function
        returns

            f(x)     = relu(x) ** 3
            df/dx(x) = relu(x) ** 2
    """
    rx = torch.relu(x)
    ramp = rx.pow(3)
    grad = rx.pow(2) * 3.0
    return ramp, grad


def _bump_fn(x, alpha=1.0, with_pdf=False):
    """ computes bump function (pdf) together with cdf """
    # TODO: speed up numerics

    # compute ramps + grads
    fx, dfx = _relu_3_ramp(x)
    fx_, dfx_ = _relu_3_ramp(1.0 - x)

    # expensive: 1/x
    denom_recip = (fx + fx_).reciprocal_()
    cdf = fx * denom_recip

    numer = dfx * fx_ + fx * dfx_
    if with_pdf:
        pdf = numer * denom_recip.pow(2)
        return cdf, pdf
    else:
        return cdf, None


def _bump_fn_distr(x, mu, sigma, alpha=1.0, with_pdf=False):
    """ wraps bump function distribution around circle and returns pdf, cdf """

    # TODO: maybe more efficient way?
    u = x - mu

    # compute toroid distance
    d, i = torch.min(torch.stack([u.abs(), 1.0 - u.abs()], dim=-1), dim=-1)

    # create periodic ramp
    case = 2.0 * (mu > 0.5) - 1.0
    sign = torch.where(i.bool(), case * torch.ones_like(u), u.sign())
    z = sigma * (d * sign) + 0.5

    # compute dc offset for cdf
    d0 = torch.min(mu.abs(), 1.0 - mu.abs())
    z0 = sigma * (d0 * case) + 0.5

    case = (mu > 0.5).float()
    offset = case * (x > mu - 0.5) + (1.0 - case) * (x > mu + 0.5)

    cdf, pdf = _bump_fn(z, alpha=alpha, with_pdf=with_pdf)
    cdf0, _ = _bump_fn(z0, alpha=alpha, with_pdf=False)
    cdf.add_(offset - cdf0)
    pdf.mul_(sigma)

    return cdf, pdf


def _cdf_transform(x, mu, sigma, weight, eps, alpha=1.0):
    """
        transforms x according to a cdf defined by a mixture
        of circular bump function distributions

        Arguments:

            x:       [n_batch, dim]
            mu:      [n_batch, n_basis, dim]
            sigma:   [n_batch, n_basis, dim]
            weight:  [n_batch, n_basis, dim]
            eps:     [n_batch, dim]
    """
    # [n_batch, 1, dim]
    x = x.unsqueeze(1)
    if isinstance(eps, torch.Tensor):
        eps = eps.unsqueeze(1)

    cdf, pdf = _bump_fn_distr(x, mu, sigma, alpha, with_pdf=True)

    # apply weight
    cdf.mul_(weight)
    pdf.mul_(weight)

    # reduce and add zero probability offset
    y = (cdf.sum(1, keepdim=True) * (1.0 - eps) + x * eps).view(x.shape[0], -1)
    dlogp = ((pdf.sum(1, keepdim=True) * (1.0 - eps) + eps).log()).view(x.shape[0], -1)

    return y, dlogp


def _bisect(y, f, lb=0.0, ub=1.0, max_it=100, eps=1e-6):
    """ inverts a cdf f evaluated at y using bisection method

    lb: lower bound
    ub: upper bound
    max_it: maximum iterations of bisection
    eps: numerical tolerance
    """

    assert y.min() >= 0 and y.max() <= 1
    x = torch.zeros_like(y)
    diff = float("inf")
    it = 0
    while diff > eps and it < max_it:
        cur, dlogp = f(x)
        gt = (cur > y).to(y)
        lt = 1.0 - gt
        new_x = gt * (x + lb) / 2.0 + lt * (x + ub) / 2.0
        lb = gt * lb + lt * x
        ub = gt * x + lt * ub
        diff = (new_x - x).abs().max()
        x = new_x
        it += 1
    return x, -dlogp


class CircularTransformSimple(Flow):
    """
    Simple circular flow based on a mixture of bump function distributions.

    Here the transformation of `y` *IS NOT* conditioned on `x`.
    """

    def __init__(self, n_bases=10, n_dim=1):
        super().__init__()
        self._mu = torch.nn.Parameter(
            torch.Tensor(1, n_bases, n_dim).uniform_(0, 2 * np.pi)
        )
        self._log_sigma = torch.nn.Parameter(torch.Tensor(1, n_bases, n_dim).normal_())
        self._log_weight = torch.nn.Parameter(torch.Tensor(1, n_bases, n_dim).normal_())
        self._log_eps = torch.nn.Parameter(torch.Tensor(1, n_dim).normal_())

    def _params(self):
        mu = 0.5 * torch.sin(self._mu) + 0.5
        sigma = 1.0 + self._log_sigma.exp()
        weight = self._log_weight.softmax(1)
        eps = self._log_eps.sigmoid()
        return mu, sigma, weight, eps

    def _forward(self, x, y, *args, **kwargs):
        mu, sigma, weight, eps = self._params()
        cdf, pdf = _cdf_transform(y, mu, sigma, weight, eps=eps)
        dlogp = pdf.sum(dim=-1, keepdim=True)
        cdf = cdf.view(*y.shape)
        return cdf, dlogp

    def _inverse(self, x, y, *args, **kwargs):
        mu, sigma, weight, eps = self._params()

        def _callback(y):
            cdf, pdf = _cdf_transform(y, mu, sigma, weight, eps=eps)
            cdf = cdf.view(*y.shape)
            dlogp = pdf.sum(dim=-1, keepdim=True)
            return cdf, dlogp

        # compute inverse via bisection
        return _bisect(y, _callback)


class ConditionalCircularTransformSimple(Flow):
    """
    Circular flow based on a mixture of bump functions.

    Here `y` *IS* transformed conditioned on `x`.
    """

    def __init__(self, mu, log_sigma, log_weight, log_eps):
        super().__init__()
        self._mu = mu
        self._log_sigma = log_sigma
        self._log_weight = log_weight
        self._log_eps = log_eps

    def _params(self, x, y):
        n_batch = x.shape[0]

        mu = self._mu(x).view(n_batch, -1, y.shape[-1])
        mu = 0.5 * torch.sin(mu) + 0.5

        log_sigma = self._log_sigma(x).view(n_batch, -1, y.shape[-1])
        sigma = 1.0 + torch.exp(log_sigma)

        log_weight = self._log_weight(x).view(n_batch, -1, y.shape[-1])
        weight = torch.softmax(log_weight, 1)

        log_eps = self._log_eps(x).view(n_batch, y.shape[-1])
        eps = torch.sigmoid(log_eps)

        assert all(p.shape[1] == mu.shape[1] for p in [log_sigma, log_weight])

        return mu, sigma, weight, eps

    def _forward(self, x, y, *args, **kwargs):
        mu, sigma, weight, eps = self._params(x, y)
        cdf, pdf = _cdf_transform(y, mu, sigma, weight, eps=eps)
        cdf = cdf.view(*y.shape)
        dlogp = pdf.sum(dim=-1, keepdim=True)
        return cdf, dlogp

    def _inverse(self, x, y, *args, **kwargs):
        mu, sigma, weight, eps = self._params(x, y)

        def _callback(y):
            cdf, pdf = _cdf_transform(y, mu, sigma, weight, eps=eps)
            cdf = cdf.view(*y.shape)
            dlogp = pdf.sum(dim=-1, keepdim=True)
            return cdf, dlogp

        return _bisect(y, _callback)
