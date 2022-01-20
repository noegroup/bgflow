
import torch
from bgflow.nn.flow.transformer.gaussian import TruncatedGaussianTransformer
from bgflow.distribution.normal import TruncatedNormalDistribution


def test_constrained_affine_transformer(ctx):
    tol = 5e-4 if ctx["dtype"] == torch.float32 else 5e-7
    mu_net = torch.nn.Linear(3, 2, bias=False)
    sigma_net = torch.nn.Linear(3, 2, bias=False)
    mu_net.weight.data = torch.ones_like(mu_net.weight.data)
    sigma_net.weight.data = torch.ones_like(sigma_net.weight.data)
    constrained = TruncatedGaussianTransformer(mu_net, sigma_net, -5.0, 5.0, -8.0, 0.0).to(**ctx)

    # test if forward and inverse are compatible
    x = torch.ones(1, 3, **ctx)
    y = torch.tensor([[-2.5, 2.5]], **ctx)
    y.requires_grad = True
    out, dlogp = constrained.forward(x, y)
    assert not torch.allclose(dlogp, torch.zeros_like(dlogp))
    assert (out >= torch.tensor(-8.0, **ctx)).all()
    assert (out <= torch.tensor(0.20, **ctx)).all()
    y2, neg_dlogp = constrained.forward(x, out, inverse=True)
    assert torch.allclose(y, y2, atol=tol)
    assert torch.allclose(dlogp + neg_dlogp, torch.zeros_like(dlogp), atol=tol)

    # test if the log det agrees with the log prob of a truncated normal distribution
    mu = torch.einsum("ij,...j->...i", mu_net.weight.data, x)
    _, logsigma = constrained._get_mu_and_log_sigma(x, y)  # not reiterating the tanh stuff
    sigma = torch.exp(logsigma)
    trunc_gaussian = TruncatedNormalDistribution(mu, sigma, torch.tensor(-5, **ctx), torch.tensor(5, **ctx))
    log_prob = trunc_gaussian.log_prob(y)
    log_scale = torch.log(torch.tensor(8., **ctx))
    assert torch.allclose(dlogp, (log_prob + log_scale).sum(dim=-1, keepdim=True), atol=tol)

    # try backward pass and assert reasonable gradients
    y2.sum().backward(create_graph=True)
    neg_dlogp.backward()
    for tensor in [y, mu_net.weight]:
        assert (tensor.grad > -1e6).all()
        assert (tensor.grad < 1e6).all()
