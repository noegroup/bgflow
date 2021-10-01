
import torch
from torch.distributions import SigmoidTransform, AffineTransform, IndependentTransform
from bgflow import TorchTransform, SequentialFlow, BentIdentity


def test_torch_transform(ctx):
    """try using torch.Transform in combination with bgflow.Flow"""
    torch.manual_seed(10)
    x = torch.torch.randn(10, 3, **ctx)
    flow = SequentialFlow([
        TorchTransform(IndependentTransform(SigmoidTransform(), 1)),
        TorchTransform(
                AffineTransform(
                    loc=torch.randn(3, **ctx),
                    scale=2.0+torch.rand(3, **ctx), event_dim=1
                ),
        ),
        BentIdentity(),
        # test the reinterpret_batch_ndims arguments
        TorchTransform(SigmoidTransform(), 1)
    ])
    z, dlogp = flow.forward(x)
    y, neg_dlogp = flow.forward(z, inverse=True)
    tol = 1e-7 if ctx["dtype"] is torch.float64 else 1e-5
    assert torch.allclose(x, y, atol=tol)
    assert torch.allclose(dlogp, -neg_dlogp, atol=tol)
