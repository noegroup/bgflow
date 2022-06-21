
import pytest
import torch
from bgflow import AffineTransformer, DenseNet


class ShiftModule(torch.nn.Module):
    def forward(self, x):
        return torch.ones_like(x)


@pytest.mark.parametrize("is_circular", [True, False])
@pytest.mark.parametrize("use_scale_transform", [True, False])
def test_affine(is_circular, use_scale_transform):

    if use_scale_transform:
        scale = DenseNet([2, 2])
    else:
        scale = None

    if use_scale_transform and is_circular:
        with pytest.raises(ValueError):
            trafo = AffineTransformer(
                shift_transformation=ShiftModule(),
                scale_transformation=scale,
                is_circular=is_circular
            )

    else:
        trafo = AffineTransformer(
            shift_transformation=ShiftModule(),
            scale_transformation=scale,
            is_circular=is_circular
        )
        x = torch.rand(100, 2)
        y = torch.rand(100, 2)
        y2, dlogp = trafo.forward(x, y)
        assert y2.shape == y.shape
        if is_circular:
            assert (y2 < 1).all()
        else:
            assert (y2 > 1).any()


def test_volume_preserving(ctx):
    trafo = AffineTransformer(
        shift_transformation=DenseNet([2, 2]),
        scale_transformation=DenseNet([2, 2]),
        preserve_volume=True
    ).to(**ctx)
    x = torch.rand(100, 2, **ctx)
    y = torch.rand(100, 2, **ctx)

    y2, dlogp = trafo.forward(x, y)
    assert torch.allclose(dlogp, torch.zeros(100, 1, **ctx), atol=1e-6)

    y2, dlogp = trafo.forward(x, y, inverse=True)
    assert torch.allclose(dlogp, torch.zeros(100, 1, **ctx), atol=1e-6)

    y2, dlogp = trafo.forward(x, y, target_dlogp=torch.ones(2, **ctx))
    assert torch.allclose(dlogp, torch.ones(100, 1, **ctx), atol=1e-6)

    y2, dlogp = trafo.forward(x, y, target_dlogp=torch.ones(2, **ctx), inverse=True)
    assert torch.allclose(dlogp, torch.ones(100, 1, **ctx), atol=1e-6)

