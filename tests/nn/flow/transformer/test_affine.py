
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

