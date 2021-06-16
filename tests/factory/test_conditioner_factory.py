
import pytest
import torch
from bgflow import (
    make_conditioners, ShapeDictionary,
    BONDS, FIXED, ANGLES, TORSIONS,
    ConditionalSplineTransformer, AffineTransformer
)


@pytest.mark.parametrize(
    "transformer_type",
    [
        ConditionalSplineTransformer,
        AffineTransformer,
        # TODO: MixtureCDFTransformer
    ]
)
def test_conditioner_factory_input_dim(transformer_type, crd_trafo):
    torch.manual_seed(10981)

    crd_transform = crd_trafo
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    # check input dimensions:
    conditioners = make_conditioners(transformer_type, (BONDS,), (FIXED,), shape_info, hidden=(128, 128))
    for conditioner in conditioners.values():
        assert conditioner._layers[0].weight.shape == (128, shape_info[FIXED][0])

    # check input dimensions of wrapped:
    conditioners = make_conditioners(transformer_type, (BONDS,), (ANGLES,TORSIONS), shape_info, hidden=(128, 128))
    for conditioner in conditioners.values():
        assert conditioner.net._layers[0].weight.shape == (128, shape_info[ANGLES][0] + 2 * shape_info[TORSIONS][0])

    # check periodicity
    for conditioner in conditioners.values():
        for p in conditioner.parameters():
            p.data = torch.randn_like(p.data)
        # check torsions periodic
        low = conditioner(torch.zeros(shape_info[ANGLES][0] + shape_info[TORSIONS][0]))
        x = torch.cat([torch.zeros(shape_info[ANGLES][0]), torch.ones(shape_info[TORSIONS][0])])
        high = conditioner(x)
        assert torch.allclose(low, high, atol=5e-4)
        # check angles not periodic
        x[0] = 1.0
        high = conditioner(x)
        assert not torch.allclose(low, high, atol=5e-2)


def test_conditioner_factory_spline(crd_trafo):
    crd_transform = crd_trafo
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    # non-periodic
    conditioners = make_conditioners(
        ConditionalSplineTransformer, (BONDS,), (ANGLES,), shape_info
    )
    assert (
            conditioners["params_net"]._layers[-1].bias.shape
            == ((3 * 8 + 1)*shape_info[BONDS][0], )
    )
    # periodic
    conditioners = make_conditioners(
        ConditionalSplineTransformer, (TORSIONS,), (ANGLES,), shape_info
    )
    assert (
            conditioners["params_net"]._layers[-1].bias.shape
            == ((3 * 8)*shape_info[TORSIONS][0], )
    )
    # mixed
    conditioners = make_conditioners(
        ConditionalSplineTransformer, (BONDS, TORSIONS), (ANGLES, FIXED), shape_info
    )
    assert (
            conditioners["params_net"]._layers[-1].bias.shape
            == ((3 * 8)*(shape_info[BONDS][0] + shape_info[TORSIONS][0]) + shape_info[BONDS][0], )
    )


# TODO: fix this
@pytest.mark.skip()
@pytest.mark.parametrize("sin", [torch.sin])#, nn.dense.Sin()])
def test_conditioner_factory_sirens(crd_trafo, sin):
    crd_transform = crd_trafo
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    # non-periodic
    conditioners = make_conditioners(
        ConditionalSplineTransformer, (BONDS,), (ANGLES,), shape_info, activation=sin
    )
    for c in conditioners.values():
        assert isinstance(c, SirenDenseNet)
