import numpy as np
import pytest
import torch
from bgflow import (
    make_conditioners, ShapeDictionary,
    BONDS, FIXED, ANGLES, TORSIONS,
    ConditionalSplineTransformer, AffineTransformer, RelativeInternalCoordinateTransformation
)
import scipy


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

def test_condition_on_cartesian(ala2, crd_trafo, ctx):
    x = torch.tensor(ala2.xyz)[0]
    R = torch.Tensor(scipy.spatial.transform.Rotation.random().as_matrix()).double()
    #R = torch.eye(3)
    #R_batch = torch.repeat_interleave(torch.repeat_interleave(R.unsqueeze(0),22,0).unsqueeze(0),100,0)



    #x_rotated = torch.einsum("BNIJ,BNJ->BNI", R_batch, x)
    x_rotated = torch.zeros_like(x)
    for j in range(22):
        x_rotated[j] = R@(x[j])
    #x_rotated = R@x
    #pytest.set_trace()
    x_xyz = x.view(-1,3)
    x_rotated_xyz = x_rotated.view(-1,3)

    x = x.unsqueeze(dim = 0)
    x_rotated = x_rotated.unsqueeze(dim = 0)
    #pytest.set_trace()

    def crd_trafo(ala2, ctx):
        z_matrix = ala2.system.z_matrix
        fixed_atoms = ala2.system.rigid_block
        crd_transform = RelativeInternalCoordinateTransformation(z_matrix, fixed_atoms)
        return crd_transform

    crd_transform = crd_trafo(ala2, ctx)
    #pytest.set_trace()

    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    #
    kwargs ={"N": 16, "c": 15, "p": 6}
    conditioners = make_conditioners(
        ConditionalSplineTransformer, (BONDS,), (TORSIONS, FIXED), shape_info,
        conditioner_type="wrapdistances",
        **kwargs
    )
    #device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.device("cpu")
    dtype = torch.float32

    # a context tensor to send data to the right device and dtype via '.to(ctx)'
    ctx = {"device": device, "dtype": dtype}
    #pytest.set_trace()
    x = x.to(**ctx)
    x_rotated = x_rotated.to(**ctx)
    #pytest.set_trace()

    bx, ax, tx, fx, _ = crd_transform(x)
    br, ar, tr, fr, _ = crd_transform(x_rotated)
    fx_xyz = fx.view(-1,3)
    fr_xyz = fr.view(-1,3)
    #pytest.set_trace()

    torch.norm(fx_xyz[0] - fx_xyz[1])
    torch.norm(fr_xyz[0] - fr_xyz[1])
    onx = torch.cat((tx, fx), dim = -1)
    onr = torch.cat((tr, fr), dim = -1)
    #pytest.set_trace()
    params_x = conditioners["params_net"](onx)
    params_r = conditioners["params_net"](onr)
    #pytest.set_trace()
    assert torch.allclose(params_x, params_r, atol=5e-4)


