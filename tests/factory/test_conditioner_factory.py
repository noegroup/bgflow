import numpy as np
import pytest
import torch
from bgflow import (
    make_conditioners, ShapeDictionary,
    BONDS, FIXED, ANGLES, TORSIONS,
    ConditionalSplineTransformer, AffineTransformer
)
from bgflow.nn.periodic import WrapDistances

try:
    import nequip
    GNN_LIBRARIES_IMPORTED = True
except ImportError:
    GNN_LIBRARIES_IMPORTED = False

pytestmark = pytest.mark.skipif(not GNN_LIBRARIES_IMPORTED, reason="tests require nequip library")


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


@pytest.mark.parametrize("conditioner", ["allegro", "nequip", "wrapdistances"])
@pytest.mark.parametrize("use_checkpointing", ["True", "False"])
@pytest.mark.parametrize("attention_level", ["MHA", "Transformer", None])
def test_GNN(ala2, crd_trafo_unwhitened, conditioner, use_checkpointing, attention_level):

    x = torch.tensor(ala2.xyz)[0]
    R = torch.Tensor(np.linalg.qr(np.random.normal(size=(3, 3)))[0]).double()
    x_rotated = x@R
    x = x.unsqueeze(dim = 0)
    x_rotated = x_rotated.unsqueeze(dim = 0)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_trafo_unwhitened)

    if conditioner == "allegro":
        pytest.importorskip("allegro")
        from bgflow.factory.GNN_factory import allegro_hparams as hparams, make_allegro_config_dict as make_config_dict

    if conditioner == "nequip":
        from bgflow.factory.GNN_factory import nequip_hparams as hparams, make_nequip_config_dict as make_config_dict

    if conditioner in ["allegro", "nequip"]:
        ### gather some data to initialize the RBF to be normalized
        distances_net = WrapDistances(torch.nn.Identity())
        fixed = crd_trafo_unwhitened.forward(torch.Tensor(ala2.xyz.reshape(ala2.xyz.shape[0], -1)))[3]
        RBF_distances = distances_net(fixed.detach()).flatten().cpu()
        RBF_distances = RBF_distances[RBF_distances<=hparams["r_max"]]

        avg_num_neighbors = ((len(RBF_distances)/(ala2.xyz.shape[0]*fixed.shape[1]/3)))*2
        layers = make_config_dict(**hparams)
        layers["radial_basis"][1]["basis_kwargs"]["data"] = RBF_distances

        if conditioner == "allegro":
            layers["allegro"][1]["avg_num_neighbors"]= avg_num_neighbors
        if conditioner == "nequip":
            for i in range(hparams["num_interaction_blocks"]):
                layers[f"convnet_{i}"][1]["convolution_kwargs"]["avg_num_neighbors"] = avg_num_neighbors

        from nequip.nn import SequentialGraphNetwork
        # the layers object is a config dict, that contains all we need to construct a nequip GNN
        GNN = SequentialGraphNetwork.from_parameters(shared_params=None, layers=layers)

        from bgflow.factory.GNN_factory import NequipWrapper
        # the GNN has to be wrapped to return a Tensor, not a nequip "AtomicDataDict"
        GNN_feature_extractor = NequipWrapper(GNN, cutoff = hparams["r_max"])



        kwargs = {
            "r_max": hparams["r_max"],
            "GNN": GNN_feature_extractor,
            "use_checkpointing": use_checkpointing,
            "GNN_output_dim": hparams["GNN_feature_dim"]*(shape_info[FIXED][0]//3),
            "attention_units": (shape_info[FIXED][0]//3),
            "attention_level": attention_level
        }
    if conditioner in ["allegro", "nequip"]:
        conditioners = make_conditioners(
            ConditionalSplineTransformer, (BONDS,), (TORSIONS, FIXED), shape_info,
            conditioner_type="GNN",
            **kwargs
        )
    elif conditioner == "wrapdistances":
        from bgflow.factory.GNN_factory import WrapDistancesGNN
        num_basis = 32
        kwargs = {
            "r_max": 1.2,
            "GNN": WrapDistancesGNN,
            "GNN_kwargs": {"num_basis": num_basis,
                           "r_max": 15,
                           "env_p": 48
                           },
            "use_checkpointing": use_checkpointing,  # False,
            "GNN_output_dim": ((shape_info[FIXED][0]//3 )** 2 - shape_info[FIXED][0]//3) // 2 * num_basis,
            "attention_units": ((shape_info[FIXED][0]//3)** 2 - shape_info[FIXED][0]//3) // 2,
            "attention_level": attention_level
        }
        conditioners = make_conditioners(
            ConditionalSplineTransformer, (BONDS,), (TORSIONS, FIXED), shape_info,
            conditioner_type="GNN",
            **kwargs
        )

    device = torch.device("cpu")
    dtype = torch.float32

    # a context tensor to send data to the right device and dtype via '.to(ctx)'
    ctx = {"device": device, "dtype": dtype}
    x = x.to(**ctx)
    x_rotated = x_rotated.to(**ctx)


    bx, ax, tx, fx, _ = crd_trafo_unwhitened(x)
    br, ar, tr, fr, _ = crd_trafo_unwhitened(x_rotated)
    fx_xyz = fx.view(-1,3)
    fr_xyz = fr.view(-1,3)

    onx = torch.cat((tx, fx), dim = -1)
    onr = torch.cat((tr, fr), dim = -1)
    params_x = conditioners["params_net"](onx)
    params_r = conditioners["params_net"](onr)
    ### parameters of an untrained GNN can be quite large, so for GNNs, absolute errors are fine.
    if conditioner in ["allegro", "nequip"]:
        assert torch.allclose(params_x, params_r, rtol=1e-03, atol=1e-3)
    elif conditioner == "wrapdistances":
        assert torch.allclose(params_x, params_r, rtol=1e-03, atol=1e-3)


