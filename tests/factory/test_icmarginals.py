
import pytest
import torch

from bgflow import (
    GlobalInternalCoordinateTransformation,
    InternalCoordinateMarginals, BONDS, ANGLES,
    ShapeDictionary, TensorInfo
)


@pytest.mark.parametrize("with_data", [True, False])
def test_icmarginals_inform_api(tmpdir, ctx, with_data):
    """API test"""
    bgmol = pytest.importorskip("bgmol")
    dataset = bgmol.datasets.Ala2Implicit1000Test(
        root=tmpdir,
        download=True,
        read=True
    )
    coordinate_transform = GlobalInternalCoordinateTransformation(
        bgmol.systems.ala2.DEFAULT_GLOBAL_Z_MATRIX
    )
    current_dims = ShapeDictionary()
    current_dims[BONDS] = (coordinate_transform.dim_bonds - dataset.system.system.getNumConstraints(), )
    current_dims[ANGLES] = (coordinate_transform.dim_angles, )
    marginals = InternalCoordinateMarginals(current_dims, ctx)
    if with_data:
        constrained_indices, _ = bgmol.bond_constraints(dataset.system.system, coordinate_transform)
        marginals.inform_with_data(
            torch.tensor(dataset.xyz, **ctx), coordinate_transform,
            constrained_bond_indices=constrained_indices
        )
    else:
        marginals.inform_with_force_field(
            dataset.system.system, coordinate_transform, 1000.,
        )
