
import pytest
import numpy as np
import torch
import bgflow as bg
from bgflow.nn.flow.crd_transform.ic import (
    MixedCoordinateTransformation,
    GlobalInternalCoordinateTransformation,
    RelativeInternalCoordinateTransformation
)
from bgflow import (
    BoltzmannGeneratorBuilder, BONDS, ANGLES, TORSIONS, FIXED, AUGMENTED, TensorInfo,
    ShapeDictionary, InternalCoordinateMarginals, CouplingFlow, AffineTransformer, DenseNet,
    CDFTransform, TruncatedNormalDistribution, ProductDistribution, NormalDistribution
)

pytestmark = pytest.mark.filterwarnings("ignore:singular ")


def test_builder_api(ala2, ctx):
    pytest.importorskip("nflows")

    z_matrix = ala2.system.z_matrix
    fixed_atoms = ala2.system.rigid_block
    crd_transform = MixedCoordinateTransformation(torch.tensor(ala2.xyz, **ctx), z_matrix, fixed_atoms)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    builder = BoltzmannGeneratorBuilder(shape_info, target=ala2.system.energy_model, **ctx)
    for i in range(4):
        builder.add_condition(TORSIONS, on=FIXED)
        builder.add_condition(FIXED, on=TORSIONS)
    for i in range(2):
        builder.add_condition(BONDS, on=ANGLES)
        builder.add_condition(ANGLES, on=BONDS)
    builder.add_condition(ANGLES, on=(TORSIONS, FIXED))
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS, FIXED))
    builder.add_map_to_ic_domains()
    builder.add_map_to_cartesian(crd_transform)
    generator = builder.build_generator()
    # play forward and backward
    samples = generator.sample(2)
    generator.energy(samples)
    generator.kldiv(10)


def test_builder_augmentation_and_global(ala2, ctx):
    pytest.importorskip("nflows")

    crd_transform = GlobalInternalCoordinateTransformation(ala2.system.global_z_matrix)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform, dim_augmented=10)
    builder = BoltzmannGeneratorBuilder(shape_info, target=ala2.system.energy_model, **ctx)
    for i in range(4):
        builder.add_condition(TORSIONS, on=AUGMENTED)
        builder.add_condition(AUGMENTED, on=TORSIONS)
    for i in range(2):
        builder.add_condition(BONDS, on=ANGLES)
        builder.add_condition(ANGLES, on=BONDS)
    builder.add_condition(ANGLES, on=(TORSIONS, AUGMENTED))
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS, AUGMENTED))
    builder.add_map_to_ic_domains()
    builder.add_map_to_cartesian(crd_transform)
    generator = builder.build_generator()
    # play forward and backward
    samples = generator.sample(2)
    assert len(samples) == 2
    generator.energy(*samples)
    generator.kldiv(10)


def test_builder_add_layer_and_param_groups(ctx):
    shape_info = ShapeDictionary()
    shape_info[BONDS] = (10, )
    shape_info[ANGLES] = (20, )
    builder = BoltzmannGeneratorBuilder(shape_info, **ctx)
    # transform some fields
    builder.add_layer(
        CDFTransform(
            TruncatedNormalDistribution(torch.zeros(10, **ctx), lower_bound=-torch.tensor(np.infty)),
        ),
        what=[BONDS],
        inverse=True,
        param_groups=("group1", )
    )
    # transform all fields
    builder.add_layer(
        CouplingFlow(
            AffineTransformer(
                DenseNet([10, 20]), DenseNet([10, 20])
            )
        ),
        param_groups=("group1", "group2")
    )
    builder.targets[BONDS] = NormalDistribution(10, torch.zeros(10, **ctx))
    builder.targets[ANGLES] = NormalDistribution(20, torch.zeros(20, **ctx))
    generator = builder.build_generator().to(**ctx)
    assert builder.param_groups["group1"] == list(generator.parameters())
    assert builder.param_groups["group2"] == list(generator._flow._blocks[1].parameters())
    generator.sample(10)
    generator.kldiv(10)


def test_builder_split_merge(ctx):
    pytest.importorskip("nflows")
    shape_info = ShapeDictionary()
    shape_info[BONDS] = (10, )
    shape_info[ANGLES] = (20, )
    shape_info[TORSIONS] = (13, )
    builder = BoltzmannGeneratorBuilder(shape_info, **ctx)
    split1 = TensorInfo("SPLIT_1")
    split2 = TensorInfo("SPLIT_2")
    split3 = TensorInfo("SPLIT_3")
    builder.add_split(ANGLES, (split1, split2, split3), (6, 2, 12))
    builder.add_condition(split1, on=split2)
    generator = builder.build_generator(zero_parameters=True, check_target=False)
    samples = generator.sample(11)
    assert len(samples) == 5
    assert all(samples[i].shape == (11,j) for i, j in enumerate([10,6,2,12,13]))

    # check split + add_merge (with string arguments)
    assert builder.layers == []
    s1, split_2, s3 = builder.add_split(ANGLES, (split1, "split2", split3), (6, 2, 12))
    assert s1 == split1
    assert s3 == split3
    assert split_2.name == "split2"
    assert split_2.is_circular == ANGLES.is_circular
    builder.add_condition(split1, on=split_2)
    angles = builder.add_merge((split1, split_2, split3), "angles")
    assert angles.name == "angles"
    assert angles.is_circular == ANGLES.is_circular
    assert list(builder.current_dims) == [BONDS, angles, TORSIONS]
    generator = builder.build_generator(zero_parameters=True, check_target=False)
    samples = generator._prior.sample(11)
    assert all(torch.all(s > torch.zeros_like(s)) for s in samples)
    assert all(torch.all(s < torch.ones_like(s)) for s in samples)
    *output, dlogp = generator._flow.forward(*samples)
    assert all(s.shape == o.shape for s, o in zip(samples, output))
    assert all(torch.allclose(s, o, atol=0.01, rtol=0.0) for s, o in zip(samples, output))


def test_builder_multiple_crd(ala2, ctx):
    bgmol = pytest.importorskip("bgmol")
    pytest.importorskip("nflows")

    # all-atom trafo
    z_matrix, fixed = bgmol.ZMatrixFactory(ala2.system.mdtraj_topology, cartesian=[6, 8, 10, 14, 16]).build_naive()
    crd_transform = RelativeInternalCoordinateTransformation(z_matrix, fixed)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)

    # cg trafo
    cg_top, _ = bgmol.build_fake_topology(5)
    cg_z_matrix, _ = bgmol.ZMatrixFactory(cg_top).build_naive()
    cg_crd_transform = GlobalInternalCoordinateTransformation(cg_z_matrix)
    cg_shape_info = ShapeDictionary.from_coordinate_transform(cg_crd_transform)
    CG_BONDS = cg_shape_info.replace(BONDS, "CG_BONDS")
    CG_ANGLES = cg_shape_info.replace(ANGLES, "CG_ANGLES")
    CG_TORSIONS = cg_shape_info.replace(TORSIONS, "CG_TORSIONS")
    shape_info.update(cg_shape_info)
    del shape_info[FIXED]

    # factory
    #marginals = InternalCoordinateMarginals(builder.current_dims)
    builder = BoltzmannGeneratorBuilder(shape_info, target=ala2.system.energy_model, **ctx)
    for i in range(2):
        builder.add_condition(CG_TORSIONS, on=(CG_ANGLES, CG_BONDS))
        builder.add_condition((CG_ANGLES, CG_BONDS), on=CG_TORSIONS)
    marginals = InternalCoordinateMarginals(builder.current_dims, builder.ctx, bonds=CG_BONDS, angles=CG_ANGLES, torsions=CG_TORSIONS)
    builder.add_map_to_ic_domains(marginals)
    builder.add_map_to_cartesian(cg_crd_transform, bonds=CG_BONDS, angles=CG_ANGLES, torsions=CG_TORSIONS, out=FIXED)
    builder.transformer_type[FIXED] = bg.AffineTransformer
    for i in range(2):
        builder.add_condition(TORSIONS, on=FIXED)
        builder.add_condition(FIXED, on=TORSIONS)
    for i in range(2):
        builder.add_condition(BONDS, on=ANGLES)
        builder.add_condition(ANGLES, on=BONDS)
    builder.add_condition(ANGLES, on=(TORSIONS, FIXED))
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS, FIXED))
    builder.add_map_to_ic_domains()
    builder.add_map_to_cartesian(crd_transform)
    generator = builder.build_generator()
    # play forward and backward
    samples = generator.sample(2)
    generator.energy(samples)
    generator.kldiv(10)


def test_builder_bond_constraints(ala2, ctx):
    # import logging
    # logger = logging.getLogger('bgflow')
    # logger.setLevel(logging.DEBUG)
    # logger.addHandler(
    #     logging.StreamHandler()
    # )
    pytest.importorskip("nflows")
    crd_transform = GlobalInternalCoordinateTransformation(ala2.system.global_z_matrix)
    shape_info = ShapeDictionary.from_coordinate_transform(
        crd_transform,
        dim_augmented=0,
        n_constraints=2,
        remove_origin_and_rotation=True
    )
    builder = BoltzmannGeneratorBuilder(shape_info, target=ala2.system.energy_model, **ctx)
    constrained_bond_indices = [0, 1]
    constrained_bond_lengths = [0.1, 0.1]
    assert builder.current_dims[BONDS] == (19, )
    assert builder.prior_dims[BONDS] == (19, )
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS))
    builder.add_map_to_ic_domains()
    builder.add_merge_constraints(constrained_bond_indices, constrained_bond_lengths)
    assert builder.current_dims[BONDS] == (21, )
    builder.add_map_to_cartesian(crd_transform)
    generator = builder.build_generator()
    # play forward and backward
    samples = generator.sample(2)
    assert samples.shape == (2, 66)
    generator.energy(samples)
    generator.kldiv(10)


def test_constrain_chirality(ala2, ctx):
    bgmol = pytest.importorskip("bgmol")
    top = ala2.system.mdtraj_topology
    zmatrix, _ = bgmol.ZMatrixFactory(top).build_naive()
    crd_transform = GlobalInternalCoordinateTransformation(zmatrix)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    builder = BoltzmannGeneratorBuilder(shape_info, target=ala2.system.energy_model, **ctx)
    chiral_torsions = bgmol.is_chiral_torsion(crd_transform.torsion_indices, top)
    builder.add_constrain_chirality(chiral_torsions)
    builder.add_map_to_ic_domains()
    builder.add_map_to_cartesian(crd_transform)
    generator = builder.build_generator()
    # play forward and backward
    samples = generator.sample(20)
    b, a, t, *_ = crd_transform.forward(samples)
    assert torch.all(t[:, chiral_torsions] >= 0.5)
    assert torch.all(t[:, chiral_torsions] <= 1.0)
