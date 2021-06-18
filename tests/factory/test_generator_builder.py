
import pytest
import torch
import bgflow as bg
from bgflow.nn.flow.crd_transform.ic import (
    MixedCoordinateTransformation,
    GlobalInternalCoordinateTransformation,
    RelativeInternalCoordinateTransformation
)
from bgflow import (
    BoltzmannGeneratorBuilder, BONDS, ANGLES, TORSIONS, FIXED, AUGMENTED, TensorInfo,
    ShapeDictionary, InternalCoordinateMarginals
)


def test_builder_api(ala2, ctx):
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
    energy = generator.energy(samples)
    generator.kldiv(10)


@pytest.mark.skip() # TODO
def test_builder_augmentation_and_global(ala2, ctx):
    z_matrix, _ = ZMatrixFactory(ala2.system.mdtraj_topology).build_naive()
    crd_transform = GlobalInternalCoordinateTransformation(z_matrix)
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


def test_builder_custom(ala2, ctx):
    """This would be cool. Three possible implementations:
    - keep track of indices inside the factory. since layers are added in order, this is no problem.
    - allow selecting tensors by name in the logistical flows (coupling, split, ...). Give the
      `Flow` class two optional attributes names_in, names_out; then either
        - (a) just set a custom tensor._name/tensor._info attribute - no idea where this may break in the future
        - (b) give the flow

    Is it clever to have the crd_transform as the central piece to all this?
    Or can we somehow make it more general? Maybe start from the prior.
    """
    pytest.skip()
    z_matrix = ala2.system.z_matrix
    fixed_atoms = ala2.system.rigid_block
    crd_transform = MixedCoordinateTransformation(torch.tensor(ala2.xyz, **ctx), z_matrix, fixed_atoms)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform)
    builder = BoltzmannGeneratorBuilder(shape_info)
    builder.prior_type[BONDS] = bg.NormalDistribution
    builder.transformer_type[BONDS] = bg.AffineTransformer
    builder.add_condition(BONDS, on=ANGLES)
    BONDS1 = TensorInfo("BONDS1")
    BONDS2 = TensorInfo("BONDS2")
    builder.add_split(BONDS, into=(BONDS1, BONDS2), sizes_or_indices=(14, 3))
    builder.add_condition(BONDS1, on=BONDS2)
    builder.add_merge((BONDS1, BONDS2), to=BONDS)

    marginal_cdf = MarginalICTransform()
    marginal_cdf = InformedMarginalICTransform(crd_transform, system, topology)
    marginal_cdf.scan_data(data)
    marginal_cdf.scan_torsions(temperature=300.)
    builder.add_map_to_ic_domains(marginal_cdf)
    builder.add_map_to_cartesian(crd_transform)
    builder.build_generator()


def test_builder_split_merge(ctx):
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


# TODO
@pytest.mark.skip()
def test_builder_bond_constraints(ala2, ctx):
    z_matrix, _ = ZMatrixFactory(ala2.system.mdtraj_topology).build_naive()
    crd_transform = GlobalInternalCoordinateTransformation(z_matrix)
    shape_info = ShapeDictionary.from_coordinate_transform(crd_transform, dim_augmented=10)

    builder = BoltzmannGeneratorBuilder(shape_info, target=ala2.system.energy_model, **ctx)
    assert builder.prior_dims[BONDS] == (21, )
    constrained_bonds = builder.set_constrained_bonds(ala2.system.system, crd_transform)
    assert builder.prior_dims[BONDS] == (9, )
    assert builder.current_dims[BONDS] == (9,)
    assert constrained_bonds not in builder.prior_dims
    builder.add_condition(BONDS, on=(ANGLES, TORSIONS))
    builder.add_map_to_ic_domains()
    builder.add_map_to_cartesian(crd_transform)
    generator = builder.build_generator()
    # play forward and backward
    samples = generator.sample(2)
    assert len(samples) == 2
    generator.energy(*samples)
    generator.kldiv(10)
