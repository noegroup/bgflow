
import os
from types import SimpleNamespace
import pytest
import numpy as np
import torch
from bgflow import MixedCoordinateTransformation, OpenMMBridge, OpenMMEnergy, RelativeInternalCoordinateTransformation


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA not available."
            )
        )
    ]
)
def device(request):
    """Run a test case for all available devices."""
    return torch.device(request.param)


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request, device):
    """Run a test case in single and double precision."""
    return request.param


@pytest.fixture()
def ctx(dtype, device):
    return {"dtype": dtype, "device": device}


@pytest.fixture(params=[torch.enable_grad, torch.no_grad])
def with_grad_and_no_grad(request):
    """Run a test with and without torch grad enabled"""
    with request.param():
        yield


@pytest.fixture(scope="session")
def ala2():
    """Mock bgmol dataset."""
    mm = pytest.importorskip("simtk.openmm")
    md = pytest.importorskip("mdtraj")
    pdb = mm.app.PDBFile(os.path.join(os.path.dirname(__file__), "data/alanine-dipeptide-nowater.pdb"))
    system = SimpleNamespace()
    system.topology = pdb.getTopology()
    system.mdtraj_topology = md.Topology.from_openmm(system.topology)
    system.system = mm.app.ForceField("amber99sbildn.xml").createSystem(
        pdb.getTopology(),
        removeCMMotion=True,
        nonbondedMethod=mm.app.NoCutoff,
        constraints=mm.app.HBonds,
        rigidWater=True
    )
    system.energy_model = OpenMMEnergy(
        bridge=OpenMMBridge(
            system.system,
            mm.LangevinIntegrator(300, 1, 0.001),
            n_workers=1
        )
    )
    system.positions = pdb.getPositions()
    system.rigid_block = np.array([6, 8, 9, 10, 14])
    system.z_matrix = np.array([
            [0, 1, 4, 6],
            [1, 4, 6, 8],
            [2, 1, 4, 0],
            [3, 1, 4, 0],
            [4, 6, 8, 14],
            [5, 4, 6, 8],
            [7, 6, 8, 4],
            [11, 10, 8, 6],
            [12, 10, 8, 11],
            [13, 10, 8, 11],
            [15, 14, 8, 16],
            [16, 14, 8, 6],
            [17, 16, 14, 15],
            [18, 16, 14, 8],
            [19, 18, 16, 14],
            [20, 18, 16, 19],
            [21, 18, 16, 19]
        ])
    system.global_z_matrix = np.row_stack([
        system.z_matrix,
        np.array([
            [9, 8, 6, 14],
            [10, 8, 14, 6],
            [6, 8, 14, -1],
            [8, 14, -1, -1],
            [14, -1, -1, -1]
        ])
    ])
    dataset = SimpleNamespace()
    dataset.system = system
    # super-short simulation
    xyz = []
    simulation = mm.app.Simulation(system.topology, system.system, mm.LangevinIntegrator(300,1,0.001))
    simulation.context.setPositions(system.positions)
    for i in range(100):
        simulation.step(10)
        pos = simulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        xyz.append(pos._value)
    dataset.xyz = np.stack(xyz, axis=0)
    return dataset


@pytest.fixture()
def crd_trafo(ala2, ctx):
    z_matrix = ala2.system.z_matrix
    fixed_atoms = ala2.system.rigid_block
    crd_transform = MixedCoordinateTransformation(torch.tensor(ala2.xyz, **ctx), z_matrix, fixed_atoms)
    return crd_transform


@pytest.fixture()
def crd_trafo_unwhitened(ala2, ctx):
    z_matrix = ala2.system.z_matrix
    fixed_atoms = ala2.system.rigid_block
    crd_transform = RelativeInternalCoordinateTransformation(z_matrix, fixed_atoms)
    return crd_transform

