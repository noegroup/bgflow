__author__ = 'solsson'

import numpy as np


def save_latent_samples_as_trajectory(samples, mdtraj_topology, filename=None, topology_fn=None, return_openmm_traj=True):
    """
    Save Boltzmann Generator samples as a molecular dynamics trajectory.
    `samples`: posterior (Nsamples, n_atoms*n_dim)
    `mdtraj_topology`: an MDTraj Topology object of the molecular system
    `filename=None`: output filename with extension (all MDTraj compatible formats)
    `topology_fn=None`: outputs a PDB-file of the molecular topology for external visualization and analysis.
    """
    import mdtraj as md
    trajectory = md.Trajectory(samples.reshape(-1, mdtraj_topology.n_atoms, 3), mdtraj_topology)
    if isinstance(topology_fn, str):
        trajectory[0].save_pdb(topology_fn)
    if isinstance(filename, str):
        trajectory.save(filename)
    if return_openmm_traj:
        return trajectory


class NumpyReporter(object):
    def __init__(self, reportInterval, enforcePeriodicBox=True):
        self._coords = []
        self._reportInterval = reportInterval
        self.enforcePeriodicBox = enforcePeriodicBox

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, True, False, False, False, self.enforcePeriodicBox)

    def report(self, simulation, state):
        self._coords.append(state.getPositions(asNumpy=True).ravel())

    def get_coordinates(self, superimpose=None):
        """
            return saved coordinates as numpy array
            `superimpose`: openmm/mdtraj topology, will superimpose on first frame
        """
        import mdtraj as md
        try:
            from openmm.app.topology import Topology as _Topology
        except ImportError: # fall back to older version < 7.6
            from simtk.openmm.app.topology import Topology as _Topology
        if superimpose is None:
            return np.array(self._coords)
        elif isinstance(superimpose, _Topology):
            trajectory = md.Trajectory(np.array(self._coords).reshape(-1, superimpose.getNumAtoms(), 3), 
                md.Topology().from_openmm(superimpose))
        else:
            trajectory = md.Trajectory(np.array(self._coords).reshape(-1, superimpose.n_atoms, 3), 
                superimpose)        
        
        trajectory.superpose(trajectory[0])
        return trajectory.xyz.reshape(-1, superimpose.n_atoms * 3)
