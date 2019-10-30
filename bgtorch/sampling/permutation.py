__author__ = 'noe'

import numpy as np
from scipy.optimize import linear_sum_assignment

from deep_boltzmann.util import ensure_traj, distance_matrix_squared


class HungarianMapper:
    def __init__(self, xref, dim=2, identical_particles=None):
        """ Permutes identical particles to minimize distance to reference structure.

        For a given structure or set of structures finds the permutation of identical particles
        that minimizes the mean square distance to a given reference structure. The optimization
        is done by solving the linear sum assignment problem with the Hungarian algorithm.

        Parameters
        ----------
        xref : array
            reference structure
        dim : int
            number of dimensions of particle system to define relation between vector position and
            particle index. If dim=2, coordinate vectors are [x1, y1, x2, y2, ...].
        indentical_particles : None or array
            indices of particles subject to permutation. If None, all particles are used

        """
        self.xref = xref
        self.dim = dim
        if identical_particles is None:
            identical_particles = np.arange(xref.size)
        self.identical_particles = identical_particles
        self.ip_indices = np.concatenate([dim * self.identical_particles + i for i in range(dim)])
        self.ip_indices.sort()

    def map(self, X):
        """ Maps X (configuration or trajectory) to reference structure by permuting identical particles """
        X = ensure_traj(X)
        Y = X.copy()
        C = distance_matrix_squared(np.tile(self.xref[:, self.ip_indices], (X.shape[0], 1)), X[:, self.ip_indices])

        for i in range(C.shape[0]):  # for each configuration
            _, col_assignment = linear_sum_assignment(C[i])
            assignment_components = [self.dim*col_assignment+i for i in range(self.dim)]
            col_assignment = np.vstack(assignment_components).T.flatten()
            Y[i, self.ip_indices] = X[i, self.ip_indices[col_assignment]]
        return Y

    def is_permuted(self, X):
        """ Returns True for permuted configurations """
        X = ensure_traj(X)
        C = distance_matrix_squared(np.tile(self.xref[:, self.ip_indices], (X.shape[0], 1)), X[:, self.ip_indices])
        isP = np.zeros(X.shape[0], dtype=bool)

        for i in range(C.shape[0]):  # for each configuration
            _, col_assignment = linear_sum_assignment(C[i])
            assignment_components = [self.dim*col_assignment+i for i in range(self.dim)]
            col_assignment = np.vstack(assignment_components).T.flatten()
            if not np.all(col_assignment == np.arange(col_assignment.size)):
                isP[i] = True
        return isP
