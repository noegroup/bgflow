__author__ = "noe"

import numpy as np


class UmbrellaModel(object):
    def __init__(self, energy_function, rc_function, k_umbrella, m_umbrella):
        """ Umbrella Energy Model

        Parameters
        ----------
        energy_function : Energy function
            callable when evaluated on x yields energy u(x)
        k_umbrella : float
            force constant of umbrella
        m_umbrella : float
            mean position of umbrella
        rc_function : function
            function to compute reaction coordinate value

        """
        self.energy_function = energy_function
        self.rc_function = rc_function
        self.k_umbrella = k_umbrella
        self.m_umbrella = m_umbrella

    @classmethod
    def from_dict(cls, D):
        k_umbrella = D["k_umbrella"]
        m_umbrella = D["m_umbrella"]
        um = cls(None, None, k_umbrella, m_umbrella)
        if "rc_traj" in D:
            um.rc_traj = D["rc_traj"]
        return um

    def to_dict(self):
        D = {}
        D["k_umbrella"] = self.k_umbrella
        D["m_umbrella"] = self.m_umbrella
        if hasattr(self, "rc_traj"):
            D["rc_traj"] = self.rc_traj
        return D

    def bias_energy(self, rc):
        return self.k_umbrella * (rc - self.m_umbrella) ** 2

    def energy(self, x):
        rc = self.rc_function(x)
        return self.energy_function(x) + self.bias_energy(rc)


class UmbrellaSampling(object):
    def __init__(
        self,
        energy_function,
        sampler,
        rc_function,
        x0,
        n_umbrella,
        k,
        m_min,
        m_max,
        forward_backward=True,
    ):
        """ Umbrella Sampling

        Parameters
        ----------
        energy_function : Energy function
            Callable when evaluated on x yields u(x)
        sample : Sampler
            Object with a run(nsteps) and reset(x) method
        x0 : [array]
            Initial configuration

        """
        self.energy_function = energy_function
        self.sampler = sampler
        self.rc_function = rc_function
        self.x0 = x0
        self.forward_backward = forward_backward

        d = (m_max - m_min) / (n_umbrella - 1)
        m_umbrella = [m_min + i * d for i in range(n_umbrella)]
        if forward_backward:
            m_umbrella += m_umbrella[::-1]
        self.umbrellas = [
            UmbrellaModel(energy_function, rc_function, k, m) for m in m_umbrella
        ]

    @classmethod
    def load(cls, filename):
        """ Loads parameters into model. The resulting model is just a data container.
        """
        from deep_boltzmann.util import load_obj

        D = load_obj(filename)
        umbrellas = [UmbrellaModel.from_dict(u) for u in D["umbrellas"]]
        us = cls(
            None,
            None,
            None,
            None,
            len(umbrellas),
            umbrellas[0].k_umbrella,
            umbrellas[0].m_umbrella,
            umbrellas[-1].m_umbrella,
            forward_backward=D["forward_backward"],
        )
        us.umbrellas = umbrellas
        if "rc_discretization" in D:
            us.rc_discretization = D["rc_discretization"]
            us.rc_free_energies = D["rc_free_energies"]
        return us

    def save(self, filename):
        from deep_boltzmann.util import save_obj

        D = {}
        D["umbrellas"] = [u.to_dict() for u in self.umbrellas]
        D["forward_backward"] = self.forward_backward
        if hasattr(self, "rc_discretization"):
            D["rc_discretization"] = self.rc_discretization
            D["rc_free_energies"] = self.rc_free_energies
        save_obj(D, filename)

    def run(self, nsteps=10000, verbose=True):
        xstart = self.x0
        for i in range(len(self.umbrellas)):
            if verbose:
                print("Umbrella", i + 1, "/", len(self.umbrellas))
            self.sampler.energy_function = self.umbrellas[
                i
            ]  # this is a hot fix - find a better way to do this.
            self.sampler.reset(xstart)
            self.sampler.run(nsteps=nsteps)
            traj = self.sampler.traj
            rc_traj = self.rc_function(traj)
            self.umbrellas[i].rc_traj = rc_traj
            xstart = np.array([traj[-1]])

    @property
    def rc_trajs(self):
        return [u.rc_traj for u in self.umbrellas]

    @property
    def bias_energies(self):
        return [u.bias_energy(u.rc_traj) for u in self.umbrellas]

    @property
    def umbrella_positions(self):
        return np.array([u.m_umbrella for u in self.umbrellas])

    def umbrella_free_energies(self):
        from deep_boltzmann.sampling.analysis import bar

        free_energies = [0]
        for i in range(len(self.umbrellas) - 1):
            k_umbrella = self.umbrellas[i].k_umbrella
            # free energy differences between umbrellas
            sampled_a_ua = (
                k_umbrella
                * (self.umbrellas[i].rc_traj - self.umbrellas[i].m_umbrella) ** 2
            )
            sampled_a_ub = (
                k_umbrella
                * (self.umbrellas[i].rc_traj - self.umbrellas[i + 1].m_umbrella) ** 2
            )
            sampled_b_ua = (
                k_umbrella
                * (self.umbrellas[i + 1].rc_traj - self.umbrellas[i].m_umbrella) ** 2
            )
            sampled_b_ub = (
                k_umbrella
                * (self.umbrellas[i + 1].rc_traj - self.umbrellas[i + 1].m_umbrella)
                ** 2
            )
            Delta_F = bar(sampled_a_ub - sampled_a_ua, sampled_b_ua - sampled_b_ub)
            free_energies.append(free_energies[-1] + Delta_F)
        return np.array(free_energies)

    def mbar(self, rc_min=None, rc_max=None, rc_bins=50):
        """ Estimates free energy along reaction coordinate with binless WHAM / MBAR.

        Parameters
        ----------
        rc_min : float or None
            Minimum bin position. If None, the minimum RC value will be used.
        rc_max : float or None
            Maximum bin position. If None, the maximum RC value will be used.
        rc_bins : int or None
            Number of bins

        Returns
        -------
        bins : array
            Bin positions
        F : array
            Free energy / -log(p) for all bins

        """
        import pyemma

        if rc_min is None:
            rc_min = np.concatenate(self.rc_trajs).min()
        if rc_max is None:
            rc_max = np.concatenate(self.rc_trajs).max()
        xgrid = np.linspace(rc_min, rc_max, rc_bins)
        dtrajs = [np.digitize(rc_traj, xgrid) for rc_traj in self.rc_trajs]
        umbrella_centers = [u.m_umbrella for u in self.umbrellas]
        umbrella_force_constants = ([2.0 * u.k_umbrella for u in self.umbrellas],)
        mbar_obj = pyemma.thermo.estimate_umbrella_sampling(
            self.rc_trajs,
            dtrajs,
            umbrella_centers,
            umbrella_force_constants,
            estimator="mbar",
        )
        xgrid_mean = np.concatenate([xgrid, [2 * xgrid[-1] - xgrid[-2]]])
        xgrid_mean -= 0.5 * (xgrid[1] - xgrid[0])
        F = np.zeros(xgrid_mean.size)
        F[mbar_obj.active_set] = mbar_obj.stationary_distribution
        F = -np.log(F)

        self.rc_discretization = xgrid_mean
        self.rc_free_energies = F

        return xgrid_mean, F
