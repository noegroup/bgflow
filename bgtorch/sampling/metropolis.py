__author__ = "noe"

import numpy as np
import torch


class SystemWrapper(object):
    def __init__(self, system, use_numpy=False, cuda_device=None):
        self._system = system
        if hasattr(self._system, "_energy_numpy"):
            self._use_numpy = use_numpy
        else:
            self._use_numpy = False
        self._cuda_device = cuda_device

    def __call__(self, x, *args, **kwargs):
        if self._use_numpy:
            energy = self._system._energy_numpy(x)
        else:
            if not isinstance(x, torch.Tensor):
                x = torch.Tensor(x)
            if self._cuda_device is not None:
                x = x.device(self._cuda_device)
            energy = self._system.energy(x).detach().cpu().numpy()
        return energy.reshape(-1)


class MetropolisGauss(object):
    def __init__(
        self,
        energy_function,
        x0,
        temperature=1.0,
        noise=0.1,
        burnin=0,
        stride=1,
        nwalkers=1,
        mapper=None,
    ):
        """ Metropolis Monte-Carlo Simulation with Gaussian Proposal Steps

        Parameters
        ----------
        energy_function : Energy function
            Callable when evaluated on x yields energy u(x)
        x0 : [array]
            Initial configuration
        noise : float
            Noise intensity, standard deviation of Gaussian proposal step
        temperatures : float or array
            Temperature. By default (1.0) the energy is interpreted in reduced units.
            When given an array, its length must correspond to nwalkers, then the walkers
            are simulated at different temperatures.
        burnin : int
            Number of burn-in steps that will not be saved
        stride : int
            Every so many steps will be saved
        nwalkers : int
            Number of parallel walkers
        mapper : Mapper object
            Callable to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.energy_function = energy_function
        self.noise = noise
        self.temperature = temperature
        self.burnin = burnin
        self.stride = stride
        self.nwalkers = nwalkers
        self.mapper = mapper
        self.reset(x0)

    def _proposal_step(self):
        # proposal step
        self.x_prop = self.x + self.noise * np.random.randn(
            self.x.shape[0], self.x.shape[1]
        )
        if self.mapper is not None:
            self.x_prop = self.mapper(self.x_prop)
        self.E_prop = self.energy_function(self.x_prop)

    def _acceptance_step(self):
        # acceptance step
        acc = -np.log(np.random.rand()) > (self.E_prop - self.E) / self.temperature
        self.x = np.where(acc[:, None], self.x_prop, self.x)
        self.E = np.where(acc, self.E_prop, self.E)

    def reset(self, x0):
        # counters
        self.step = 0
        self.traj_ = []
        self.etraj_ = []

        # initial configuration
        self.x = np.tile(x0, (self.nwalkers, 1))
        if self.mapper is not None:
            self.x = self.mapper(self.x)
        self.E = self.energy_function(self.x)

        # save first frame if no burnin
        if self.burnin == 0:
            self.traj_.append(self.x)
            self.etraj_.append(self.E)

    @property
    def trajs(self):
        """ Returns a list of trajectories, one trajectory for each walker """
        T = np.array(self.traj_).astype(np.float32)
        return [T[:, i, :] for i in range(T.shape[1])]

    @property
    def traj(self):
        return self.trajs[0]

    @property
    def etrajs(self):
        """ Returns a list of energy trajectories, one trajectory for each walker """
        E = np.array(self.etraj_)
        return [E[:, i] for i in range(E.shape[1])]

    @property
    def etraj(self):
        return self.etrajs[0]

    def run(self, nsteps=1, verbose=0):
        for i in range(nsteps):
            self._proposal_step()
            self._acceptance_step()
            self.step += 1
            if verbose > 0 and i % verbose == 0:
                print("Step", i, "/", nsteps)
            if self.step > self.burnin and self.step % self.stride == 0:
                self.traj_.append(self.x)
                self.etraj_.append(self.E)


class ReplicaExchangeMetropolisGauss(object):
    def __init__(
        self,
        energy_function,
        x0,
        temperatures,
        noise=0.1,
        burnin=0,
        stride=1,
        mapper=None,
    ):
        if temperatures.size % 2 == 0:
            raise ValueError("Please use an odd number of temperatures.")
        self.temperatures = temperatures
        self.sampler = MetropolisGauss(
            energy_function,
            x0,
            temperature=temperatures,
            noise=noise,
            burnin=burnin,
            stride=stride,
            nwalkers=temperatures.size,
            mapper=mapper,
        )
        self.toggle = 0

    @property
    def trajs(self):
        return self.sampler.trajs

    @property
    def etrajs(self):
        return self.sampler.etrajs

    def run(self, nepochs=1, nsteps_per_epoch=1, verbose=0):
        for i in range(nepochs):
            self.sampler.run(nsteps=nsteps_per_epoch)
            # exchange
            for k in range(self.toggle, self.temperatures.size - 1, 2):
                c = -(self.sampler.E[k + 1] - self.sampler.E[k]) * (
                    1.0 / self.temperatures[k + 1] - 1.0 / self.temperatures[k]
                )
                acc = -np.log(np.random.rand()) > c
                if acc:
                    h = self.sampler.x[k].copy()
                    self.sampler.x[k] = self.sampler.x[k + 1].copy()
                    self.sampler.x[k + 1] = h
                    h = self.sampler.E[k]
                    self.sampler.E[k] = self.sampler.E[k + 1]
                    self.sampler.E[k + 1] = h
            self.toggle = 1 - self.toggle


# class MetropolisGauss(object):
#
#     def __init__(self, model, x0, noise=0.1, burnin=0, stride=1):
#         """ Metropolis Monte-Carlo Simulation with Gaussian Proposal Steps
#
#         Parameters
#         ----------
#         model : Energy model
#             Energy model object, must provide the function energy(x)
#         x0 : [array]
#             Initial configuration
#         noise : float
#             Noise intensity, standard deviation of Gaussian proposal step
#         burnin : int
#             Number of burn-in steps that will not be saved
#         stride : int
#             Every so many steps will be saved
#
#         """
#         self.model = model
#         self.noise = noise
#         self.burnin = burnin
#         self.stride = stride
#         self.reset(x0)
#
#     def _step(self, x1, e1):
#         # proposal step
#         x2 = x1 + self.noise*np.random.randn(1, self.model.dim)
#         e2 = self.model.energy(x2)[0]
#         # acceptance step
#         if -np.log(np.random.rand()) > e2-e1:
#             return x2, e2
#         else:
#             return x1, e1
#
#     def reset(self, x0):
#         self.x = x0
#         self.E = self.model.energy(x0)[0]
#
#         self.step = 0
#         self.traj_ = []
#         self.etraj_ = []
#
#         if self.burnin == 0:
#             self.traj_.append(x0[0])
#             self.etraj_.append(self.E)
#
#     @property
#     def traj(self):
#         return np.array(self.traj_).astype(np.float32)
#
#     @property
#     def etraj(self):
#         return np.array(self.etraj_)
#
#     def run(self, nsteps=1):
#         for i in range(nsteps):
#             self.x, self.E = self._step(self.x, self.E)
#             self.step += 1
#             if self.step > self.burnin and self.step % self.stride == 0:
#                 self.traj_.append(self.x[0].copy())
#                 self.etraj_.append(self.E)
