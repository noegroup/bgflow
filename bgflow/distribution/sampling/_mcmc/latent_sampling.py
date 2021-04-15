__author__ = "noe"

from deep_boltzmann.util import ensure_traj
from scipy.misc import logsumexp
import numpy as np
import keras


def plot_latent_sampling(rc, Z, E, rclabel="Reaction coord.", maxener=100):
    import matplotlib.pyplot as plt
    from deep_boltzmann.plot import plot_traj_hist

    plt.figure(figsize=(20, 12))
    ax1 = plt.subplot2grid((3, 4), (0, 0), colspan=3)
    ax2 = plt.subplot2grid((3, 4), (0, 3))
    plot_traj_hist(rc, ax1=ax1, ax2=ax2, color="blue", ylabel=rclabel)
    ax3 = plt.subplot2grid((3, 4), (1, 0), colspan=3)
    ax4 = plt.subplot2grid((3, 4), (1, 3))
    Elow = np.minimum(E, maxener + 1)
    plot_traj_hist(
        Elow,
        ax1=ax3,
        ax2=ax4,
        color="blue",
        ylim=[Elow.min() - 0.5 * (maxener - Elow.min()), maxener],
        ylabel="Energy",
    )
    ax5 = plt.subplot2grid((3, 4), (2, 0), colspan=3)
    ax6 = plt.subplot2grid((3, 4), (2, 3))
    plot_traj_hist(
        np.mean(Z ** 2, axis=1), ax1=ax5, ax2=ax6, color="blue", ylabel="|Z|"
    )


def sample_RC(
    network,
    nsamples,
    compute_rc,
    batchsize=10000,
    verbose=True,
    temperature=1.0,
    xmapper=None,
    failfast=True,
):
    """ Generates x samples from latent network and computes their weights

    Parameters
    ----------
    network : latent network
        Network to generate samples and compute energy
    nsamples : int
        Number of samples
    compute_rc : function
        function to compute RC
    batchsize : int
        Number of samples to generate at a time
    verbose : bool
        True in order to print progress
    xmapper : Mapper
        If given, permuted samples will be discarded
    failfast : bool
        Raise exception if a NaN is generated

    """
    D = []
    W = []
    niter = int(nsamples / batchsize) + 1
    for i in range(niter):
        print("Iteration", i, "/", niter)
        _, sample_x, _, E_x, logw = network.sample(
            temperature=temperature, nsample=batchsize
        )
        if np.any(np.isnan(E_x)) and failfast:
            raise ValueError("Energy NaN")
        if xmapper is not None:
            notperm = np.logical_not(xmapper.is_permuted(sample_x))
            sample_x = sample_x[notperm]
            logw = logw[notperm]
        D.append(compute_rc(sample_x))
        W.append(logw)
    D = np.concatenate(D)[:nsamples]
    W = np.concatenate(W)[:nsamples]
    W -= W.max()
    return D, W


class LatentModel:
    def __init__(self, network):
        self.network = network
        self.dim = network.energy_model.dim

    def energy(self, z):
        x = self.network.transform_zx(z)
        return self.network.energy_model.energy(x)


class BiasedModel:
    def __init__(self, model, bias_energy, rc_value=None):
        """
        Parameters
        ----------
        network
            Latent Boltzmann Generator
        bias_energy : function
            Function to compute bias on configuration or reaction coordinate
        rc_value
            Function to compute reaction coordinate. If given, bias energy will be evaluated
            on the result of this function.

        """
        self.energy_model = model
        self.dim = model.dim
        self.bias_energy = bias_energy
        self.rc_value = rc_value

    def energy(self, x):
        if self.rc_value is None:
            return self.energy_model.energy(x) + self.bias_energy(x)
        else:
            return self.energy_model.energy(x) + self.bias_energy(self.rc_value(x))


class GaussianPriorMCMC(object):
    def __init__(
        self,
        network,
        energy_model=None,
        z0=None,
        std_z=1.0,
        batchsize=10000,
        xmapper=None,
        tf=False,
        temperature=1.0,
    ):
        """ Latent Prior Markov-Chain Monte Carlo

        Samples from a Gaussian prior in latent space and accepts according to energy in configuration space.

        Parameters
        ----------
        network : latent network
            Network mapping between latent and configuration space
        energy : energy model
            If None, will use the network's energy model
        z0 : None or array
            if None, will be sampled from scratch
        std_z : float or array of float
            Standard deviation of Gaussian prior. If an array is given, will select std_z with uniform probability.
        batchsize : int
            Number of samples generated at a time
        tf : bool
            If True, use tensorflow implementation of energies. If False, use numpy implementation
        xmapper : Configuration Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.
        temperature : float
            Temperature factor. If not equal to 1.0 the energy will be scaled accordingly.

        """
        self.network = network
        if energy_model is None:
            self.energy_model = network
        else:
            self.energy_model = energy_model
        self.std_z = np.array(std_z)
        self.std_z = self.std_z.reshape((self.std_z.size,))
        self.batchsize = batchsize
        self.tf = tf
        self.temperature = temperature
        if temperature != 1.0:
            self.std_z = np.array([np.sqrt(temperature)])
        # generate first sample
        s = np.random.randint(low=0, high=self.std_z.size)  # step chosen
        if z0 is None:
            self.z = self.std_z[s] * np.random.randn(1, self.network.zdim)
        else:
            self.z = ensure_traj(z0)
        self.x, self.J = self.network.transform_zxJ(self.z)
        self.xmapper = xmapper
        if self.xmapper is not None:
            if self.xmapper.is_permuted(self.x)[0]:
                raise RuntimeError(
                    "Starting configuration is already permuted. Choose a different z0."
                )
        self.e = self.energy_model.energy(self.x) / self.temperature

    def _propose_batch(self):
        # sample and set first data point to current sample for easy acceptance
        sample_s = np.random.randint(
            low=0, high=self.std_z.size, size=self.batchsize + 1
        )  # step chosen
        sample_z = self.std_z[sample_s][:, None] * np.random.randn(
            self.batchsize + 1, self.network.zdim
        )
        sample_z[0] = self.z
        sample_x, sample_J = self.network.transform_zxJ(sample_z)
        if self.xmapper is not None:
            isP = self.xmapper.is_permuted(sample_x)
            I_permuted = np.where(isP == True)[0]
            # resample
            while I_permuted.size > 0:
                sample_z[I_permuted] = self.std_z[sample_s[I_permuted]][
                    :, None
                ] * np.random.randn(I_permuted.size, self.network.zdim)
                sample_x[I_permuted], sample_J[I_permuted] = self.network.transform_zxJ(
                    sample_z[I_permuted]
                )
                isP[I_permuted] = self.xmapper.is_permuted(sample_x[I_permuted])
                I_permuted = np.where(isP == True)[0]
        if self.tf:
            sample_e = (
                keras.backend.eval(self.energy_model.energy(sample_x))
                / self.temperature
            )
        else:
            sample_e = self.energy_model.energy(sample_x) / self.temperature
        return sample_s, sample_z, sample_x, sample_e, sample_J

    def _accept_batch(self, sample_s, sample_z, sample_x, sample_e, sample_J):
        n = np.size(sample_e)
        R = -np.log(np.random.rand(n))  # random array
        sel = np.zeros(n, dtype=int)  # selector array
        factor = 1.0 / (2.0 * self.std_z * self.std_z)
        for i in range(1, n):
            if self.std_z.size == 1:
                log_p_forward = -factor[0] * np.sum(sample_z[i] ** 2)
                log_p_backward = -factor[0] * np.sum(self.z ** 2)
            else:
                log_p_forward = logsumexp(
                    -factor * np.sum(sample_z[i] ** 2)
                    - self.network.zdim * np.log(self.std_z)
                )
                log_p_backward = logsumexp(
                    -factor * np.sum(self.z ** 2)
                    - self.network.zdim * np.log(self.std_z)
                )
                # use sequential stepping
                # log_p_forward = - factor[sample_s[i]] * np.sum(sample_z[i]**2)
                # log_p_backward = - factor[sample_s[i]] * np.sum(self.z**2)
            if (
                R[i]
                > self.J
                - sample_J[i]
                + sample_e[i]
                - self.e
                + log_p_forward
                - log_p_backward
            ):
                sel[i] = i
                self.z = sample_z[i]
                self.e = sample_e[i]
                self.J = sample_J[i]
            else:
                sel[i] = sel[i - 1]
        sel = sel[1:]
        return sample_s[sel], sample_z[sel], sample_x[sel], sample_e[sel], sample_J[sel]

    def run(self, N, return_proposal=False):
        """ Generates N samples

        Returns
        -------
        Z : array(N, dim)
            Prior (z) samples
        X : array(N, dim)
            Sampled Configurations
        E : array(N)
            Energies of sampled configurations

        """
        n = 0
        Zp = []
        Xp = []
        Ep = []
        Jp = []
        Z = []
        X = []
        E = []
        J = []
        while n < N:
            sample_s, sample_z, sample_x, sample_e, sample_J = self._propose_batch()
            Zp.append(sample_z)
            Xp.append(sample_x)
            Ep.append(sample_e)
            Jp.append(sample_J)
            acc_s, acc_z, acc_x, acc_e, acc_J = self._accept_batch(
                sample_s, sample_z, sample_x, sample_e, sample_J
            )
            Z.append(acc_z)
            X.append(acc_x)
            E.append(acc_e)
            J.append(acc_J)
            n += sample_e.size
        Zp = np.vstack(Zp)[:N]
        Xp = np.vstack(Xp)[:N]
        Ep = np.concatenate(Ep)[:N]
        Jp = np.concatenate(Jp)[:N]
        Z = np.vstack(Z)[:N]
        X = np.vstack(X)[:N]
        E = np.concatenate(E)[:N]
        J = np.concatenate(J)[:N]
        # return Zp, Xp, Ep, Jp
        if return_proposal:
            return Zp, Xp, Ep, Jp, Z, X, E, J
        else:
            return Z, X, E, J


def eval_GaussianPriorMCMC(
    network,
    metric,
    nrepeat,
    nsteps,
    energy_model=None,
    burnin=10000,
    z0=None,
    temperature=1.0,
    batchsize=10000,
    xmapper=None,
    tf=False,
    verbose=True,
):
    z2s = []
    ms = []
    Es = []
    Js = []
    for i in range(nrepeat):
        print("Iteration", i)
        gp_mcmc = GaussianPriorMCMC(
            network,
            energy_model=energy_model,
            z0=z0,
            batchsize=batchsize,
            xmapper=xmapper,
            tf=tf,
            temperature=temperature,
        )
        _, _, _, _ = gp_mcmc.run(burnin, return_proposal=False)
        Z, X, E, J = gp_mcmc.run(nsteps, return_proposal=False)
        z2s.append(np.sum(Z ** 2, axis=1))
        ms.append(metric(X))
        Es.append(E)
        Js.append(J)
    return z2s, ms, Es, Js


# TODO: Currently not compatible with RealNVP networks. Refactor to include Jacobian
# TODO: Mapping handling should be changed, so as to reject permuted configurations
class LatentMetropolisGauss(object):
    def __init__(
        self,
        latent_network,
        z0,
        noise=0.1,
        burnin=0,
        stride=1,
        nwalkers=1,
        xmapper=None,
    ):
        """ Metropolis Monte-Carlo Simulation with Gaussian Proposal Steps

        Parameters
        ----------
        model : Energy model
            Energy model object, must provide the function energy(x)
        z0 : [array]
            Initial configuration
        noise : float
            Noise intensity, standard deviation of Gaussian proposal step
        burnin : int
            Number of burn-in steps that will not be saved
        stride : int
            Every so many steps will be saved
        nwalkers : int
            Number of parallel walkers
        xmapper : Configuration Mapper object
            Object with function map(X), e.g. to remove permutation.
            If given will be applied to each accepted configuration.

        """
        self.network = latent_network
        self.model = latent_network.energy_model
        self.noise = noise
        self.burnin = burnin
        self.stride = stride
        self.nwalkers = nwalkers
        if xmapper is None:

            class DummyMapper(object):
                def map(self, X):
                    return X

            xmapper = DummyMapper()
        self.xmapper = xmapper
        self.reset(z0)

    def _proposal_step(self):
        # proposal step
        self.z_prop = self.z + self.noise * np.random.randn(
            self.z.shape[0], self.z.shape[1]
        )
        x_prop_unmapped = self.network.transform_zx(self.z_prop)
        self.x_prop = self.xmapper.map(x_prop_unmapped)
        if np.max(np.abs(self.x_prop - x_prop_unmapped)) > 1e-7:
            self.z_prop = self.network.transform_xz(self.x_prop)
        self.E_prop = self.model.energy(self.x_prop)

    def _acceptance_step(self):
        # acceptance step
        self.acc = -np.log(np.random.rand()) > self.E_prop - self.E
        self.z = np.where(self.acc[:, None], self.z_prop, self.z)
        self.x = np.where(self.acc[:, None], self.x_prop, self.x)
        self.E = np.where(self.acc, self.E_prop, self.E)

    def reset(self, z0):
        # counters
        self.step = 0
        self.accs_ = []
        self.traj_ = []
        self.ztraj_ = []
        self.etraj_ = []

        # initial configuration
        self.z = np.tile(z0, (self.nwalkers, 1))
        self.x = self.network.transform_zx(self.z)
        self.x = self.xmapper.map(self.x)
        self.E = self.model.energy(self.x)

        # save first frame if no burnin
        if self.burnin == 0:
            self.traj_.append(self.x)
            self.ztraj_.append(self.z)
            self.etraj_.append(self.E)

    @property
    def trajs(self):
        """ Returns a list of trajectories, one trajectory for each walker """
        T = np.array(self.traj_).astype(np.float32)
        return [T[:, i, :] for i in range(T.shape[1])]

    @property
    def ztrajs(self):
        """ Returns a list of trajectories, one trajectory for each walker """
        Z = np.array(self.ztraj_).astype(np.float32)
        return [Z[:, i, :] for i in range(Z.shape[1])]

    @property
    def etrajs(self):
        """ Returns a list of energy trajectories, one trajectory for each walker """
        E = np.array(self.etraj_)
        return [E[:, i] for i in range(E.shape[1])]

    def run(self, nsteps=1):
        for i in range(nsteps):
            self._proposal_step()
            self._acceptance_step()
            self.step += 1
            if self.step > self.burnin and self.step % self.stride == 0:
                self.accs_.append(self.acc.copy())
                self.traj_.append(self.x.copy())
                self.ztraj_.append(self.z.copy())
                self.etraj_.append(self.E)


def sample_hybrid_zprior_zmetro(
    network,
    niter,
    nprior,
    nmetro,
    prior_std=1.0,
    noise=0.1,
    z0=None,
    x0=None,
    mapper=None,
    verbose=0,
):
    """ Samples iteratively using Prior MCMC in z-space and Metropolis MCMC in z-space

    Parameters
    ----------
    network : network
        Latent transformer network
    niter : int
        Number of sampling iterations
    nprior : int
        Number of steps in each Prior MCMC sampling
    nmetro : int
        Number of steps in each Metropolis MCMC sampling
    prior_std : float or array
        Standard deviation of Gaussian in z for Prior MCMC
    noise : float
        Standard deviation of Gaussian proposal step in Metropolis MCMC
    z0 : None or array
        Initial configuration in z-space, if desired
    x0 : None or array
        Initial configuration in z-space, if desired
    mapper : Mapper object
        Mapper object, e.g. to remove invariances in x
    verbose : int
        Print every "verbose" iterations. 0 means never

    Returns
    -------
    Z : array(N, dim)
        Sampled z
    X : array(N, dim)
        Sampled x Configurations
    E : array(N)
        Energies of sampled configurations

    """
    Z = []
    X = []
    E = []

    # initial configuration
    if z0 is not None and x0 is not None:
        raise ValueError("Cannot set both x0 and z0.")
    if x0 is not None:
        z0 = network.transform_xz(x0)

    for i in range(niter):
        if verbose > 0 and (i + 1) % verbose == 0:
            print((i + 1), "/", niter)
        # Gaussian prior MCMC
        prior_mc = GaussianPriorMCMC(network, z0=z0, std_z=prior_std, batchsize=nprior)
        z, x, e = prior_mc.run(nprior)
        if mapper is not None:
            x = mapper.map(x)
        X.append(x)
        Z.append(z)
        E.append(e)
        z0 = prior_mc.z.copy()

        lmg = LatentMetropolisGauss(network, z0, noise=noise, xmapper=mapper)
        lmg.run(nmetro)
        X.append(lmg.trajs[0])
        Z.append(lmg.ztrajs[0])
        E.append(lmg.etrajs[0])
        z0 = lmg.ztrajs[0][-1]

    Z = np.vstack(Z)
    X = np.vstack(X)
    E = np.concatenate(E)

    return Z, X, E


def sample_hybrid_zprior_xmetro(
    network,
    niter,
    nprior,
    nmetro,
    prior_std=1.0,
    noise=0.02,
    z0=None,
    x0=None,
    mapper=None,
    verbose=0,
):
    """ Samples iteratively using Prior MCMC in z-space and Metropolis MCMC in z-space

    Parameters
    ----------
    network : network
        Latent transformer network
    niter : int
        Number of sampling iterations
    nprior : int
        Number of steps in each Prior MCMC sampling
    nmetro : int
        Number of steps in each Metropolis MCMC sampling
    prior_std : float or array
        Standard deviation of Gaussian in z for Prior MCMC
    noise : float
        Standard deviation of Gaussian proposal step in Metropolis MCMC
    z0 : None or array
        Initial configuration in z-space, if desired
    x0 : None or array
        Initial configuration in z-space, if desired
    mapper : Mapper object
        Mapper object, e.g. to remove invariances in x
    verbose : int
        Print every "verbose" iterations. 0 means never

    Returns
    -------
    Z : array(N, dim)
        Sampled z
    X : array(N, dim)
        Sampled x Configurations
    E : array(N)
        Energies of sampled configurations

    """
    from deep_boltzmann.sampling import MetropolisGauss

    Z = []
    X = []
    E = []
    J = []

    # initial configuration
    if z0 is not None and x0 is not None:
        raise ValueError("Cannot set both x0 and z0.")
    if x0 is not None:
        z0 = network.transform_xz(x0)

    for i in range(niter):
        if verbose > 0 and (i + 1) % verbose == 0:
            print((i + 1), "/", niter)
        # Gaussian prior MCMC
        prior_mc = GaussianPriorMCMC(
            network, z0=z0, std_z=prior_std, batchsize=nprior, xmapper=mapper
        )
        z, x, e, j = prior_mc.run(nprior)
        if mapper is not None:
            x = mapper.map(x)
        X.append(x)
        Z.append(z)
        E.append(e)
        J.append(j)
        z0 = prior_mc.z.copy()

        # Run Metropolis MCMC in x
        x0 = prior_mc.x
        lmg = MetropolisGauss(network.energy_model, x0, noise=noise, mapper=mapper)
        lmg.run(nmetro)
        X.append(lmg.trajs[0])
        E.append(lmg.etrajs[0])

        # transform to z
        ztraj = network.transform_xz(lmg.trajs[0])
        Z.append(ztraj)
        z0 = ztraj[-1]

    Z = np.vstack(Z)
    X = np.vstack(X)
    E = np.concatenate(E)
    J = np.concatenate(J)

    return Z, X, E, J
