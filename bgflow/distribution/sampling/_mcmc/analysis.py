__author__ = "noe"

import numpy as np


def mean_finite_(x, min_finite=1):
    """ Computes mean over finite values """
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) > min_finite:
        return np.mean(x[isfin])
    else:
        return np.nan


def std_finite_(x, min_finite=2):
    """ Computes mean over finite values """
    isfin = np.isfinite(x)
    if np.count_nonzero(isfin) > min_finite:
        return np.std(x[isfin])
    else:
        return np.nan


def mean_finite(x, axis=None, min_finite=1):
    if axis is None:
        return mean_finite_(x)
    if axis == 0 or axis == 1:
        M = np.zeros((x.shape[axis - 1],))
        for i in range(x.shape[axis - 1]):
            if axis == 0:
                M[i] = mean_finite_(x[:, i])
            else:
                M[i] = mean_finite_(x[i])
        return M
    else:
        raise NotImplementedError("axis value not implemented:", axis)


def std_finite(x, axis=None, min_finite=2):
    if axis is None:
        return mean_finite_(x)
    if axis == 0 or axis == 1:
        S = np.zeros((x.shape[axis - 1],))
        for i in range(x.shape[axis - 1]):
            if axis == 0:
                S[i] = std_finite_(x[:, i])
            else:
                S[i] = std_finite_(x[i])
        return S
    else:
        raise NotImplementedError("axis value not implemented:", axis)


def metropolis_function(x):
    return np.minimum(np.exp(-x), 1)


def bar(sampled_a_uab, sampled_b_uba):
    """
    Parameters:
    -----------
    sampled_a_uab : array
        Ub-Ua for samples in A
    sampled_b_uba : array
        Ua-Ub for samples in B
    """
    R = np.mean(metropolis_function(sampled_a_uab)) / np.mean(
        metropolis_function(sampled_b_uba)
    )
    return -np.log(R)


def free_energy_bootstrap(
    D, l, r, n, sample=100, weights=None, bias=None, temperature=1.0
):
    """ Bootstrapped free energy calculation

    If D is a single array, bootstraps by sample. If D is a list of arrays, bootstraps by trajectories

    Parameters
    ----------
    D : array of list of arrays
        Samples in the coordinate in which we compute the free energy
    l : float
        leftmost bin boundary
    r : float
        rightmost bin boundary
    n : int
        number of bins
    sample : int
        number of bootstraps
    weights : None or arrays matching D
        sample weights
    bias : function
        if not None, the given bias will be removed.

    Returns
    -------
    bin_means : array((nbins,))
        mean positions of bins
    Es : array((sample, nbins))
        for each bootstrap the free energies of bins.

    """
    bins = np.linspace(l, r, n)
    Es = []
    I = np.arange(len(D))
    by_traj = isinstance(D, list)
    for s in range(sample):
        Isel = np.random.choice(I, size=len(D), replace=True)
        if by_traj:
            Dsample = np.concatenate([D[i] for i in Isel])
            Wsample = None
            if weights is not None:
                Wsample = np.concatenate([weights[i] for i in Isel])
            Psample, _ = np.histogram(Dsample, bins=bins, weights=Wsample, density=True)
        else:
            Dsample = D[Isel]
            Wsample = None
            if weights is not None:
                Wsample = weights[Isel]
            Psample, _ = np.histogram(Dsample, bins=bins, weights=Wsample, density=True)
        Es.append(-np.log(Psample))
    Es = np.vstack(Es)
    Es -= Es.mean(axis=0).min()
    bin_means = 0.5 * (bins[:-1] + bins[1:])

    if bias is not None:
        B = bias(bin_means) / temperature
        Es -= B

    return bin_means, Es  # / temperature


def free_energy_bootstrap_2BGs(
    bg1, bg2, nsamples, nbootstrap, temperature=1.0, verbose=False
):
    """ Computes free energy difference between the states sampled by two Boltzmann generators
    with a joint latent space

    Parameters
    ----------
    bg1 : EnergyInvNet
        Boltzmann Generator 1
    bg2 : EnergyInvNet
        Boltzmann Generator 2
    nsamples : int
        number of samples used in each bootstrap
    nbootstrap : int
        number of bootstrap samples

    Returns
    -------
    dFs : array
        Array of bootstrapped samples of free energy differences F2-F1.

    """
    dEs = []
    W1s = []
    W2s = []
    nsample_per_bootstrap = min(100000, nsamples)
    niter = int(nsamples / nsample_per_bootstrap)
    for i in range(niter):
        if verbose:
            print(i)
        sample_z = np.sqrt(temperature) * np.random.randn(
            nsample_per_bootstrap, bg1._dim
        )
        sample_x1, sampleJzx1 = bg1.transform_zxJ(sample_z)
        energies_sample_x1 = bg1.energy(sample_x1)
        sample_x2, sampleJzx2 = bg2.transform_zxJ(sample_z)
        energies_sample_x2 = bg1.energy(sample_x2)

        energies_sample_z = bg1._dim + np.sum(sample_z ** 2, axis=1) / (
            2.0 * temperature
        )
        logw1 = -energies_sample_x1 + energies_sample_z + sampleJzx1
        w1 = np.exp((logw1 - logw1.max()) / temperature)
        logw2 = -energies_sample_x2 + energies_sample_z + sampleJzx2
        w2 = np.exp((logw2 - logw2.max()) / temperature)

        dE = energies_sample_x2 - sampleJzx2 - energies_sample_x1 + sampleJzx1
        dEs.append(dE)
        W1s.append(w1)
        W2s.append(w2)
    dEs = np.concatenate(dEs)
    W1s = np.concatenate(W1s)
    W2s = np.concatenate(W2s)

    # Bootstrap free energy difference
    dFs = []
    for i in range(nbootstrap):
        I = np.arange(W1s.size)
        Isel = np.random.choice(I, I.size)
        f = np.minimum(np.exp(-dEs[Isel] / temperature), 1.0)
        p12 = np.sum(W1s[Isel] * f) / np.sum(W1s[Isel])
        f = np.minimum(np.exp(dEs[Isel] / temperature), 1.0)
        p21 = np.sum(W2s[Isel] * f) / np.sum(W2s[Isel])
        dFs.append(-temperature * np.log(p12 / p21))

    return dFs
