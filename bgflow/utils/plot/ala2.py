

def plot_2ala_ramachandran(traj, ax=None, weights=None):
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import numpy as np
    import mdtraj as md

    if ax is None:
        ax = plt.gca()

    if isinstance(weights, np.ndarray):
        ax.hist2d(
            md.compute_phi(traj)[1].reshape(-1),
            md.compute_psi(traj)[1].reshape(-1),
            bins=[np.linspace(-np.pi, np.pi, 64), np.linspace(-np.pi, np.pi, 64)],
            norm=mpl.colors.LogNorm(),
            weights=weights,
        )
    else:
        ax.hist2d(
            md.compute_phi(traj)[1].reshape(-1),
            md.compute_psi(traj)[1].reshape(-1),
            bins=[np.linspace(-np.pi, np.pi, 64), np.linspace(-np.pi, np.pi, 64)],
            norm=mpl.colors.LogNorm(),
        )

    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r"$\psi$")
