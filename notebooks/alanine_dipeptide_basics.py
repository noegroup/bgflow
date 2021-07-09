#!/usr/bin/env python
# coding: utf-8

# # Training a Boltzmann Generator for Alanine Dipeptide
# 
# This notebook introduces basic concepts behind `bgflow`. 
# 
# It shows how to build an train a Boltzmann generator for a small peptide. The most important aspects it will cover are
# 
# - retrieval of molecular training data
# - defining a internal coordinate transform
# - defining normalizing flow classes
# - combining different normalizing flows
# - training a Boltzmann generator via NLL and KLL
# 
# The main purpose of this tutorial is to introduce the implementation. The network design is optimized for educational purposes rather than good performance. In the conlusions, we will discuss some aspects of the generator that are not ideal and outline improvements.
# 
# ## Some Preliminaries
# 
# We instruct jupyter to reload any imports automatically and define the device and datatype, on which we want to perform the computations.


# In[3]:


import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# a context tensor to send data to the right device and dtype via '.to(ctx)'
ctx = torch.zeros([], device=device, dtype=dtype)

# a brief check if this module is the main executable (or imported)
main = (__name__ == "__main__")


# 
# 
# ## Load the Data and the Molecular System
# 
# Molecular trajectories and their corresponding potential energy functions are available from the `bgmol` repository.

# In[5]:


import os
from bgmol.datasets import Ala2TSF300

is_data_here = os.path.isfile("Ala2TSF300.npy")
dataset = Ala2TSF300(download=(not is_data_here), read=True)
system = dataset.system
coordinates = dataset.coordinates
temperature = dataset.temperature
dim = dataset.dim


# The energy model is a `bgflow.Energy` that wraps around OpenMM. The `n_workers` argument determines the number of openmm contexts that are used for energy evaluations. In notebooks, we set `n_workers=1` to avoid hickups. In production, we can omit this argument so that `n_workers` is automatically set to the number of CPU cores.

# In[6]:


target_energy = dataset.get_energy_model(n_workers=1)


# ### Visualize Data: Ramachandran Plot for the Backbone Angles

# In[7]:


import numpy as np
import mdtraj as md 
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

def plot_phi_psi(ax, trajectory, system):
    if not isinstance(trajectory, md.Trajectory):
        trajectory = md.Trajectory(
            xyz=trajectory.cpu().detach().numpy().reshape(-1, 22, 3), 
            topology=system.mdtraj_topology
        )
    phi, psi = system.compute_phi_psi(trajectory)
    
    ax.hist2d(phi, psi, 50, norm=LogNorm())
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_xlabel("$\phi$")
    _ = ax.set_ylabel("$\psi$")
    
    return trajectory


# In[8]:


if main:
    fig, ax = plt.subplots(figsize=(3,3))
    _ = plot_phi_psi(ax, dataset.trajectory, system)


# ## Split Data and Randomly Permute Samples

# In[9]:


n_train = len(dataset)//2
n_test = len(dataset) - n_train
permutation = np.random.permutation(n_train)

all_data = coordinates.reshape(-1, dataset.dim)
training_data = torch.tensor(all_data[permutation]).to(ctx)
test_data = torch.tensor(all_data[permutation + n_train]).to(ctx)


# ## Define the Internal Coordinate Transform
# 
# Rather than generating all-Cartesian coordinates, we use a mixed internal coordinate transform.
# The five central alanine atoms will serve as a Cartesian "anchor", from which all other atoms are placed with respect to internal coordinates (IC) defined through a z-matrix. We have deposited a valid `z_matrix` and the corresponding `rigid_block` in the `dataset.system` from `bgmol`.

# In[10]:


import bgflow as bg


# In[11]:


# throw away 6 degrees of freedom (rotation and translation)
dim_cartesian = len(system.rigid_block) * 3 - 6
dim_bonds = len(system.z_matrix)
dim_angles = dim_bonds
dim_torsions = dim_bonds


# In[12]:


coordinate_transform = bg.MixedCoordinateTransformation(
    data=training_data, 
    z_matrix=system.z_matrix,
    fixed_atoms=system.rigid_block,
    keepdims=dim_cartesian, 
    normalize_angles=True,
).to(ctx)


# For demonstration, we transform the first 3 samples from the training data set into internal coordinates as follows:

# In[13]:


bonds, angles, torsions, cartesian, dlogp = coordinate_transform.forward(training_data[:3])
bonds.shape, angles.shape, torsions.shape, cartesian.shape, dlogp.shape


# ## Prior Distribution
# 
# The next step is to define a prior distribution that we can easily sample from. The normalizing flow will be trained to transform such latent samples into molecular coordinates. Here, we just take a normal distribution, which is a rather naive choice for reasons that will be discussed in other notebooks.

# In[14]:


dim_ics = dim_bonds + dim_angles + dim_torsions + dim_cartesian
mean = torch.zeros(dim_ics).to(ctx) 
# passing the mean explicitly to create samples on the correct device
prior = bg.NormalDistribution(dim_ics, mean=mean)


# ## Normalizing Flow
# 
# Next, we set up the normalizing flow by stacking together different neural networks. For now, we will do this in a rather naive way, not distinguishing between bonds, angles, and torsions. Therefore, we will first define a flow that splits the output from the prior into the different IC terms.
# 
# ### Split Layer

# In[15]:


split_into_ics_flow = bg.SplitFlow(dim_bonds, dim_angles, dim_torsions, dim_cartesian)


# In[16]:


# test
_ics = split_into_ics_flow(prior.sample(3))[:-1]
coordinate_transform.forward(*_ics, inverse=True)[0].shape


# ### Coupling Layers
# 
# Next, we will set up so-called RealNVP coupling layers, which split the input into two channels and then learn affine transformations of channel 1 conditioned on channel 2. Here we will do the split naively between the first and second half of the degrees of freedom.

# In[17]:


class RealNVP(bg.SequentialFlow):
    
    def __init__(self, dim, hidden):
        self.dim = dim
        self.hidden = hidden
        super().__init__(self._create_layers())
    
    def _create_layers(self):
        dim_channel1 =  self.dim//2
        dim_channel2 = self.dim - dim_channel1
        split_into_2 = bg.SplitFlow(dim_channel1, dim_channel2)
        
        layers = [
            # -- split
            split_into_2,
            # --transform
            self._coupling_block(dim_channel1, dim_channel2),
            bg.SwapFlow(),
            self._coupling_block(dim_channel2, dim_channel1),
            # -- merge
            bg.InverseFlow(split_into_2)
        ]
        return layers
        
    def _dense_net(self, dim1, dim2):
        return bg.DenseNet(
            [dim1, *self.hidden, dim2],
            activation=torch.nn.ReLU()
        )
    
    def _coupling_block(self, dim1, dim2):
        return bg.CouplingFlow(bg.AffineTransformer(
            shift_transformation=self._dense_net(dim1, dim2),
            scale_transformation=self._dense_net(dim1, dim2)
        ))
    


# In[18]:


RealNVP(dim_ics, hidden=[128]).to(ctx).forward(prior.sample(3))[0].shape


# ### Boltzmann Generator
# 
# Finally, we define the Boltzmann generator.
# It will sample molecular conformations by 
# 
# 1. sampling in latent space from the normal prior distribution,
# 2. transforming the samples into a more complication distribution through a number of RealNVP blocks (the parameters of these blocks will be subject to optimization),
# 3. splitting the output of the network into blocks that define the internal coordinates, and
# 4. transforming the internal coordinates into Cartesian coordinates through the inverse IC transform.

# In[19]:


n_realnvp_blocks = 5
layers = []

for i in range(n_realnvp_blocks):
    layers.append(RealNVP(dim_ics, hidden=[128, 128, 128]))
layers.append(split_into_ics_flow)
layers.append(bg.InverseFlow(coordinate_transform))

flow = bg.SequentialFlow(layers).to(ctx)


# In[20]:


# test
flow.forward(prior.sample(3))[0].shape


# In[21]:


# print number of trainable parameters
"#Parameters:", np.sum([np.prod(p.size()) for p in flow.parameters()])


# In[22]:


generator = bg.BoltzmannGenerator(
    flow=flow,
    prior=prior,
    target=target_energy
)


# ## Train
# 
# Boltzmann generators can be trained in two ways:
# 1. by matching the density of samples from the training data via the negative log likelihood (NLL), and
# 2. by matching the target density via the backward Kullback-Leibler loss (KLL).
# 
# NLL-based training is faster, as it does not require the computation of molecular target energies. Therefore, we will first train the generator solely by density estimation.
# 
# ### NLL Training

# In[23]:


nll_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-3)
nll_trainer = bg.KLTrainer(
    generator, 
    optim=nll_optimizer,
    train_energy=False
)


# In[ ]:


if main:
    nll_trainer.train(
        n_iter=20000, 
        data=training_data,
        batchsize=128,
        n_print=1000, 
        w_energy=0.0
    )


# To see what the generator has learned so far, let's first create a bunch of samples and compare their backbone angles with the molecular dynamics data. Let's also plot their energies.

# In[ ]:


def plot_energies(ax, samples, target_energy, test_data):
    sample_energies = target_energy.energy(samples).cpu().detach().numpy()
    md_energies = target_energy.energy(test_data[:len(samples)]).cpu().detach().numpy()
    cut = max(np.percentile(sample_energies, 80), 20)
    
    ax.set_xlabel("Energy   [$k_B T$]")
    # y-axis on the right
    ax2 = plt.twinx(ax)
    ax.get_yaxis().set_visible(False)
    
    ax2.hist(sample_energies, range=(-50, cut), bins=40, density=False, label="BG")
    ax2.hist(md_energies, range=(-50, cut), bins=40, density=False, label="MD")
    ax2.set_ylabel(f"Count   [#Samples / {len(samples)}]")
    ax2.legend()


# In[ ]:


if main:
    
    n_samples = 10000
    samples = generator.sample(n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    fig.tight_layout()

    plot_phi_psi(axes[0], samples, system)
    plot_energies(axes[1], samples, target_energy, test_data)

    del samples


# ### Mixed Training
# 
# The next step is "mixed" training with a combination of NLL and KLL. To retain some of the progress made in the NLL phase, we decrease the learning rate and increase the batch size.

# In[ ]:


mixed_optimizer = torch.optim.Adam(generator.parameters(), lr=1e-4)
mixed_trainer = bg.KLTrainer(
    generator, 
    optim=mixed_optimizer,
    train_energy=True
)


# Mixed training will be considerably slower. 
# To speed it up, you can change the settings for the OpenMM energy when creating the energy model. For example, consider not passing `n_workers=1`.
# 
# To avoid large potential energy gradients from singularities, the components of the KL gradient are constrained to (-100, 100). 

# In[ ]:


if main:
    mixed_trainer.train(
        n_iter=2000, 
        data=training_data,
        batchsize=1000,
        n_print=100, 
        w_energy=0.1,
        w_likelihood=0.9,
        clip_forces=20.0
    )


# Plot the results:

# In[ ]:


if main:
    
    n_samples = 10000
    samples = generator.sample(n_samples)

    fig, axes = plt.subplots(1, 2, figsize=(6,3))
    fig.tight_layout()

    plot_phi_psi(axes[0], samples, system)
    plot_energies(axes[1], samples, target_energy, test_data)

    del samples


# ## Conclusions
# 
# This tutorial has introduced the most basic concepts and implementations underlying Boltzmann generators and `bgflow`. That said, the trained networks did not do a particularly good job in reproducing the molecular Boltzmann distribution. Specifically, they only modeled the major modes of the $\phi$ angle and still produced many samples with unreasonably large energies. Let's look at a few shortcomings of the present architecture:
# 
# ### 1) Unconstrained Internal Coordinates
# Bonds, angles, and torsions must not take arbitrary values in principle. Bond lengths need to be positive, angles live in $[0,\pi],$ and torsions are periodic in $[-\pi, \pi].$ Neither those bounds nor the periodicity of torsions distributions have been taken into account by the present Boltzmann generator. The layers of the normalizing flow should be build in a way that preserves these constraints on the ICs.
# 
# ### 2)  Arbitrary Coupling
# The input for the coupling layers was split into two channels rather arbitrarily (first vs. second half). A partial remedy is to define the conditioning in a physically informed manner. Another solution is to augment the base space by momenta, which can be done with augmented normalizing flows (see for instance the notebook on temperature-steering flows).
# 
# ### 3) RealNVP Layers
# Affine coupling layers are well-known to perform poorly in separating modes. This explains that the metastable region around $\phi \approx \pi/2$ was not captured by the generator. Other architectures such as augmented flows or neural spline flows do a better job for complicated, multimodal distributions.
# 
# ### 4) Training
# The generators were only trained for relatively few iterations and performance may improve with longer training and better choices of the learning rate and hyperparameters.

# In[ ]:




