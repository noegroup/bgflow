# bgflow

[![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/noegroup/bgflow.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/noegroup/bgflow/context:python)

Bgflow is a [pytorch](https://github.com/pytorch/pytorch) framework for
[Boltzmann Generators](https://science.sciencemag.org/content/365/6457/eaaw1147) (BG) and other sampling methods.

> Boltzmann Generators use neural networks to learn a coordinate transformation
> of the complex configurational equilibrium distribution to a distribution that 
> can be easily sampled. Accurate computation of free-energy differences 
> and discovery of new configurations are demonstrated, 
> providing a statistical mechanics tool that can 
> avoid rare events during sampling without prior knowledge of reaction coordinates.
> -- [Noe et al. (2019)](https://science.sciencemag.org/content/365/6457/eaaw1147)

This framework provides:

* A general API for Boltzmann Generators
* Different invertible [normalizing flow](https://arxiv.org/abs/1912.02762) structures to build BGs
    * [Coupling flows](https://arxiv.org/abs/1410.8516)
    * [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
    * [Stochastic Normalizing Flows](https://arxiv.org/abs/2002.06707)
    * [Equivariant flows](https://arxiv.org/abs/2006.02425)
    * [Temperature-steerable flows](https://arxiv.org/abs/2108.01590)
    * [P4Inv Linear Flows](https://arxiv.org/abs/2010.07033)
    * [Neural Spline Flows](https://arxiv.org/abs/1906.04032)
    * [Augmented Normalizing Flows](https://arxiv.org/abs/2002.07101)
* Other sampling methods
    * Markov-Chain Monte Carlo
    * Molecular dynamics
    * ... replica exchange and more coming soon ...
* API to combine BGs with other sampling methods
* [OpenMM](https://github.com/openmm/openmm) bridge 
* Internal Coordinate Transformations to map between Z-matrices and Cartesian coordinates
***
## [Disclaimer](#disclaimer)
This library is alpha software under active development.
Certain elements of its API are going to change in the 
future and some current implementations may not be tested.

If you are interested in contributing to the library,
please feel free to open a 
[pull request](https://github.com/noegroup/bgflow/pulls)
or report an [issue](https://github.com/noegroup/bgflow/issues).

When using bgflow in your research, please cite our preprint (coming soon).
***
## [Minimal example](#minimal-example)
Implementation of a BG with a single [Real NVP coupling block](https://arxiv.org/abs/1605.08803)
as the invertible transformation. The two-dimensional target potential is given by a double well potential in one
dimension and a harmonic potential in the other. The prior distribution is a two-dimensional standard normal
distribution. Note that the training procedure is not included in this example.

``` python
import torch
import matplotlib.pyplot as plt
import bgflow as bg

# define prior and target
dim = 2
prior = bg.NormalDistribution(dim)
target = bg.DoubleWellEnergy(dim)

# here we aggregate all layers of the flow
layers = []
layers.append(bg.SplitFlow(dim // 2))
layers.append(bg.CouplingFlow(
        # we use a affine transformation to transform 
        # the RHS conditioned on the LHS
        bg.AffineTransformer(
            # use simple dense nets for the affine shift/scale
            shift_transformation=bg.DenseNet(
                [dim // 2, 4, dim // 2], 
                activation=torch.nn.ReLU()
            ), 
            scale_transformation=bg.DenseNet(
                [dim // 2, 4, dim // 2], 
                activation=torch.nn.Tanh()
            )
        )
))
layers.append(bg.InverseFlow(bg.SplitFlow(dim // 2)))
    
# now define the flow as a sequence of all operations stored in layers
flow = bg.SequentialFlow(layers)

# The BG is defined by a prior, target and a flow
generator = bg.BoltzmannGenerator(prior, flow, target)

# sample from the BG
samples = generator.sample(1000)
plt.hist2d(
    samples[:, 0].detach().numpy(), 
    samples[:, 1].detach().numpy(), bins=100
)
```

***

## [Examples](#examples)

* [Basic Boltzmann Generator example](https://github.com/noegroup/bgflow/blob/master/notebooks/example.ipynb)
* [Training a Boltzmann Generator for Alanine Dipeptide](https://github.com/noegroup/bgflow/blob/master/notebooks/alanine_dipeptide_basics.ipynb)
* [Equivariant kernel flow example](https://github.com/noegroup/bgflow/blob/master/notebooks/example_equivariant_nODE.ipynb)
* [Equivariant Real NVP flow example](https://github.com/noegroup/bgflow/blob/master/notebooks/example_equivariant_RNVP.ipynb)

***

## [Installation](#installation)


* Clone this repository from github
* Navigate to the cloned repository
* Run the installation scrip

```
python setup.py install
```

* Install all required [dependencies](#dependencies) 
* Validate your installation by running all tests in the repository with the command

```
pytest
```

* Depending on the optional installations some tests might be skipped. 

***
## [Dependencies](#dependencies)
* Mandatory
  * [einops](https://github.com/arogozhnikov/einops/)
  * [pytorch](https://github.com/pytorch/pytorch)
  * [numpy](https://github.com/numpy/numpy)
* Optional
  * [matplotlib](https://github.com/matplotlib/matplotlib)
  * [pytest](https://github.com/pytest-dev/pytest) (for testing)
  * [nflows](https://github.com/bayesiains/nflows) (for Neural Spline Flows)
  * [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (for neural ODEs)
  * [ANODE](https://github.com/amirgholami/anode) (for neural ODEs)
  * [OpenMM](https://github.com/openmm/openmm) (for molecular mechanics energies)
  * [ase](https://wiki.fysik.dtu.dk/ase/index.html) (for quantum and molecular mechanics energies through the atomic simulation environment)
  * [xtb-python](https://xtb-python.readthedocs.io) (for semi-empirical GFN quantum energies)
  * [netCDF4](https://unidata.github.io/netcdf4-python/) (for the `ReplayBufferReporter`)
  * [jax](https://github.com/google/jax) (for smooth flows / implicit backprop)
  * [jax2torch](https://github.com/lucidrains/jax2torch) (for smooth flows / implicit backprop)
  * [allegro](https://github.com/mir-group/allegro) (for Graph Neural Networks)
  * [nequip](https://github.com/mir-group/nequip) (for Graph Neural Networks)
  * [bgmol](https://github.com/noegroup/bgmol) (for some example notebooks)


***
## [License](#dependencies)
[MIT License](LICENSE)
