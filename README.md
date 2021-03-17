# bgtorch

Bgtorch is a [pytorch](https://github.com/pytorch/pytorch) framework for
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
    * [Temperature-steerable flows](https://arxiv.org/abs/2012.00429)
    * [P4Inv Linear Flows](https://arxiv.org/abs/2010.07033)
    * [Neural Spline Flows](https://arxiv.org/abs/1906.04032)
    * [Augmented Normalizing Flows](https://arxiv.org/abs/2002.07101)
    * [Neural Spline Flows](https://arxiv.org/abs/1906.04032)
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
[pull request](https://github.com/noegroup/bgtorch/pulls)
or report an [issue](https://github.com/noegroup/bgtorch/issues).

When using bgtorch in you research, please cite our preprint (coming soon).
***
## [Minimal example](#minimal-example)
Implementation of a BG with a single [Real NVP coupling block](https://arxiv.org/abs/1605.08803)
as the invertible transformation. The two-dimensional target potential is given by a double well potential in one
dimension and a harmonic potential in the other. The prior distribution is a two-dimensional standard normal
distribution. Note that the training procedure is not included in this example.

``` python
import torch
import matplotlib.pyplot as plt
import bgtorch as bg

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

* [Basic Boltzmann Generator example](https://github.com/noegroup/bgtorch/blob/master/notebooks/example.ipynb)
* [Training a Boltzmann Generator for Alanine Dipeptide](https://github.com/noegroup/bgtorch/blob/master/notebooks/alanine_dipeptide_basics.ipynb)

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
  * [pytorch](https://github.com/pytorch/pytorch)
  * [numpy](https://github.com/numpy/numpy)
  * [matplotlib](https://github.com/matplotlib/matplotlib)
* Optional
  * [pytest](https://github.com/pytest-dev/pytest) (for testing)
  * [nflows](https://github.com/bayesiains/nflows) (for Neural Spline Flows)
  * [OpenMM](https://github.com/openmm/openmm) (for molecular examples)
  * [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (for neural ODEs)
  * [ANODE](https://github.com/amirgholami/anode) (for neural ODEs)

***
## [License](#dependencies)
[MIT License](LICENSE)