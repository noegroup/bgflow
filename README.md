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
* Different invertible [normalizing flow]() structures to build BGs
    * [Coupling flows]()
    * [Neural ODEs]()
    * [Stochastic flows]()
    * [Equivariant flows]()
    * [Temperature-steerable flows]()
    * [Neural spline flows]()
    * [Augmented normalizing flows]()
* Other sampling methods
    * Markov-Chain Monte Carlo
    * Molecular dynamics
    * Replica exchange
* API to combine BGs with other sampling methods
* [OpenMM](https://github.com/openmm/openmm) bridge 

***
## [Minimal example](#minimal-example)
Implementation of a BG with a single [Real NVP coupling block](https://arxiv.org/abs/1605.08803)
as the invertible transformation.
The target potential is given by a double well potential in one dimension and a standard Gaussian in the other. 
The prior is a two-dimensional standard Gaussian distribution. 
The training procedure is not included. 

``` python
import torch
import matplotlib.pyplot as plt
from bgtorch import BoltzmannGenerator
from bgtorch.distribution import NormalDistribution
from bgtorch.distribution.energy import DoubleWellEnergy
from bgtorch.nn import DenseNet
from bgtorch.nn.flow import (SplitFlow, InverseFlow, SequentialFlow, CouplingFlow)
from bgtorch.nn.flow.transformer import AffineTransformer 

# define prior and target
dim = 2
prior = NormalDistribution(dim)
target = DoubleWellEnergy(dim)

# here we aggregate all layers of the flow
layers = []
layers.append(SplitFlow(dim // 2))
layers.append(CouplingFlow(
        # we use a affine transformation to transform the RHS conditioned on the LHS
        AffineTransformer(
            # use simple dense nets for the affine shift/scale
            shift_transformation=DenseNet([dim // 2, 4, dim // 2], activation=torch.nn.ReLU()), 
            scale_transformation=DenseNet([dim // 2, 4, dim // 2], activation=torch.nn.Tanh())
        )
))
layers.append(InverseFlow(SplitFlow(dim // 2)))
    
# now define the flow as a sequence of all operations stored in layers
flow = SequentialFlow(layers)

# The BG is defined by a prior, target and a flow
bg = BoltzmannGenerator(prior, flow, target)

# sample from the BG
samples = bg.sample(1000)
plt.hist2d(samples[:, 0].detach().numpy(), samples[:, 1].detach().numpy(), bins=100);
```


***
## [Examples](#examples)
* [Coupling flow example](https://github.com/noegroup/bgtorch/blob/master/notebooks/example.ipynb)
* [neural ODE example](https://github.com/noegroup/bgtorch/blob/notebooks/example_black_box_nODE.ipynb) TODO
* [equivariant neural ODE example](https://github.com/noegroup/bgtorch/blob/notebooks/example_equivariant_nODE.ipynb) TODO
* TODO




***
## [Installation](#installation)
* Clone this repository from github
* Navigate to the cloned repository
* Run the installation scrip
  
```
python setup.py install
```

* Install all required [dependencies](#dependencies) <-- TODO requirements.txt
* Validate your installation by running all tests

```
pytest or python setup.py test <-- TODO
```

***
## [Dependencies](#dependencies)

* [pytorch](https://github.com/pytorch/pytorch)
* [numpy](https://github.com/numpy/numpy)
* [pytest](https://github.com/pytest-dev/pytest)
* [matplotlib](https://github.com/matplotlib/matplotlib)
* [openMM](https://github.com/openmm/openmm) (for)
* [torchdiffeq](https://github.com/rtqichen/torchdiffeq) (for neural ODEs)
* [ANODE](https://github.com/amirgholami/anode) (for neural ODEs)

***
## [License](#dependencies)
TODO