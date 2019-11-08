# API Whitepaper

This whitepaper proposes an API for the Boltzmann generator library.

The library should be structured into two repositories.

1. `bgtorch`: this package will contain al code involving the _algorithmic_ aspects, such as
    - api
    - architecture components
    - training
    - sampling
2. `bgsystems`: this package will contain all code involving _concrete target systems_

Rationale behind this design decision is, that

1. will be changed whenever new algorithmic designs are added and evaluated against fixed systems.
2. will be changed whenever working code is run against a new system, or new systems are set up.

Otherwise every new application using established algorithms will result in new commit messages and bloat the repository
possibly quickly.

After explaining the design for both repositories a proposed general design / style guidline for the project is
proposed.

## Repository `bgtorch`

This library will contain the essentials that make up Boltzmann generators:

1. invertible layers
2. samplers
3. loss functions and training routines
4. api for `bgsystems`
5. utility functions (e.g. computing all-distance matrices etc.)

__NOTE__: It might be worth to just implement the API from
[`torch.distributions`](https://pytorch.org/docs/stable/distributions.html) and augment it with our requirements. This
would a) give us many priors b) allow us to integrate new contributions to PyTorch easily and c) use their tested code.
Especially their API for `Constraints` and `Transforms` could be useful to integrate.

### Modules

- nn
    - invertible
        - transformer
        - dynamics
- systems
- sampling
- train
    - loss
- util
- api
- validation

#### Module `bgtorch.nn`

This will contain all neural network components that are structured as `torch.nn.Module` objects and stacked to form a
differentiable computation graph.

An important primitve that should be defined here is a `DenseNet` that allows to construct dense neural network blocks
with arbitrary hidden layers and activations in one call (see example below).

#### Module `bgtorch.nn.invertible`

This will contain all components that can be used to build diffeomorphisms and thus be used as components within a flow.
A possible API that fits all possible flows is given by

```python
    class Invertible(torch.nn.Module):

        def __init__(self, ...):
            ...

        def forward(self, *xs, inverse=False):
            ....
            return *zs, dlogp
```
Here `xs` and `zs` are iterables of PyTorch tensors with matching dimensions and `dlogp` expresses the difference of the
log probability induced by the transformation. This is identical to the log determinant of the Jacobian for a discrete
flow. In order to keep things consistent, we should have `dlogp` of shape `[n_batch, 1]`.

The boolean flag indicates the direction of the flow. Using the flag does not break the `torch.nn.Module`
API where `forward` is automatically called when `__call__()` is invoked and easily allows complex compositionality
compared to seperated `forward`/`inverse` functions. Furthermore it encourages code reuse.

The API should later be usable in form like this

```python
from bgtorch.nn.invertible import (
    AffineLayer,
    SplittingLayer,
    ShufflingLayer,
    CouplingLayer,
    DiffEqFlow,
    SequentialFlow
)
from bgtorch.nn.invertible.transformer import (
    AffineTransformer
)
from bgtorch.nn.invertible.dynamics import (
    SimpleDynamics
)

# start with an affine layer
layers = [AffineLayer(dimension)]

# split into left and right dimensions used
# within a flow of coupling blocks
layers.append(SplittingLayer(n_left=dimension // 2))

# add some coupling blocks
for _ in range(n_real_nvp_layers):
    # shuffle left and right blocks
    permutation = torch.cat(
        torch.range(dimension // 2, dimension),
        torch.range(0, dimension)
    )
    layers.append(ShufflingLayer(permutation))

    # a RealNVP layer is a coupling layer with an
    # affine transformer
    transformer = AffineTransformer(
        shift_transformation = DenseNet(
            n_units=[dimension, 128, dimension],
            activation=[torch.nn.Tanh(), torch.nn.ReLU()]
        ),
        scale_transformation = DenseNet(
            n_units=[dimension, 128, dimension],
            activation=torch.nn.Tanh()
        )
    )
    layers.append(CouplingLayer(transformer=AffineTransformer))


# merge left and right dimensions back into one tensor
layers.append(
    InvertedLayer(
        SplittingLayer(n_left=dimension // 2)
    )
)

# add a continuous flow afterwards using
# a simple dynamics function that relies
# on the hutchinson estimator
layers.append(
    DiffEqFlow(
        dynamics=SimpleDynamics(
            dynamics_function=DenseNet(
                n_units=[dimension, 128, dimension],
                activation=torch.nn.Tanh()
            ),
            trace_estimator=HutchinsonEstimator()
        ),
        max_time=1.,
        integrator="dopri5",
        ...
    )
)

# combine the layers into a seqential flow
flow = SequentialFlow(layers)

# use the flow
x, dlogp = flow(z, inverse=False)
z, dlogp = flow(x, inverse=True)
```

Such a low-level API can be further wrapped e.g. in the notebook api, such that commonly used flows (e.g. simple RealNVP
blocks with affine transformers based on dense nets etc.) can be constructed using one function call. But such simple
factory functions should not become part of the low-level API. This example is given in full verbosity to highlight the
flexibility and readability benefit of using the delegate pattern for this application. Many common details (`n_left` in
the `SplittingLayer` constructor, `permutation` in the `ShufflingLayer` constructor) can be set via default parameters.

#### Module `bgtorch.nn.invertible.transformer`

A transformer is a conditional invertible transformation. Given a pair `x` (an iterable of tensors) and `y` it will transform `y` into `y' =
f(y, x)` where `f(..., x)` is an invertible function once conditioned on `x`. This can be seen as an autoregressive
invertible flow. RealNVP is now just a coupling layer (there is a "left" and a "right" split) where the "right" part is
transformed conditioned on the "left" part using an affine transformer.

The general API for such a transformer must look like

```python

    class Transformer(torch.nn.Module):

        def __init__(self, ...):
            ...

        def forward(self, x, y, inverse=False):
            # transform y into y_new conditioned on x
            # and compute the change in log density
            y_new, dlogp = ...
            return y_new, dlogp
```

This can probably be even more generalized (and thus be reused) if considered as an autoregressive transformation.

#### Module `bgtorch.nn.invertible.dynamics`

Dynamics functions used in a continuous (normalizing) flow can have a general form and do not need to be invertible.
However, in order to be useful, they should provide a way to compute their divergence in an effective way. A general API
should look like this

```python

class Dynamics(torch.nn.Module):

    def __init__(self, ...):
        ...

    def forward(self, x, compute_divergence=True):
        # if `compute_divergence` is `True`
        # compute the vector field and divergence
        ...
        return dx, divergence

        # if `compute_divergence` is `False` just compute the vector field
        return dx, None
```

#### Module `bgtorch.systems`

This will contain an API for systems (energies / distributions), which then can be used as priors (Gaussian
distributions) or as targets (Double-Well, Lennard-Jones, Alanine Dipeptide, ...).

A minimal API should support

```python
from bgtorch.util import compute_force

class System(torch.nn.Module):

    def __init__(self, dimension):
        super().__init__()
        self._dimension
        ...

    @property
    def dimension(self):
        return self._dimension

    def _sample_iid(self, n_samples, temperature=1.0):
        # if possible (e.g. for gaussians / mixture model)
        # implement something that allows i.i.d. sampling.
        raise NotImplementedError()

    def _log_prob(self, state):
        # if possible (e.g. for gaussians)
        # implement the exact normalized log probability
        raise NotImplementedError()

    def energy(self, state):
        # compute the energy / unnormalized log prob for the state
        # if a log probability is given and tractable, we can just
        # use this. otherwise should be overrideen by child class
        return self._log_prob(state)

    def force(self, state):
        # possibly inefficient generic implementation
        # that can be overridden by child class
        return compute_force(self.energy, state)

```


In order to keep things consistent the energy should be of shape `[n_batch, 1]`.

Standard toy systems (Muller potential, DW), standard priors (Gaussian) and should be directly implemented here. It is
further possible to use compositionality here as well:

1. Mixture models can be composed from arbitrary systems.
2. A prior together with a flow gives a new `TransformedSystem`. Basically every boltzmann generator can be considered
   as such.

Possible implementations for these two systems could look like

```python
from bgtorch.utils import to_numpy

class MixtureModel(System):

    def __init__(
        self,
        components,
        unnormalized_log_weights=None,
        trainable_weights=False
    ):
        super().__init__()
        n_components = len(components)
        self._components = components
        if unnormalized_log_weights is None:
            unnormalized_log_weights = torch.zeros(
                n_components
            )
        self._unnormalized_log_weights = unnormalized_log_weights
        if trainable_weights:
            self._unnormalized_log_weights = torch.nn.Parameter(
                self._unnormalized_log_weights
            )

    @property
    def _log_weights(self):
        return torch.log_softmax(
            self._unnormalized_log_weights, dim=-1
        )

    def _sample_iid(self, n_total_samples, temperature=1.0):
        # yet not clear how to incorporate temperature in
        # a mixture model appropriately
        assert temperature == 1.0

        weights = self._log_weights.exp()
        weights_numpy = to_numpy(log_weights.exp())

        n_samples_per_comp = np.random.multinomial(
            n_total_samples, weights_numpy, 1
        )

        samples = [
            component._sample_iid(
                    n_samples,
                    temperature
                )
            for n_sample, component in zip(
                n_samples_per_comp,
                self._components
            )
        ]

        return torch.cat(samples, dim=0)

    def log_prob(self, state):
        log_probs = [
            component.log_prob()
            for component in self._components
        ]
        log_weights = self._weights.log()
        return torch.logsumexp(
            log_weights + log_probs,
            dim=-1,
            keepdim=True
        )
```

```python
class TransformedSystem(System):

    def __init(self, prior, flow):
        self._prior = prior
        self._flow = flow

    def _sample_iid(self, n_samples):
        samples_prior = self._prior._sample_iid(n_samples)
        samples, _ = self.flow(samples_prior)
        return samples

    def log_prob(self, state):
        transformed_state, dlogp = self._flow(state, inverse=True)
        return self._prior.log_prob(transformed_state) + dlogp

    def energy(self, state):
        transformed_state, dlogp = self._flow(state, inverse=True)
        return self._prior.energy(transformed_state) + dlogp
```

This abstraction would allow to define complex priors and targets in the same language making reusability and code much
simpler.

That way we can define a mixture flow as

```pythone
    # define a list of priors
    priors = [ ... ]

    # construct a list of individual flows following `bgtorch.invertible`
    flows = [ ... ]

    # new target distribution
    proposal_distribution = MixtureModel([
        TransformedSystem(prior, flow)
        for prior, flow in zip(priors, flows)
    ])
```

Target systems like Ala2, BPTI, ... are set up using OpenMM (or possibly other external libraries in the future). Here
`bgtorch.systems` should only support a generic `OpenMMSystem` object, that e.g. takes a pre-defined OpenMM context (which could fully
specify Ala2) and wraps OpenMM to PyTorch.

Concrete instatiations of OpenMM (or other third-party) systems should be specified in the `bgsystems` repository
(details given in respective section of this whitepaper).

#### Module `bgtorch.sampling`

This will contain all possible samplers. A sampler should implement the API

```python
class Sampler():

    def sample(self, n_samples):
        ...
        return samples

```

A generic sampler for a system is given by

```python
class SystemSampler():

    def __init__(self, system, temperature=1.):
        self._system = system
        self._temperature = temperature

    def sample(self, n_samples):
        raise NotImplementedError()
```

Two samplers that should exist are

```python
class IIDSampler(SystemSampler):

    def sample(self, n_samples):
        return self._system._sample_iid(n_samples, self._temperature)
```

```python
class _RandomizedIndexBuffer():

    def __init__(self, n_elements):
        # just a simple ring buffer which iterates
        # through a shuffled set of indices
        ...

    def query(self, n_samples):
        # get the next batch of `n_samples` indices
        # of the shuffled range. when the end is reached
        # shuffle all indices and start over
        ...

class EmpiricalSampler():

    def __init__(self, data):
        self._data = data
        self._index_buffer = _RandomizedIndexBuffer(len(data))

    def sample(self, n_samples):
        idxs = self._index_buffer.query(n_samples)
        return self._data[idxs]
```

The module should further contain all helper functions needed for reweighing etc.

__NOTE:__ this section is the most incomplete yet - any good suggestion is very welcome here!

#### Module `bgtorch.train`

This will contain everything necessary for training the flows. As training will be something that will be extended
heavily (there are ML loss, KL loss, RC loss, possibly adversarial losses in the future...) this should be kept flexible
as well.

A useful abstraction would be using stateful iterators which emit a loss at each iteration.

```python
class LossIterator():

    def __iter__(self):
        return self

    def __next__(self):
        raise NotImplementedError
```

A concrete example is a loss iterator that emits the KL divergence between an origin and a target distribution.

``` python
def _unnormalized_kl_divergence(flow, target, x, inverse=false):
    z, dlogp = flow(x, inverse=inverse)
    kl_divergence = target.energy(z).view(-1) - dlogp.view(-1)
    return kl_divergence

class KLLossIterator(LossIterator):

    def __init__(
        self,
        n_samples,
        sampler,
        target,
        flow,
        inverse=False
    ):
        self._n_samples = n_samples
        self._sampler = sampler
        self._target = target
        self._flow = flow
        self._inverse = inverse

    def __next__(self):
        samples = self._sampler.sample(self._n_samples)
        return _unnormalized_kl_divergence(
            self._flow,
            self._target,
            samples,
            inverse=self._inverse
        )
```

If we have a target system `target`, a flow `flow` and a prior `prior`, we can now define two loss iterators
that represent our usual ML and KL losses:

```
from bgtorch.sampling import DataSampler

ml_loss_iter = KLLossIterator(
    n_samples=n_ml_samples,
    sampler=DataSampler(data)
    target=prior,
    flow=flow,
    inverse=True
)

kl_loss_iter = KLLossIterator(
    n_samples=n_kl_samples,
    sampler=IIDSampler(prior),
    target=target,
    flow=flow,
    inverse=False
)
```

This is a flexible design, e.g. if our prior is not given by a distribution that is accessible via iid sampling we could
also define the following for the kl loss

```
from bgtorch.sampling import GaussianMCMCSampler

kl_loss_iter = KLLossIterator(
    n_samples=n_kl_samples,
    sampler=GaussianMCMCSampler(prior),
    target=target,
    flow=flow,
    inverse=False
)
```

This design allows us to easily add new losses (RC loss, adversarial losses, ...).

We can now combine these losses using a gradient accumulator. Simply summing and calling `backward()` will break down in
the case of continuous flows, as you cannot compute gradients for the forward and backward flow simultaneously und thus this would break this API. Thus aggregating gradients for each loss individually is the
more robust option.

A possible way is using an epoch iterator

```python
class TrainingEpochIterator():

    def __init__(
        self,
        n_iterations,
        loss_iterators,
        optimizer,
        unnormalized_weights=None
    ):
        n_iterators = len(loss_iterators)
        self._step_iter = range(n_iterations)
        self._loss_iters = zip(loss_iterators)
        self._optimizer = optimizer
        if unnormalized_weights is None:
            unnormalized_weights = torch.ones(n_iterators)
        self._unnormalized_weights = unnormalized_weights

    @property
    def _weights:
        return torch.softmax(
            self._unnormalized_weights,
            dim=-1
        )

    def __iter__(self):
        return self

    def __next__(self):
        step = next(self._step_iter)
        losses = next(self._loss_iters)

        total_loss = 0.

        self._optimizer.zero_grad()

        for weight, loss in zip(self._weights, losses):
            weighted_loss = weight * loss
            weighted_loss.backward()
            total_loss += weighted_loss

        self._optimizer.step()

        return total_loss, losses
```

Finally, we can use it to train one epoch e.g. like this

```python
def train_epoch(
    n_iterations,
    loss_iters,
    optimizer,
    weights=None,
    reporter=None
):
    train_iter = TrainingEpochIterator(
        n_epoch_iterations,
        loss_iterators=loss_iters,
        optimizer=optimizer,
        weights=weights
    )
    for total_loss, losses in train_iter:
        if report is not None:
            reporter.report(total_loss, losses)
```

And use this function to define arbitrarily complex training routines

```python
loss_iters = [
    ml_loss_iter,
    kl_loss_iter
]

reporter = LossReporter(n_reported=len(loss_iters))

learning_rates = np.linspace(5e-3, 5e-5, n_epochs)
kl_loss_weights = np.linspace(0., 1., n_epochs)

for epoch, learning_rate, kl_loss_weight in zip(
    range(n_epochs),
    learning_rates,
    kl_loss_weights
):
    # as discussed in https://discuss.pytorch.org/t/adaptive-learning-rate/320/6
    # creating new optimizers is better than fiddling with the learning rate
    optimizer = torch.optim.Adam(flow.parameter(), lr=learning_rate)

    # weigh ML and KL loss
    weights = [1. - kl_loss_weight, kl_loss_weight]

    # train one epoch
    train_epoch(n_iterations, loss_iters, optimizer, weights, reporter)

    # do something with the reporter (plot losses etc.)
    ...
```

#### Module `bgtorch.util`

This module will just contain all helper functions that are not tied to a specific object in the API. This can involve
things like shape operations, autograd operations, type checking etc.

It could make sense to structure this in submodules for specific problems.

#### Module `bgtorch.api`

This module will contain all wrapper functions that makes it easy to setup standard flows / systems and train them.

Each function here should use elements of the aforementioned API and plug them together for a concrete problem, but never re-implement non-trivial solutions.

For constructing flows a [builder patter](https://en.wikipedia.org/wiki/Builder_pattern) would be probably useful and
preferred over long nested error-prone and difficult to maintain dicts.

Similarily such a pattern could be used define a training routine containing the losses, learning rate scheduler, weight
scheduler etc.

To make a Boltzmann Generator usable like a normal numpy object it could be an option to have a wrapper class, that
after construction using the builder for the flow and the training procedure, can be simply trained using a `train()`
procedure and then used using `sample()`, `energy()`, `forward()`, `inverse()` etc. functions which are completely
transparent to the user (she/he will not need to work with jacobians etc.).

Furthermore, this module should contain all plotting functionality.

#### Module `bgtorch.validation`

Optional: here we could bundle all functions that are used to validate the quality of results for a given trained BG.

## Repository `bgsystem`

This library will contain concrete systems, like Alanine Dipeptide in implicit solvent or BPTI.

Each system consists of a Python class that
1. Yields the correct `bgtorch.system.System` object.
   This implies setting up OpenMM in the correct way, given the
   configuration parameters etc.
2. Generates the data from the system for a fixed seed.

There should be the following modules

1. `bgsystems.systems`
2. `bgsystems.util`

The first module will list all systems that can be loaded, the second module provides the loader functions to reproduce
the system.

### Module `bgsystems.systems`

Here we list all systems that we want to set up (e.g. complicated OpenMM systems). Each system should be given by a
class which implements an API like

```python

class BPTI()

    def __init__(self, ... possibly many params ...):
        ...
        # setup OpenMM correctly given the params
        # it is important that only JSON/YAML values are supported
        # this means:
        #   - numbers
        #   - boolean
        #   - strings
        # as well as lists and dicts build from them
        ...

    def get_system(self):
        ...
        # return a bgtorch.system.System from the setup
        return system

    def generate_data(self, seed, n_data):
        ...
        # generate data from the system given some sampling / MD method
        # a fixed seed is a mandatory parameter to guarantee reproducibility
        ...
        return data

```

### YAML system specification

For each system we can now freeze an instance that we want to reproduce 1:1 by saving it into a YAML file. Such a file
could look like this.

```yaml
system: "bgsystems.openmm.BTPI"
commit_hashes:
    bgsystems: git commit hash
    bgtorch: git commit hash
params:
    a: 1
    b: "test"
    c:
        c1: 42
        c2: "foo"
    d: [1, 5, 6, 9]
data:
    - seed: 42
      n_samples: 10000
      url: http://url/some.file
    - seed: 5
      n_samples: 500000
      url: http:///url/some.other.file
```

__NOTE__: These YAML files should probably be maintained in a separate repository to avoid conflicts when checking out
specific states of the repositories.

### Reproducing a system state

Reproducing results is now straightforward:

1.  Checkout `bgsystems` / `bgtorch` with the specified commit hashes. (can be automated using a script)
```bash
$ ./checkout_repositories.sh path/to/bgtorch path/to/bgsystems path/to/file.yaml
```
2. Load the system and data in your notebook / batch code

```python
from bgsystems.util import YAMLLoader

loader = YAMLLoader("path/to/file.yaml")

seeds = loader.frozen_seeds()

# pick one of those stored ones
seed = seeds[13]

system = loader.get_system(seed)
data = loader.get_data(seed)

# now do whatever you want to do with the reproduced state
```

## General design / style guidelines

To maintain consistency while still allowing for flexibility and to maximize code reuse

1. Compositions should be preferred over inheritance:

```python
class Specialized():

    def __init__(self, delegate):
        self._delegate = delegate

    def foo(self, arg):
        # do something
        result = self._delegate(arg)
        # do something else with `result test`
```

Python is a duck-typed language, thus polymorphism does not require inheritance compared to statically typed languages like C++ / Java.
Inheritance can result in very complicated dependencies and [semantic paradoxes](https://en.wikipedia.org/wiki/Multiple_inheritance).
Delegates do not have this problem and do not possess any performance drawbacks.
Use inheritance when a single member function is the same for all child classes. Do not use inheritance when overloading
in a child class will occur - this can be solved with a delegator.

2. Variables should have self-explanatory names. Avoid `x`, `mu`, `tau` etc. Try to use `samples`, `temperature`, `mean` etc.
   instead. Python encourages the use of explanatory variable names by providing a very flexible formatting.
3. Private members that are not to be intended to be accessible from outside should be prefixed with an underscore, e.g.
   `self._private_variable` or `self._private_function()`. Object states (variables) which should be accessible from
   outside should be revealed with a `@property` function. Changes to object states should always happen via procedure /
   function calls (in the best case they should not happen - mutable object states are in many cases the cause for
   headache). Private functions make sense whenever (some code is reused in more than one other member function __or__
   a long function can be broken into semantic chunks) __and__ the function must access `self`. Otherwise a module
   private function in the same file is preferred which are not exported in the `__init__.py`.

```python
def _some_helper(arg):
    # do something

class Object():

    def __init__(self, foo, bar):
        self._foo = foo
        self._bar = bar

    def _some_private_procedure(self, arg1, arg2):
        # do something that is used in multiple
        # procedures of this class
        # and requires access to `self`

    def blub(self, arg):
        # something
        result = _some_helper(arg)
        # something
        self._some_private_procedure(self._foo, result)
        # something

    def bla(self, arg):
        # something
        self._some_private_procedure(arg, self._bar)
        # something

    @property
    def foo(self):
        return self._foo

```

4. Functions longer than 30 lines should be broken into semantically meaningful subroutines. It should be possible to
   understand the code without the need to scroll. This will improve maintainable and readability of the code
   tremendously.

5. Flexibility should come from architecture design, not from monstrous configuration dicts. Try to avoid configuration
   dicts wherever possible. Dicts should be used as random access, hash-based containers (like `std::map` in C++) but
   never substitute a flexible api. Possible patterns that allow avoiding dicts
   - Delegate patterns (see 1.)
   - Factory patterns
   - Command patterns
   - Default parameters / parameter overloading

6. No hard-coding of specific design decisionsn the API. The API should allow to build every possible instantiation of
   the matheamtical framework. Concrete instatiations (e.g. a specific design for the scale/shift transformations of a
   RealNVP layer) must be specified in a notebook. It makes sense to provide a notebook API for shortcuts, but this
   should be strictly seperated.

7. Classes that do not directly depend on each other and serve more orthogonal purposes should be structured in
   individual files. Helper classes and functions should be in the same file as the major class they support. Classes
   that are used in the API should be exported in the `__init__.py` from each invidual python file for transparency and
   to avoid lengthy import names.

8. Code should be [blackened](https://github.com/psf/black). Avoid [train
   wrecks](https://en.wikipedia.org/wiki/Method_chaining) (a common problem with PyTorch) and try to split intermediary results in meaningfully named
   variables. This will improve code readability tremendously.

9. Besides docstrings code should be self-documentary (variable naming, breaking into clearly named subroutines
   etc.). Whenever things can not meaningfully named, broken into simpler chunks or just use some "smart trick" an
   inline comment explaining the rationale is important. It has been proven to be useful to indicate the variable shapes
   in inline comments if non-trivial shape transformations are conducted.

```python
def some_nontrivial_manipulation(foo, bar):
    ''' Some short doc string.

        Some lengthy explanation with an equation

        `x = x^2`

        which can span multiple lines.

        Refer to some method [1] or some implementation [2].

        [1] http://arxiv/link/to/reference
        [2] http://github/link/to/reference

        Parameters
        ----------
        foo: PyTorch float tensor.
             Some semantic description.
             Shape `[n_batch, n_particles, dimension]`.
        bar: boolean
             Some semantic description

        Returns
        -------
        bla: PyTorch double tensor.
             Some semantic description.
             if bar is `True`
                Some semantic descritpion.
                Shape `[n_batch, dimension]`.
            if bar is `False`
                Some other semantic description.
                Shape `[n_batch, n_new_particles, dimension]`.
    '''
    n_batch = foo.shape[0]
    n_particles = foo.shape[1]
    dimension = foo.shape[2]

    # [n_batch * n_particles, dimension]
    foo_flat = foo.view(n_batch * n_particles, dimension)

    ...
```
10. Every computation must be performed on fixed seeds. There is just too much source for errors otherwise and it makes
    reproducibility and bug-fixing impossible. This can be performed by

```python
def freeze_seed(seed=42):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
```

We should consider putting this as a utility function to the library.

