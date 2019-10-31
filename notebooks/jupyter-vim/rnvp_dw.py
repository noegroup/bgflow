## ### import default modules
import torch
import numpy as np
import matplotlib.pyplot as plt

## ### fix random seeds
SEED = 0
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

## ### nice notebooks
import importlib
from tqdm import tqdm

if importlib.util.find_spec("darkplot") is not None:
    from darkplot import darkplot

    darkplot()

## ### import bgtorch modules

from bgtorch.flow import DiscreteFlow
from bgtorch.invertible import (
    AffineLayer,
    AffineTransformer,
    RealNVP,
    InvertedLayer,
    SplittingLayer,
    SwappingLayer,
)
from bgtorch.nn import DenseNet
from bgtorch.train import CombinedTrainer
from bgtorch.systems import HarmonicOscillator, DoubleWell
from bgtorch.sampling import SystemWrapper, MetropolisGauss

## ### global variables

N_DIMS = 2
N_MCMC_SAMPLES = 3000
N_RNVP_LAYERS = 4
DENSE_UNITS = [1, 32, 32, 2]
ACTIVATION = torch.nn.Tanh()
DT = 1.0 / N_RNVP_LAYERS
N_ML_SAMPLES = 64
N_KL_SAMPLES = 1000
N_EPOCHS = 10
N_PLOT_SAMPLES = 10000

## ### define a prior and a target energy

prior = HarmonicOscillator(N_DIMS)
target = DoubleWell()

## ### sample from the target energy

sampler = MetropolisGauss(
    SystemWrapper(target), np.random.normal(size=(1, N_DIMS)), burnin=1000
)
sampler.run(N_MCMC_SAMPLES)
data = torch.Tensor(sampler.traj)

## ### setup normalizing flow

layers = []
layers.append(AffineLayer(N_DIMS))
layers.append(SplittingLayer())
for _ in range(N_RNVP_LAYERS):
    layers.append(SwappingLayer())
    transformer = AffineTransformer(DenseNet(DENSE_UNITS, ACTIVATION))
    rnvp = RealNVP(transformer)
    layers.append(rnvp)
layers.append(InvertedLayer(SplittingLayer()))
flow = DiscreteFlow(layers)

## ### setup optimizer

optimizer = torch.optim.Adam(flow.parameters(), lr=5e-3)

## ### setup trainer

trainer = CombinedTrainer(flow, data, prior, target, optimizer)

## ### training loop (ML only)

kl_ratio = 0.0
nlls = []
klls = []
for epoch in tqdm(range(N_EPOCHS), desc="epoch"):
    epoch_iter = trainer.train_epoch(N_ML_SAMPLES, N_KL_SAMPLES, kl_ratio)
    for it, (nll, kll) in tqdm(
        enumerate(epoch_iter), desc="iteration", bar_format=None, total=len(epoch_iter)
    ):
        if nll is not None:
            nll = nll.detach().cpu().numpy()
            nlls.append(nll)
        if kll is not None:
            kll = kll.detach().cpu().numpy()
            klls.append(kll)
nlls = np.array(nlls)
plt.figure(figsize=(4, 4))
plt.plot(nlls)
plt.xlabel("Iteration")
plt.ylabel("NLL")
plt.tight_layout()


## ### plot result (ML)

z = prior.sample((N_PLOT_SAMPLES,))
x, _ = flow(z)
z_, _ = flow(data, inverse=True)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("config space")
plt.scatter(*data.detach().numpy().T, marker="x", alpha=0.05, color="g", label="data")
plt.scatter(*x.detach().numpy().T, marker="x", alpha=0.05, color="r", label="samples")
legend = plt.legend()
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.subplot(1, 2, 2)
plt.title("latent space")
plt.scatter(*z.detach().numpy().T, marker="x", alpha=0.05, color="g", label="samples")
plt.scatter(*z_.detach().numpy().T, marker="x", alpha=0.05, color="r", label="data")
legend = plt.legend()
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.show()

## ### training loop (ML+KL)

kl_ratios = np.linspace(0.0, 1.0, N_EPOCHS)
nlls = []
klls = []
for epoch, kl_ratio in tqdm(enumerate(kl_ratios), desc="epoch", total=N_EPOCHS):
    epoch_iter = trainer.train_epoch(N_ML_SAMPLES, N_KL_SAMPLES, kl_ratio)
    for it, (nll, kll) in tqdm(
        enumerate(epoch_iter), desc="iteration", bar_format=None, total=len(epoch_iter)
    ):
        if nll is not None:
            nll = nll.detach().cpu().numpy()
            nlls.append(nll)
        if kll is not None:
            kll = kll.detach().cpu().numpy()
            klls.append(kll)
nlls = np.array(nlls)
klls = np.array(klls)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(nlls)
plt.xlabel("Iteration")
plt.ylabel("NLL")
plt.subplot(1, 2, 2)
plt.plot(klls)
plt.xlabel("Iteration")
plt.ylabel("KLL")
plt.tight_layout()

## ### plot result (ML + KL)

z = prior.sample((N_PLOT_SAMPLES,))
x, _ = flow(z)
z_, _ = flow(data, inverse=True)
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title("config space")
plt.scatter(*data.detach().numpy().T, marker="x", alpha=0.05, color="g", label="data")
plt.scatter(*x.detach().numpy().T, marker="x", alpha=0.05, color="r", label="samples")
legend = plt.legend()
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.subplot(1, 2, 2)
plt.title("latent space")
plt.scatter(*z.detach().numpy().T, marker="x", alpha=0.05, color="g", label="samples")
plt.scatter(*z_.detach().numpy().T, marker="x", alpha=0.05, color="r", label="data")
legend = plt.legend()
for lh in legend.legendHandles:
    lh.set_alpha(1)
plt.show()
