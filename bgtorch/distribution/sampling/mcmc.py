import torch
import numpy as np

from ..energy import Energy
from .base import Sampler


class GaussianMCMCSampler(Energy, Sampler):
    def __init__(
        self,
        energy,
        init_state=None,
        temperature=1.,
        noise_std=.1,
        n_stride=1,
        n_burnin=0
    ):
        super().__init__(energy.dim)
        self._energy_function = energy
        self._init_state = init_state
        self._temperature = temperature
        self._noise_std = noise_std
        self._n_stride = n_stride
        self._n_burnin = n_burnin
        self._reset(init_state)
        
    def _step(self):
        shape = self._x_curr.shape
        noise = self._noise_std * torch.Tensor(self._x_curr.shape).normal_()
        x_prop = self._x_curr + noise
        e_prop = self._energy_function.energy(x_prop, temperature=self._temperature)
        e_diff = e_prop - self._e_curr
        r = -torch.Tensor(x_prop.shape[0]).uniform_(0, 1).log()
        acc = (r > e_diff).float()
        rej = 1. - acc
        self._x_curr = rej * self._x_curr + acc * x_prop
        self._e_curr = rej * self._e_curr + acc * e_prop
        self._xs.append(self._x_curr)
        self._es.append(self._e_curr)
        
    def _reset(self, init_state):
        self._x_curr = self._init_state
        self._e_curr = self._energy_function.energy(self._x_curr, temperature=self._temperature)
        self._xs = [self._x_curr]
        self._es = [self._e_curr]
        self._run(self._n_burnin)
    
    def _run(self, n_steps):
        for i in range(n_steps):
            self._step()
    
    def _sample(self, n_samples):
        self._run(n_samples)
        return torch.cat(self._xs[-n_samples::self._n_stride], dim=0)
    
    def _energy(self, x):
        return self._energy_function.energy(x)