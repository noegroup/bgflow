import numpy as np
import torch

from bgtorch.nn.flow.base import Flow


class BrownianFlow(Flow):
    def __init__(self, energy_model, nsteps=1, stepsize=0.01):
        """ Stochastic Flow layer that simulates overdamped Langevin dynamics / Brownian dynamics

        """
        super().__init__()
        self.energy_model = energy_model
        self.nsteps = nsteps
        self.stepsize = stepsize

    def _forward(self, x, **kwargs):
        """ Run a stochastic trajectory forward

        Parameters
        ----------
        x : PyTorch Tensor
            Batch of input configurations

        Returns
        -------
        y: PyTorch Tensor
            Transformed input
        dW : PyTorch Tensor
            Nonequilibrium work done, log ratio of forward-backward path probabilities.
        """
        dW = torch.zeros((x.shape[0], 1))
        for i in range(self.nsteps):
            # forward noise
            self.w = torch.Tensor(x.shape).normal_()
            # forward step
            y = x + self.stepsize * self.energy_model.force(x) + np.sqrt(2*self.stepsize) * self.w
            # backward noise
            self.w_ = (x - y - self.stepsize * self.energy_model.force(y)) / np.sqrt(2*self.stepsize)
            # noise ratio
            dW += 0.5 * (self.w**2 - self.w_**2).sum(axis=1, keepdims=True)
            # update state
            x = y

        return x, dW

    def _inverse(self, x, **kwargs):
        """ Same as forward """
        return self._forward(x, **kwargs)

OverdampedLangevinFlow = BrownianFlow  # alias


class LangevinFlow(Flow):
    def __init__(self, energy_model, nsteps=1, stepsize=0.01, mass=1.0, gamma=1.0, kT=1.0):
        """ Stochastic Flow layer that simulates Langevin dynamics

        """
        super().__init__()
        self.energy_model = energy_model
        self.nsteps = nsteps
        self.stepsize = stepsize
        self.mass = mass
        self.gamma = gamma
        self.kT = kT
    
    def _forward(self, q, v, **kwargs):
        """ Run a stochastic trajectory forward 
        
        Parameters
        ----------
        q : PyTorch Tensor
            Batch of input configurations
        v : PyTorch Tensor
            Batch of input velocities
        
        Returns
        -------
        q' : PyTorch Tensor
            Transformed configurations
        v' : PyTorch Tensor
            Transformed velocities
        dW : PyTorch Tensor
            Nonequilibrium work done, log ratio of forward-backward path probabilities.
        """
        dW = torch.zeros((q.shape[0], 1))
        gamma_m = self.gamma*self.mass
        # naming convention: 1,h,2 timesteps. _: backward
        q1 = q
        v1 = v

        fac1 = np.sqrt(4.0 * gamma_m * self.kT / self.stepsize)
        fac2 = np.sqrt(gamma_m * self.stepsize / (self.kT))

        self.q111 = []
        for i in range(self.nsteps):
            self.q111.append(q1[0, 0])
            # forward noise
            w1 = torch.Tensor(q.shape).normal_()
            w2 = torch.Tensor(q.shape).normal_()
            # forward step
            vh = v1 + (self.stepsize / (2.0*self.mass)) * (self.energy_model.force(q1)
                                                           - gamma_m * v1
                                                           + fac1 * w1)
            q2 = q1 + self.stepsize * vh
            v2 = 1.0/(1.0 + self.gamma * self.stepsize / 2.0) * \
                    (vh + (self.stepsize / (2.0*self.mass)) * (self.energy_model.force(q2)
                                                               + fac1 * w2))
            # backward noises
            w1_ = w2 - fac2 * v2
            w2_ = w1 - fac2 * v1
            # noise ratio
            dW += 0.5 * (w1**2 + w2**2 - w1_**2 - w2_**2).sum(axis=1, keepdims=True)
            # update state
            q1 = q2
            v1 = v2
        
        return q1, v1, dW

    def _inverse(self, q, v, **kwargs):
        """ Same as forward """
        return self._forward(q, v, **kwargs)
    