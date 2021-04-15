import torch
from bgflow.nn.flow.base import Flow

class MetropolisMCFlow(Flow):
    def __init__(self, energy_model, nsteps=1, stepsize=0.01):
        """ Stochastic Flow layer that simulates Metropolis Monte Carlo

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
        x' : PyTorch Tensor
            Transformed configurations
        dW : PyTorch Tensor
            Nonequilibrium work done, always 0 for this process
            
        """
        E0 = self.energy_model.energy(x)
        E = E0

        for i in range(self.nsteps):
            # proposal step
            dx = self.stepsize * torch.zeros_like(x).normal_()
            xprop = x + dx
            Eprop = self.energy_model.energy(xprop)
            
            # acceptance step
            acc = (torch.rand(x.shape[0], 1) < torch.exp(-(Eprop - E))).float()  # selection variable: 0 or 1.
            x = (1-acc) * x + acc * xprop
            E = (1-acc) * E + acc * Eprop

        # Work is energy difference
        dW = E - E0
        
        return x, dW

    def _inverse(self, x, **kwargs):
        """ Same as forward """
        return self._forward(x, **kwargs)
    