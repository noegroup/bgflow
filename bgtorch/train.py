import torch

from .utils.train import IndexBatchIterator


def _unnormalized_kl_divergence(flow, distribution, x, inverse=False):
    z, dlogp = flow(x, inverse=inverse)
    kl_divergence = distribution.energy(z).view(-1) - dlogp.view(-1)
    return kl_divergence


class CombinedTrainer(object):
    
    def __init__(self, flow, data, prior, target, optimizer):
        """
            Trainer class, which optimizes a flow to minimize
            
                1. the negative log-likelihood (NLL) on a provided data set.
                2. the reverse KL divergence (KLL) to a target distribution with accessible energy.
            
            The final minimized loss is 
            
                `(1-a) * NLL + a * KLL`
                
            where the trade-off parameter `a` is a value in `[0, 1]`.
                
            Training is performed per epoch, where in each epoch the trade-off parameter can be adjusted.
            
            Parameters:
            -----------
            flow : invertible flow object
                The flow which is optimized.
            data : PyTorch Tensor.
                The data set which is used for minimizing the NLL.
                Tensor of shape `[n_batch, n_dimensions]`.
            prior : Distribution
                The prior distribution of the flow.
                Must provide an implementation of the `energy` and the `sample` function.
            target : Distribution
                The target distribution of the flow.
                Must provide an implementation of the `energy` function.
            optimizer : torch.optim.Optimizer
                The optimizer used during optimization.
            
        """
        self._flow = flow
        self._data = data
        self._prior = prior
        self._target = target
        self._optim = optimizer

    def _aggregate_gradients(self, x, target, weight=1., inverse=False):
        loss = weight * _unnormalized_kl_divergence(self._flow, target, x, inverse=inverse).mean()
        loss.backward()
        return loss
    
    def _step(self, data, n_samples, ratio):
        self._optim.zero_grad()
        nll = kll = None
        if ratio < 1.:
            nll = self._aggregate_gradients(data, self._prior, 1. - ratio, inverse=True)
        if ratio > 0.:
            samples = self._prior.sample((n_samples,))
            kll = self._aggregate_gradients(samples, self._target, ratio, inverse=False)
        self._optim.step()
        return nll, kll
    
    def train_epoch(self, n_batch, n_samples=None, kl_ratio=0.):
        """
            Train for one epoch (= one iteration through the data set).
            
            Parameters:
            -----------
            n_batch : Integer
                Batch size used for the batch iteration through the data set.
            n_samples : Integer or None
                Number of samples drawn from the prior distribution for computing the KLL loss.
                If set to None will be set to `n_batch`
            kl_ratio : Float between 0 and 1.
                The trade-off parameter `a` weighing NLL loss vs. KLL loss.
        """
        batch_iterator = IndexBatchIterator(len(self._data), n_batch)
        nlls = []
        klls = []
        if n_samples is None:
            n_samples = n_batch
        for idxs in batch_iterator:
            if kl_ratio < 1.:
                data = self._data[idxs]
                if not isinstance(data, torch.Tensor):
                    data = torch.Tensor(data)
            else:
                data = None
            nll, kll = self._step(data, n_samples, kl_ratio)
            yield nll, kll
            
            
            
                           