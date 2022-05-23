import torch
import numpy as np
from tqdm import tqdm
from ipdb import set_trace as bp
from bgflow.utils.types import assert_numpy
from bgflow.distribution.sampling import DataSetSampler


__all__ = ["LossReporter", "KLTrainer"]


class LossReporter:
    """
    Simple reporter use for reporting losses and plotting them.
    """

    def __init__(self, *labels):
        self._labels = labels
        self._n_reported = len(labels)
        self._raw = [[] for _ in range(self._n_reported)]

    def report(self, *losses):
        assert len(losses) == self._n_reported
        for i in range(self._n_reported):
            self._raw[i].append(assert_numpy(losses[i]))

    def print(self, *losses):
        iter = len(self._raw[0])
        report_str = "{0}\t".format(iter)
        for i in range(self._n_reported):
            report_str += "{0}: {1:.4f}\t".format(self._labels[i], self._raw[i][-1])
        print(report_str)

    def losses(self, n_smooth=1):
        x = np.arange(n_smooth, len(self._raw[0]) + 1)
        ys = []
        for (label, raw) in zip(self._labels, self._raw):
            raw = assert_numpy(raw).reshape(-1)
            kernel = np.ones(shape=(n_smooth,)) / n_smooth
            ys.append(np.convolve(raw, kernel, mode="valid"))
        return self._labels, x, ys

    def recent(self, n_recent=1):
        return np.array([raw[-n_recent:] for raw in self._raw])


class KLTrainer(object):
    def __init__(
        self, bg, optim=None, train_likelihood=True, train_energy=True, custom_loss=None, test_likelihood=False,
    ):
        """Trainer for minimizing the forward or reverse

        Trains in either of two modes, or a mixture of them:
        1. Forward KL divergence / energy based training. Minimize KL divergence between
           generation probability of flow and target distribution
        2. Reverse KL divergence / maximum likelihood training. Minimize reverse KL divergence between
           data mapped back to latent space and prior distribution.

        """
        self.bg = bg

        if optim is None:
            optim = torch.optim.Adam(bg.parameters(), lr=5e-3)
        self.optim = optim

        loss_names = []
        self.train_likelihood = train_likelihood
        self.w_likelihood = 0.0
        self.train_energy = train_energy
        self.w_energy = 0.0
        self.test_likelihood = test_likelihood
        if train_energy:
            loss_names.append("KLL")
            self.w_energy = 1.0
        if train_likelihood:
            loss_names.append("NLL")
            self.w_likelihood = 1.0
        if test_likelihood: 
            loss_names.append("NLL(Test)")
        self.reporter = LossReporter(*loss_names)
        self.custom_loss = custom_loss

    def train(
        self,
        n_iter,
        data=None,
        testdata=None,
        batchsize=128,
        w_likelihood=None,
        w_energy=None,
        w_custom=None,
        custom_loss_kwargs={},
        n_print=0,
        temperature=1.0,
        schedulers=(),
        clip_forces=None
    ):
        """
        Train the network.

        Parameters
        ----------
        n_iter : int
            Number of training iterations.
        data : torch.Tensor or Sampler
            Training data
        testdata : torch.Tensor or Sampler
            Test data
        batchsize : int
            Batchsize
        w_likelihood : float or None
            Weight for backward KL divergence during training;
            if specified, this argument overrides self.w_likelihood
        w_energy : float or None
            Weight for forward KL divergence divergence during training;
            if specified, this argument overrides self.w_energy
        n_print : int
            Print interval
        temperature : float
            Temperature at which the training is performed
        schedulers : iterable
            A list of pairs (int, scheduler), where the integer specifies the number of iterations between
            steps of the scheduler. Scheduler steps are invoked before the optimization step.

        Returns
        -------
        """
        if w_likelihood is None:
            w_likelihood = self.w_likelihood
        if w_energy is None:
            w_energy = self.w_energy

        if isinstance(data, torch.Tensor):
            data = DataSetSampler(data)
        if isinstance(testdata, torch.Tensor):
            testdata = DataSetSampler(testdata)

        for iter in tqdm(range(n_iter)):
            # invoke schedulers
            for interval, scheduler in schedulers:
                if iter % interval == 0:
                    scheduler.step()
            self.optim.zero_grad()
            reports = []

            if self.train_energy:
                # kl divergence to the target
                kll = self.bg.kldiv(batchsize, temperature=temperature).mean()
                reports.append(kll)
                # aggregate weighted gradient
                if w_energy > 0:
                    l = w_energy / (w_likelihood + w_energy)
                    (l * kll).backward(retain_graph=True)
                # constrain forces
                if clip_forces is not None:
                    torch.nn.utils.clip_grad_value_(self.bg.parameters(), clip_forces)

            if self.train_likelihood:
                batch = data.sample(batchsize)
                if isinstance(batch, torch.Tensor):
                    batch = (batch,)
                # negative log-likelihood of the batch is equal to the energy of the BG

                nll = self.bg.energy(*batch, temperature=temperature).mean()
                reports.append(nll)
                # aggregate weighted gradient
                #bp()
                if w_likelihood > 0:
                    l = w_likelihood / (w_likelihood + w_energy)
                    (l * nll).backward(retain_graph=False) # was True
                #if iter == 79:
                #    import ipdb
                #    ipdb.set_trace()                    
                
            # compute NLL over test data 
            if self.test_likelihood:
                testnll = torch.zeros_like(nll)
                if testdata is not None:
                    testbatch = testdata.sample(batchsize)
                    if isinstance(testbatch, torch.Tensor):
                        testbatch = (testbatch,)
                    with torch.no_grad():
                        testnll = self.bg.energy(*testbatch, temperature=temperature).mean()
                reports.append(testnll)

            if w_custom is not None:
                cl = self.custom_loss(**custom_loss_kwargs)
                (w_custom * cl).backward(retain_graph=True)
                #import ipdb
                #ipdb.set_trace()
                reports.append(cl)

            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)
            
            if any(torch.any(torch.isnan(p.grad)) for p in self.bg.parameters() if p.grad is not None):
                print("found nan in grad; skipping optimization step")
            else:
                self.optim.step()
                #bp()
                #pass

    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)
