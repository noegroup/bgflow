import torch
import numpy as np

import warnings

from bgflow.utils.types import assert_numpy
from bgflow.distribution.sampling import DataSetSampler


__all__ = ["LossReporter", "KLTrainer", "FlowMatchingTrainer"]


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
        clip_forces=None,
        progress_bar=lambda x:x
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
        progress_bar : callable
            To show a progress bar, pass `progress_bar = tqdm.auto.tqdm`

        Returns
        -------
        """
        if w_likelihood is None:
            w_likelihood = self.w_likelihood
        if w_energy is None:
            w_energy = self.w_energy
        if clip_forces is not None:
            warnings.warn(
                "clip_forces is deprecated and will be ignored. "
                "Use GradientClippedEnergy instances instead",
                DeprecationWarning
            )

        if isinstance(data, torch.Tensor):
            data = DataSetSampler(data)
        if isinstance(testdata, torch.Tensor):
            testdata = DataSetSampler(testdata)

        for iter in progress_bar(range(n_iter)):
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

            if self.train_likelihood:
                batch = data.sample(batchsize)
                if isinstance(batch, torch.Tensor):
                    batch = (batch,)
                # negative log-likelihood of the batch is equal to the energy of the BG
                nll = self.bg.energy(*batch, temperature=temperature).mean()
                reports.append(nll)
                # aggregate weighted gradient
                if w_likelihood > 0:
                    l = w_likelihood / (w_likelihood + w_energy)
                    (l * nll).backward(retain_graph=True)
                
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
                reports.append(cl)

            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)
            
            if any(torch.any(torch.isnan(p.grad)) for p in self.bg.parameters() if p.grad is not None):
                print("found nan in grad; skipping optimization step")
            else:
                self.optim.step()


    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)
    

class FlowMatchingTrainer(object):
    def __init__(
        self, bg, optim=None, optimal_transport=False, equivariant_optimal_transport=False, test_loss=False
    ):
        """Trainer for minimizing the flow matching objective. Supports also optimal transport flow matching.
        """

        self.bg = bg

        if optim is None:
            optim = torch.optim.Adam(bg.parameters(), lr=5e-4)
        self.optim = optim

        loss_names = []
        self.optimal_transport = optimal_transport
        self.equivariant_optimal_transport = equivariant_optimal_transport
        assert not (self.optimal_transport and self.equivariant_optimal_transport), \
            "Choose either OT flow matching or equivariant OT flow matching (not both)."
        loss_names.append("FM")
        if test_loss:
            self.test_loss = test_loss
            loss_names.append("FM(Test)")
        self.reporter = LossReporter(*loss_names)



    def flow_matching_loss(self, batchsize, prior_data, target_data, noise_data, noise_scaling):
        """
        Computes the flow matching or optimal flow matching loss. 
        """
        x1 = target_data.sample(batchsize)
        t = torch.rand(batchsize, 1).to(x1)
        x0 = prior_data.sample(batchsize)

        if self.optimal_transport:
            import ot as pot
            # Resample x0, x1 according to transport matrix
            a1, b1 = pot.unif(x0.size()[0]), pot.unif(x1.size()[0])
            M = torch.cdist(x0, x1) ** 2
            M = M / M.max()
            pi = pot.emd(a1, b1, M.detach().cpu().numpy())
            # Sample random interpolations on pi
            p = pi.flatten()
            p = p / p.sum()
            choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batchsize)
            i, j = np.divmod(choices, pi.shape[1])
            x0 = x0[i]
            x1 = x1[j]
        # evaluation points
        mu_t = x0 * (1 - t) + x1 * t
        noise = noise_data.sample(batchsize)
        x = mu_t + noise_scaling * noise
        # target vectorfield
        ut = x1 - x0
        vt = self.bg.flow._dynamics._dynamics._dynamics_function(t, x)
        loss = torch.mean((vt - ut) ** 2)                  
        return loss
    
    def train(
        self,
        n_iter,
        prior_data=None,
        target_data=None,
        test_prior_data=None,
        test_target_data=None,
        noise_data=None,
        noise_scaling=0.01,
        batchsize=128,
        n_print=0,
        schedulers=(),
        progress_bar=lambda x:x
    ):
        """
        Train the network.

        Parameters
        ----------
        n_iter : int
            Number of training iterations.
        prior_data : torch.Tensor or Sampler
            Training data (prior)
        target_data : torch.Tensor or Sampler
            Training data
        test_prior_data : torch.Tensor or Sampler
            Test data (target)
        test_target_data : torch.Tensor or Sampler
            Test data (prior)
        noise_data : torch.Tensor or Sampler
            Noise data for the flow matching objective         
        noise_scaling : float
            Noise scaling for the flow matching objective            
        batchsize : int
            Batchsize
        n_print : int
            Print interval
        schedulers : iterable
            A list of pairs (int, scheduler), where the integer specifies the number of iterations between
            steps of the scheduler. Scheduler steps are invoked before the optimization step.
        progress_bar : callable
            To show a progress bar, pass `progress_bar = tqdm.auto.tqdm`

        Returns
        -------
        """
        if isinstance(prior_data, torch.Tensor):
            prior_data = DataSetSampler(prior_data)
        if isinstance(target_data, torch.Tensor):
            target_data = DataSetSampler(target_data)
        if isinstance(test_prior_data, torch.Tensor):
            test_prior_data = DataSetSampler(test_prior_data)
        if isinstance(test_target_data, torch.Tensor):
            test_target_data = DataSetSampler(test_target_data)
        if isinstance(noise_data, torch.Tensor):
            noise_data = DataSetSampler(noise_data)     
            
        for iter in progress_bar(range(n_iter)):
            # invoke schedulers
            for interval, scheduler in schedulers:
                if iter % interval == 0:
                    scheduler.step()
            self.optim.zero_grad()
            reports = []
            # FM loss
            loss = self.flow_matching_loss(batchsize, prior_data, target_data, noise_data, noise_scaling)
            loss.backward()
            reports.append(loss)            
                
            # compute FM loss over test data 
            if self.test_loss:
                with torch.no_grad():
                    test_loss = self.flow_matching_loss(batchsize, test_prior_data, test_target_data, noise_data, noise_scaling)
                reports.append(test_loss)

            self.reporter.report(*reports)
            if n_print > 0:
                if iter % n_print == 0:
                    self.reporter.print(*reports)
            
            if any(torch.any(torch.isnan(p.grad)) for p in self.bg.parameters() if p.grad is not None):
                print("found nan in grad; skipping optimization step")
            else:
                self.optim.step()


    def losses(self, n_smooth=1):
        return self.reporter.losses(n_smooth=n_smooth)

