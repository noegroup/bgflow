
import torch
from .base import Sampler


class IterativeSampler(Sampler, torch.utils.data.Dataset):
    def _sample(self, n_samples):
        pass
