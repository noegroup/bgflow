

from bgflow.utils import as_numpy
import numpy as np
import torch


def test_as_numpy():
    assert np.allclose(as_numpy(2.0), np.array(2.0))
    assert np.allclose(as_numpy(np.ones(2)), np.ones(2))
    assert as_numpy(1) == np.array(1)


def test_tensor_as_numpy(ctx):
    out = as_numpy(torch.zeros(2, **ctx))
    assert isinstance(out, np.ndarray)
    assert np.allclose(out, np.zeros(2))
