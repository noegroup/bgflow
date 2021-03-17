
import torch

from bgtorch import DataSetSampler


def test_dataset_sampler(device, dtype):
    data = torch.arange(12).reshape(4,3).to(device, dtype)
    sampler = DataSetSampler(data)
    idxs = sampler._idxs.copy()
    # test sampling out of range
    assert torch.allclose(sampler.sample(3), data[idxs[:3]])
    assert torch.allclose(sampler.sample(3)[0], data[idxs[3]])
    for i in range(10):
        assert sampler.sample(3).shape == (3, 3)
    # test sampling larger #samples than len(data)
    sampler._current_index = 0
    samples = sampler.sample(12)
    assert samples.shape == (12, 3)
    # check that rewinding works
    for i in range(3):
        assert torch.allclose(
            data.flatten(),
            torch.sort(samples[4*i: 4*(i+1)].flatten())[0]
        )

