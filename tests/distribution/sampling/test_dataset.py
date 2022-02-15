
import torch

from bgflow import DataSetSampler, DataLoaderSampler


def test_dataset_sampler(ctx):
    data = torch.arange(12).reshape(4,3).to(**ctx)
    sampler = DataSetSampler(data).to(**ctx)
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


def test_dataset_to_device_sampler(ctx):
    data = torch.arange(12).reshape(4,3)
    sampler = DataSetSampler(data).to(**ctx)
    assert sampler.sample(10).device == ctx["device"]


def test_multiple_dataset_sampler(ctx):
    data = torch.arange(12).reshape(4,3).to(**ctx)
    data2 = torch.arange(8).reshape(4,2).to(**ctx)
    sampler = DataSetSampler(data, data2).to(**ctx)
    samples = sampler.sample(3)
    assert len(samples) == 2
    assert samples[0].shape == (3, 3)
    assert samples[1].shape == (3, 2)
    assert samples[0].device == ctx["device"]


def test_resizing(ctx):
    data = torch.arange(12).reshape(4, 3).to(**ctx)
    sampler = DataSetSampler(data)
    sampler.resize_(5)
    assert len(sampler) == 5
    assert sampler.sample(2).shape == (2, 3)
    sampler.resize_(3)
    assert len(sampler) == 3
    assert sampler.sample(2).shape == (2, 3)


def test_dataloader_sampler(ctx):
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.randn(10, 2, 2, **ctx)),
        batch_size=4,
    )
    sampler = DataLoaderSampler(loader, **ctx)
    samples = sampler.sample(4)
    assert samples.shape == (4, 2, 2)
    assert samples.device == ctx["device"]

