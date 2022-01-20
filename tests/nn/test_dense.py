
import bgflow as bg
import torch
import numpy as np


def test_siren_net():
    siren_net = bg.SirenDenseNet((100,200,300,1))
    linear_layers = [layer for layer in siren_net._layers if isinstance(layer, torch.nn.Linear)]
    activations = [layer for layer in siren_net._layers if isinstance(layer, bg.nn.dense.Sin)]
    assert len(activations) == 2
    assert len(linear_layers) == 3
    for layer in linear_layers:
        assert torch.isclose(layer.weight.mean(), torch.zeros(1), atol=0.1)
    assert torch.any(linear_layers[0].weight > 30 * 0.9 * np.sqrt(6/100))
    assert torch.all(linear_layers[0].weight < 30 * np.sqrt(6/100))
    assert torch.any(linear_layers[0].weight < - 30 * 0.9 * np.sqrt(6/100))
    assert torch.all(linear_layers[0].weight > - 30 * np.sqrt(6/100))
    for layer in linear_layers[1:]:
        assert torch.any(layer.weight > 0.9 * np.sqrt(6/layer.weight.shape[-1]))
        assert torch.all(layer.weight < np.sqrt(6/layer.weight.shape[-1]))
        assert torch.any(layer.weight < - 0.9 * np.sqrt(6/layer.weight.shape[-1]))
        assert torch.all(layer.weight > - np.sqrt(6/layer.weight.shape[-1]))
