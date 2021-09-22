import pytest
import torch
from bgflow.utils import distance_vectors, distances_from_vectors, remove_mean


@pytest.mark.parametrize("remove_diagonal", [True, False])
def test_distances_from_vectors(remove_diagonal):
    """Test if distances are calculated correctly from a particle configuration."""
    particles = torch.tensor([[[3., 0], [3, 0], [0, 4]]])
    distances = distances_from_vectors(
        distance_vectors(particles, remove_diagonal=remove_diagonal))
    if remove_diagonal == True:
        assert torch.allclose(distances, torch.tensor([[[0., 5], [0, 5], [5, 5]]]), atol=1e-2)
    else:
        assert torch.allclose(distances, torch.tensor([[[0., 0, 5], [0, 0, 5], [5, 5, 0]]]), atol=1e-2)


def test_mean_free():
    """Test if the mean of random configurations is removed correctly"""
    samples = torch.rand(100, 100, 3) - 0.3
    samples = remove_mean(samples, 100, 3)
    mean_deviation = samples.mean(dim=(1, 2))
    threshold = 1e-5
    return torch.all(mean_deviation < threshold)
