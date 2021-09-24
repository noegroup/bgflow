import pytest
import torch
from bgflow.utils import distance_vectors, distances_from_vectors, remove_mean, compute_distances


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


def test_mean_free(ctx):
    """Test if the mean of random configurations is removed correctly"""
    samples = torch.rand(100, 100, 3, **ctx) - 0.3
    samples = remove_mean(samples, 100, 3)
    mean_deviation = samples.mean(dim=(1, 2))
    threshold = 1e-5
    assert torch.all(mean_deviation.abs() < threshold)

@pytest.mark.parametrize("remove_duplicates", [True, False])
def test_compute_distances(remove_duplicates, ctx):
    """Test if distances are calculated correctly from a particle configuration."""
    particles = torch.tensor([[[3., 0], [3, 0], [0, 4]]], **ctx)
    distances = compute_distances(particles, n_particles=3, n_dimensions=2, remove_duplicates=remove_duplicates)
    if remove_duplicates == True:
        assert torch.allclose(distances, torch.tensor([[0., 5, 5]], **ctx), atol=1e-5)
    else:
        assert torch.allclose(distances, torch.tensor([[[0., 0, 5], [0, 0, 5], [5, 5, 0]]], **ctx), atol=1e-5)
