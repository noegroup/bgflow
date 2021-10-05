import pytest
import torch

from bgflow.distribution import Energy


pytestmark = pytest.mark.filterwarnings("ignore:This Energy instance is defined on multidimensional events")


class DummyEnergy(Energy):
    def __init__(self, dim):
        super().__init__(dim=dim)

    def _energy(self, *xs, **kwargs):
        return sum(
            0.5 * x.pow(2).view(x.shape[0], -1).sum(dim=-1, keepdim=True) for x in xs
        )


def test_energy_event_parser():
    from bgflow.distribution.energy.base import (
        _is_non_empty_sequence_of_integers,
        _is_sequence_of_non_empty_sequences_of_integers,
    )

    assert _is_non_empty_sequence_of_integers([10, 2, 6, 9])
    assert _is_non_empty_sequence_of_integers(torch.Size([10, 20, 3]))
    assert not _is_non_empty_sequence_of_integers([10, 2, 2.4])
    assert not _is_non_empty_sequence_of_integers([10, 2, [3]])
    assert not _is_non_empty_sequence_of_integers([[10], [10]])

    assert _is_sequence_of_non_empty_sequences_of_integers([[10, 10], [5], [10]])
    assert _is_sequence_of_non_empty_sequences_of_integers(
        [torch.Size([10, 10]), torch.Size([10])]
    )
    assert _is_sequence_of_non_empty_sequences_of_integers(
        [[10, 5], torch.Size([10, 10])]
    )
    assert not _is_sequence_of_non_empty_sequences_of_integers([[10, 10], 10, [10]])
    assert not _is_sequence_of_non_empty_sequences_of_integers([[10, 5], [10, 2.0]])
    assert not _is_sequence_of_non_empty_sequences_of_integers([10, 10])
    assert not _is_sequence_of_non_empty_sequences_of_integers([[], [10]])
    assert not _is_sequence_of_non_empty_sequences_of_integers(torch.Size([10, 10]))
    #assert not _is_sequence_of_non_empty_sequences_of_integers([[10, 10, 10]])


@pytest.mark.parametrize("batch", [[23], [23, 71], [23, 71, 13]])
def test_energy_event_types(batch, with_grad_and_no_grad):

    # test single dimension input
    dim = 11
    dummy = DummyEnergy(dim)
    x = torch.randn(*batch, dim)
    f = dummy.force(x)

    assert torch.allclose(-x, f)
    assert dummy.dim == 11
    assert dummy.event_shape == torch.Size([11])

    # this should fail (too many inputs)
    with pytest.raises(AssertionError):
        dummy.force(x, x)

    # test tensor input
    shape = [11, 7, 4, 3]
    dummy = DummyEnergy(shape)
    x = torch.randn(*batch, *shape)
    f = dummy.force(x)
    assert torch.allclose(-x, f)
    assert dummy.dim == 11 * 7 * 4 * 3
    assert dummy.event_shape == torch.Size([11, 7, 4, 3])

    # this should fail (too many inputs)
    with pytest.raises(AssertionError):
        dummy.force(x, x)

    # test multi-tensor input
    shapes = [[11, 7], [5, 3], [13, 17]]
    dummy = DummyEnergy(shapes)
    x, y, z = [torch.randn(*batch, *shape) for shape in shapes]
    fx, fy, fz = dummy.force(x, y, z)
    assert all(torch.allclose(-x, f) for (x, f) in zip([x, y, z], [fx, fy, fz]))
    fx, fy, fz = dummy.force(x, y, z, ignore_indices=[1])
    assert fy is None and all(
        torch.allclose(-x, f) for (x, f) in zip([x, z], [fx, fz])
    )

    # this should fail: inconsistent batch dimension
    with pytest.raises(AssertionError):
        batches = [[5, 7], [5, 7], [5, 6]]
        x, y, z = [
            torch.randn(*batch, *shape)
            for (batch, shape) in zip(batches, shapes)
        ]
        fx, fy, fz = dummy.force(x, y, z)

    # this should fail: wrong input shapes
    with pytest.raises(AssertionError):
        batches = [[5, 7], [5, 7], [5, 7]]
        x, y, z = [
            torch.randn(*batch, *shape)
            for (batch, shape) in zip(batches, shapes)
        ]
        y = y[..., :-1]
        fx, fy, fz = dummy.force(x, y, z)

    # this should fail: dim not defined for multiple tensor input
    with pytest.raises(ValueError):
        dummy.dim

    # this should fail: single event_shape is not defined for multipe tensor input
    with pytest.raises(ValueError):
        dummy.event_shape

    # this should fail (too few inputs)
    with pytest.raises(AssertionError):
        dummy.force(x, y)

    # this should fail (too many inputs)
    with pytest.raises(AssertionError):
        dummy.force(x, y, z, x)

    # test that `requires_grad` state of input vars stays preserved
    shapes = [[11, 7], [5, 3], [13, 17]]
    dummy = DummyEnergy(shapes)
    x, y, z = [torch.randn(*batch, *shape) for shape in shapes]
    x.requires_grad_(True)
    y.requires_grad_(False)
    z.requires_grad_(False)
    fx, fy, fz = dummy.force(x, y, z, ignore_indices=[1])
    assert x.requires_grad and not (y.requires_grad) and (not z.requires_grad)
    assert fy is None and all(
        torch.allclose(-x, f) for (x, f) in zip([x, z], [fx, fz])
    )

    # test for singular shapes in multi-tensor setting
    shapes = [[11], [5], [13]]
    dummy = DummyEnergy(shapes)
    x, y, z = [torch.randn(*batch, *shape) for shape in shapes]
    fx, fy, fz = dummy.force(x, y, z)
    assert all(torch.allclose(-x, f) for (x, f) in zip([x, y, z], [fx, fy, fz]))
    fx, fy, fz = dummy.force(x, y, z, ignore_indices=[1])
    assert fy is None and all(
        torch.allclose(-x, f) for (x, f) in zip([x, z], [fx, fz])
    )

