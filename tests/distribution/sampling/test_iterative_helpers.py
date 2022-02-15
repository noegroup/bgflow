
import torch
from bgflow.distribution.sampling._iterative_helpers import _map_to_primary_cell


def test_map_to_primary_cell():
    cell = torch.eye(3)
    x = torch.tensor([[1.2, -0.1, 4.5]])
    assert torch.allclose(_map_to_primary_cell(x, cell), torch.tensor([[0.2, 0.9, 0.5]]))

    cell = torch.tensor(
        [
            [1.0, 2.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )
    assert torch.allclose(_map_to_primary_cell(x, cell), torch.tensor([[2.2, 1.9, 0.5]]))

