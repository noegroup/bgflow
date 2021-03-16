
import pytest
import torch


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="CUDA not available."
            )
        )
    ]
)
def device(request):
    """Run a test case for all available devices."""
    return request.param


@pytest.fixture(params=[torch.float32, torch.float64])
def dtype(request, device):
    """Run a test case in single and double precision."""
    return request.param
