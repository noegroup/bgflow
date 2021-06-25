
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


@pytest.fixture()
def ctx(dtype, device):
    return {"dtype": dtype, "device": device}


@pytest.fixture(params=[torch.enable_grad, torch.no_grad])
def with_grad_and_no_grad(request):
    """Run a test with and without torch grad enabled"""
    with request.param():
        yield
