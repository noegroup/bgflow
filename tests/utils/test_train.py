import torch
from bgflow.utils import ClipGradient


def torch_example(grad_clipping, ctx):
    positions = torch.arange(6).reshape(2, 3).to(**ctx)
    positions.requires_grad = True
    positions = grad_clipping.to(**ctx)(positions)
    (0.5 * positions ** 2).sum().backward()
    return positions.grad


def test_clip_by_val(ctx):
    grad_clipping = ClipGradient(clip=3., norm_dim=1)
    assert torch.allclose(
        torch_example(grad_clipping, ctx),
        torch.tensor([[0., 1., 2.], [3., 3., 3.]], **ctx)
    )


def test_clip_by_atom(ctx):
    grad_clipping = ClipGradient(clip=3., norm_dim=3)
    norm2 = torch.linalg.norm(torch.arange(3, 6, **ctx)).item()
    assert torch.allclose(
        torch_example(grad_clipping, ctx),
        torch.tensor([[0., 1., 2.], [3/norm2*3, 4/norm2*3, 5/norm2*3]], **ctx)
    )


def test_clip_by_batch(ctx):
    grad_clipping = ClipGradient(clip=3., norm_dim=-1)
    norm2 = torch.linalg.norm(torch.arange(6, **ctx)).item()
    assert torch.allclose(
        torch_example(grad_clipping, ctx),
        (torch.arange(6, **ctx) / norm2 * 3.).reshape(2, 3)
    )

