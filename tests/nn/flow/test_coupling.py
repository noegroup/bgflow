import pytest

import torch
from bgflow import (
    Flow, SplitFlow, Transformer, CouplingFlow, WrapFlow,
    SetConstantFlow, VolumePreservingWrapFlow, DenseNet
)


def test_split_flow(ctx):
    tensor = torch.arange(0, 12., **ctx).reshape(3,4)
    # pass or infer last size
    for sizes in ((2,), (2,2)):
        split = SplitFlow(*sizes, dim=-1)
        *result, dlogp = split.forward(tensor)
        assert torch.allclose(result[0], tensor[...,:2])
        assert torch.allclose(result[1], tensor[...,2:])
        assert dlogp.shape == (3,1)
    # dim != -1
    split = SplitFlow(2, dim=0)
    *result, dlogp = split.forward(tensor)
    assert torch.allclose(result[0], tensor[:2,...])
    assert torch.allclose(result[1], tensor[2:,...])
    assert dlogp.shape == (1,4)  #  <- this does not make sense yet until we allow event shapes


def test_split_with_indices(ctx):
    tensor = torch.arange(0, 12., **ctx).reshape(1,3,4)
    indices = [[0,2],[1]]
    split = SplitFlow(*indices, dim=-2)
    *result, dlogp = split.forward(tensor)
    expected = (
        torch.tensor([[0, 1, 2, 3], [8, 9, 10, 11]], **ctx),
        torch.tensor([[4, 5, 6, 7]], **ctx),
    )
    for r, e in zip(result, expected):
        assert torch.allclose(r, e)
    assert dlogp.shape == (1,1,4)  #  <- this does not make sense yet until we allow event shapes
    x, dlogp2 = split.forward(*result, inverse=True)
    assert dlogp2.shape == (1, 1, 4)
    assert torch.allclose(x, tensor)


@pytest.mark.parametrize("indices", (
    [[0], [2]],  # missing indices
    [[0,1], [1,2]],  # doubly defined
))
def test_split_with_indices_failure(ctx, indices):
    split = SplitFlow(*indices, dim=-2)
    with pytest.raises(ValueError):
        tensor = torch.arange(0, 12., **ctx).reshape(1,3,4)
        split.forward(tensor)
    with pytest.raises(Exception):
        tensors = (
            torch.arange(0, 8., **ctx).reshape(1,2,4),
            torch.arange(8, 16., **ctx).reshape(1,2,4)
        )
        split.forward(*tensors, inverse=True)


class DummyTransformer(Transformer):
    def __init__(self):
        super().__init__()

    def _forward(self, x, y, **kwargs):
        return y+2*x, torch.zeros_like(x[...,[0]])

    def _inverse(self, x, y, **kwargs):
        return y-2*x, torch.zeros_like(x[...,[0]])


def test_coupling_default(device, dtype):
    """Test coupling layer with old API"""
    transformer = DummyTransformer()
    coupling = CouplingFlow(transformer)
    x1 = torch.ones((1,2), device=device, dtype=dtype)
    x2 = torch.zeros((1,2)).to(x1)
    y1, y2, dlogp = coupling.forward(x1, x2)
    assert torch.allclose(y1, x1)
    assert torch.allclose(y2, 2*x1)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))

    z1, z2, dlogp2 = coupling.forward(y1, y2, inverse=True)
    assert torch.allclose(z1, x1)
    assert torch.allclose(z2, x2)
    assert torch.allclose(dlogp2, torch.zeros_like(x1[...,[0]]))


def test_coupling_multiple_conditioned(device, dtype):
    """Test conditioning in reverse order and on multiple inputs"""
    transformer = DummyTransformer()
    coupling = CouplingFlow(transformer, transformed_indices=(0,), cond_indices=[1,2])
    x1 = torch.zeros((1, 10), device=device, dtype=dtype)
    x2 = torch.ones((1, 4)).to(x1)
    x3 = torch.ones((1, 6)).to(x1)
    y1, y2, y3, dlogp = coupling.forward(x1, x2, x3)
    assert torch.allclose(y1, x1+2.)
    assert torch.allclose(y2, x2)
    assert torch.allclose(y3, x3)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))

    z1, z2, z3, dlogp = coupling.forward(y1, y2, y3, inverse=True)
    assert torch.allclose(z1, x1)
    assert torch.allclose(z2, x2)
    assert torch.allclose(z3, x3)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))


def test_coupling_multiple_transformed(device, dtype):
    """Test conditioning with multiple transformed variables"""
    transformer = DummyTransformer()
    coupling = CouplingFlow(transformer, transformed_indices=(0,2), cond_indices=(3,1))
    x1 = torch.zeros((1, 6), device=device, dtype=dtype)
    x2 = torch.zeros((1, 7)).to(x1)
    x3 = torch.ones((1, 4)).to(x1)
    x4 = torch.ones((1, 3)).to(x1)
    y1, y2, y3, y4, dlogp = coupling.forward(x1, x2, x3, x4)
    expected = torch.tensor(
        [[2.0]*3 + [0.0]*3 + [0.0]*7 + [1.0]*4 + [1.0]*3]
    ).to(x1)
    assert torch.allclose(torch.cat([y1, y2, y3, y4], dim=-1), expected)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))

    z1, z2, z3, z4, dlogp = coupling.forward(y1, y2, y3, y4, inverse=True)
    assert torch.allclose(z1, x1)
    assert torch.allclose(z2, x2)
    assert torch.allclose(z3, x3)
    assert torch.allclose(z4, x4)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))


def test_wrap_flow(device, dtype):
    transformer = DummyTransformer()
    coupling = CouplingFlow(transformer, transformed_indices=(0,), cond_indices=[1])
    wrap = WrapFlow(coupling, indices=[1,2], out_indices=[1,0])
    # wrap      x1, (x2, x3)
    # transform x1, (x2 + 2*x3, x3)
    # reorder   (x3, x2 + 2*x3), x1
    x1 = torch.zeros(2, device=device, dtype=dtype)
    x2 = torch.ones(1).to(x1)
    x3 = torch.ones(1).to(x1)
    y1, y2, y3, dlogp = wrap.forward(x1, x2, x3)
    assert torch.allclose(y1, x3)
    assert torch.allclose(y2, x2 + 2*x3)
    assert torch.allclose(y3, x1)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))

    z1, z2, z3, dlogp = wrap.forward(y1, y2, y3, inverse=True)
    assert torch.allclose(z1, x1)
    assert torch.allclose(z2, x2)
    assert torch.allclose(z3, x3)
    assert torch.allclose(dlogp, torch.zeros_like(x1[...,[0]]))


class DummyNVPFlow(Flow):
    def __init__(self, skip_indices=[3]):
        super().__init__()
        self.skip = skip_indices

    def _forward(self, *xs, **kwargs):
        out = (x if i in self.skip else 2*x
               for i, x in enumerate(xs))
        dlogp = torch.log(2**(len(xs)-len(self.skip))*torch.ones_like(xs[0][..., [0]]))
        return (*out, dlogp)

    def _inverse(self, *xs, **kwargs):
        out = (x if i in self.skip else x/2
               for i, x in enumerate(xs))
        dlogp = torch.log(0.5**(len(xs)-len(self.skip))*torch.ones_like(xs[0][..., [0]]))
        return (*out, dlogp)


def test_volume_preserving_wrap(ctx):
    # five inputs
    inputs = torch.arange(1, 21, **ctx).reshape(2, 10).chunk(5, dim=-1)
    # flow
    wrap_flow = VolumePreservingWrapFlow(
        flow=DummyNVPFlow(),
        volume_sink_index=3,
        out_volume_sink_index=3,
        shift_transformation=None,
        scale_transformation=DenseNet([9, 2]),
        cond_indices=(0, 1, 2, 3, 5, ),
    ).to(**ctx)
    *outputs, dlogp = wrap_flow.forward(*inputs)
    assert torch.allclose(dlogp, torch.zeros_like(dlogp))
    assert torch.allclose(outputs[0], 2*inputs[0])
    assert torch.allclose(outputs[1], 2*inputs[1])
    assert torch.allclose(outputs[2], 2*inputs[2])
    expected = 1/(2**8)*inputs[3].prod()
    assert torch.allclose(outputs[3].prod(), expected)
    assert torch.allclose(outputs[4], 2*inputs[4])

    # inversion
    *z2, dlogp2 = wrap_flow.forward(*outputs, inverse=True)
    assert torch.allclose(dlogp2, -dlogp)
    for x, y in zip(inputs, z2):
        print(x, y)
    for x, y in zip(inputs, z2):
        assert torch.allclose(x, y)


def test_set_constant_flow(ctx):
    batchsize = 4
    x = torch.arange(20.0, **ctx).reshape(batchsize, -1)
    const = torch.arange(5.0, 10.0, **ctx)
    flow = SetConstantFlow((1,), (const, ))
    *y, dlogp = flow._forward(x)
    assert len(y) == 2
    assert torch.allclose(y[0], x)
    for i in range(batchsize):
        assert torch.allclose(y[1][i], const)
    assert torch.allclose(dlogp, torch.zeros(batchsize, **ctx))

    x2, dlogp = flow._inverse(*y)
    assert torch.allclose(x2, x)
    assert torch.allclose(dlogp, torch.zeros(batchsize, **ctx))
