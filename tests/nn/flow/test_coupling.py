
import torch
from bgtorch import Flow, SplitFlow, Transformer, CouplingFlow, WrapFlow


def test_split_flow(device, dtype):
    tensor = torch.arange(0, 12., device=device, dtype=dtype).reshape(3,4)
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
