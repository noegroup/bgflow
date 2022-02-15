

from bgflow.nn.flow.sequential import SequentialFlow
from bgflow.nn.flow.orthogonal import PseudoOrthogonalFlow
from bgflow.nn.flow.elementwise import BentIdentity
from bgflow.nn.flow.triangular import TriuFlow


def test_trigger_penalty():
    flow = SequentialFlow([
        PseudoOrthogonalFlow(3),
        PseudoOrthogonalFlow(3),
        PseudoOrthogonalFlow(3),
    ])
    penalties = flow.trigger("penalty")
    assert len(penalties) == 3


def test_getitem():
    a = BentIdentity()
    b = TriuFlow(2)
    flow = SequentialFlow([a,b])
    assert flow[0] == a
    assert flow[1] == b
    subflow = flow[[0]]
    assert len(subflow) == 1
    assert isinstance(subflow, SequentialFlow)
    assert subflow[0] == a