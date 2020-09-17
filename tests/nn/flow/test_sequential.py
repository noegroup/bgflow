

from bgtorch.nn.flow.sequential import SequentialFlow
from bgtorch.nn.flow.orthogonal import PseudoOrthogonalFlow


def test_trigger_penalty():
    flow = SequentialFlow([
        PseudoOrthogonalFlow(3),
        PseudoOrthogonalFlow(3),
        PseudoOrthogonalFlow(3),
    ])
    penalties = flow.trigger("penalty")
    assert len(penalties) == 3