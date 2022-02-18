import pytest
import torch

import bgflow
from bgflow import (
    make_transformer, make_conditioners,
    ShapeDictionary, BONDS, FIXED, ANGLES, TORSIONS,
    ConditionalSplineTransformer, AffineTransformer
)



@pytest.mark.parametrize(
    "transformer_type",
    [
        ConditionalSplineTransformer,
        AffineTransformer,
        # TODO: MixtureCDFTransformer
    ]
)
def test_transformers(crd_trafo, transformer_type):
    pytest.importorskip("nflows")

    shape_info = ShapeDictionary.from_coordinate_transform(crd_trafo)
    conditioners = make_conditioners(transformer_type, (BONDS,), (FIXED,), shape_info)
    transformer = make_transformer(transformer_type, (BONDS,), shape_info, conditioners=conditioners)
    out = transformer.forward(torch.zeros(2, shape_info[FIXED][0]), torch.zeros(2, shape_info[BONDS][0]))
    assert out[0].shape == (2, shape_info[BONDS][0])


def test_circular_affine(crd_trafo):
    shape_info = ShapeDictionary.from_coordinate_transform(crd_trafo)

    with pytest.raises(ValueError):
        conditioners = make_conditioners(
            bgflow.AffineTransformer,
            (TORSIONS,), (FIXED,), shape_info=shape_info
        )
        make_transformer(bgflow.AffineTransformer, (TORSIONS,), shape_info, conditioners=conditioners)

    conditioners = make_conditioners(
        bgflow.AffineTransformer,
        (TORSIONS,), (FIXED,), shape_info=shape_info, use_scaling=False
    )
    assert list(conditioners.keys()) == ["shift_transformation"]
    transformer = make_transformer(bgflow.AffineTransformer, (TORSIONS,), shape_info, conditioners=conditioners)
    assert transformer._is_circular
    out = transformer.forward(torch.zeros(2, shape_info[FIXED][0]), torch.zeros(2, shape_info[TORSIONS][0]))
    assert out[0].shape == (2, shape_info[TORSIONS][0])
