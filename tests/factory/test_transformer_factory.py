import pytest
import torch

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
    shape_info = ShapeDictionary.from_coordinate_transform(crd_trafo)
    conditioners = make_conditioners(transformer_type, (BONDS,), (FIXED,), shape_info)
    transformer = make_transformer(transformer_type, (BONDS,), shape_info, conditioners=conditioners)
    out = transformer.forward(torch.zeros(2, shape_info[FIXED][0]), torch.zeros(2, shape_info[BONDS][0]))
    assert out[0].shape == (2, shape_info[BONDS][0])
