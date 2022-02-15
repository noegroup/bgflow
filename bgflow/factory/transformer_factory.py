"""Factory for flow transformations."""

import torch
from ..nn.flow.inverted import InverseFlow
from ..nn.flow.transformer.affine import AffineTransformer
from ..nn.flow.transformer.spline import ConditionalSplineTransformer

__all__ = ["make_transformer"]


def make_transformer(transformer_type, what, shape_info, conditioners, inverse=False, **kwargs):
    """Factory function.

    Parameters
    ----------
    transformer_type : bgflow.
    """
    factory = TRANSFORMER_FACTORIES[transformer_type]
    transformer = factory(what=what, shape_info=shape_info, conditioners=conditioners, **kwargs)
    if inverse:
        transformer = InverseFlow(transformer)
    return transformer


def _make_spline_transformer(what, shape_info, conditioners, **kwargs):
    return ConditionalSplineTransformer(
        is_circular=shape_info.is_circular(what),
        **conditioners,
        **kwargs
    )


def _make_affine_transformer(what, shape_info, conditioners, **kwargs):
    if shape_info.dim_circular(what) not in [0, shape_info[what[0]][-1]]:
        raise NotImplementedError(
            "Circular affine transformers are currently "
            "not supported for partly circular indices."
        )
    return AffineTransformer(
        **conditioners,
        is_circular=shape_info.dim_circular(what) > 0,
        **kwargs
    )

# def _make_sigmoid_transformer(
#         what,
#         shape_info,
#         conditioners,
#         smoothness_type="type1",
#         zero_boundary_left=False,
#         zero_boundary_right=False,
#         **kwargs
# ):
#     assert all(field.is_circular == what[0].is_circular for field in what)
#     transformer = bg.MixtureCDFTransformer(
#         compute_weights=conditioners["weights"],
#         compute_components=bg.AffineSigmoidComponents(
#             conditional_ramp=bg.SmoothRamp(
#                 compute_alpha=conditioners["alphas"],
#                 unimodal=True,
#                 ramp_type=smoothness_type
#             ),
#             compute_params=conditioners["params"],
#             periodic=what[0].is_circular,
#             min_density=torch.tensor(1e-6),
#             log_sigma_bound=torch.tensor(1.),
#             zero_boundary_left=zero_boundary_left,
#             zero_boundary_right=zero_boundary_right,
#             **kwargs
#         )
#     )
#     transformer = bg.WrapCDFTransformerWithInverse(
#         transformer=transformer,
#         oracle=bg.GridInversion( #bg.BisectionRootFinder(
#             transformer=transformer,
#             compute_init_grid=lambda x, y: torch.linspace(0, 1, 100).view(-1, 1, 1).repeat(1, *y.shape).to(y)
#             #abs_tol=torch.tensor(1e-5), max_iters=max_iters, verbose=False, raise_exception=True
#         )
#     )
#     return transformer


TRANSFORMER_FACTORIES = {
    ConditionalSplineTransformer: _make_spline_transformer,
    AffineTransformer: _make_affine_transformer,
    # MixtureCDFTransformer: _make_sigmoid_transformer
}

