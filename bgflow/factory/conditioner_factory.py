
import torch
import bgflow as bg
from ..nn.periodic import WrapPeriodic


__all__ = ["make_conditioners"]


# === Conditioner Factory ===

def make_conditioners(
        transformer_type,
        what,
        on,
        shape_info,
        transformer_kwargs={},
        hidden=(128,128),
        activation=torch.nn.SiLU(),
        **kwargs
):
    """Create coupling layer conditioners for a given transformer type,
    taking care of circular and non-circular tensors.

    The implementation follows the factory pattern. The concrete implementations
    for different transformer types are registered in the dictionary
    `CONDITIONER_FACTORIES` within this module.


    Parameters
    ----------
    transformer_type : class
        The transformer class.
    what : tuple of TensorInfo
        Fields that shall be transformed.
    on : tuple of TensorInfo
        Fields that are input to the conditioner.
    shape_info : ShapeDictionary
        A ShapeDictionary instance that knows the shapes of
        all fields in `what` and `on`.
    **kwargs :
        Keyword arguments are passed down to the concrete implementations
        of the factory.

    Returns
    -------
    transformer : bg.Transformer
    """
    if _is_sin(activation):
        scale_first_weights = kwargs.pop("siren_scale_first_weights", False)
        initialize = kwargs.pop("siren_initialize", False)
    dim_out_factory = CONDITIONER_OUT_DIMS[transformer_type]
    dim_out = dim_out_factory(what=what, shape_info=shape_info, transformer_kwargs=transformer_kwargs, **kwargs)
    dim_in = shape_info.dim_noncircular(on) + 2 * shape_info.dim_circular(on)
    conditioners = {}
    for name in dim_out:
        if _is_sin(activation):
            dense_net = bg.SirenDenseNet(
                [dim_in, *hidden, dim_out[name]],
                scale_first_weights=scale_first_weights,
                initialize=initialize
            )
        else:
            dense_net = bg.DenseNet([dim_in, *hidden, dim_out[name]], activation=activation)
        if shape_info.dim_circular(on) > 0:
            dense_net = WrapPeriodic(dense_net, indices=shape_info.circular_indices(on))
        conditioners[name] = dense_net
    return conditioners


def _spline_out_dims(what, shape_info, transformer_kwargs={}, num_bins=8):
    # input for conditioner
    dim_out = 3 * num_bins * shape_info.dim_all(what) + shape_info.dim_noncircular(what)
    return {"params_net": dim_out}


def _affine_out_dims(what, shape_info, transformer_kwargs={}):
    dim_out = shape_info.dim_all(what)
    return {"shift_transformation": dim_out, "scale_transformation": dim_out}


def _mixture_out_dims(what, shape_info, transformer_kwargs={}, num_components=8):
    dim_out1 = num_components * shape_info.dim_all(what)
    return {"weights": dim_out1, "alphas": dim_out1, "params": 3*dim_out1}


CONDITIONER_OUT_DIMS = {
    bg.ConditionalSplineTransformer: _spline_out_dims,
    bg.AffineTransformer: _affine_out_dims,
    #TODO bg.MixtureCDFTransformer: _mixture_out_dims
}


def _is_sin(f):
    test_points = torch.arange(100., dtype=torch.float32)
    try:
        if torch.allclose(f(test_points), torch.sin(test_points)):
            return True
        else:
            return False
    except:
        return False