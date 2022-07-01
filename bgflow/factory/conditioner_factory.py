import warnings

import torch
from ..nn.periodic import WrapPeriodic
import numpy as np
import traceback
import bgflow as bg
from torch.utils.checkpoint import checkpoint
__all__ = ["make_conditioners"]


from .tensor_info import (
    TensorInfo
)
CG_BONDS = TensorInfo(name='CG_BONDS', is_circular=False)
CG_ANGLES = TensorInfo(name='CG_ANGLES', is_circular=False)
CG_TORSIONS = TensorInfo(name='CG_TORSIONS', is_circular=True)
FIXED = TensorInfo(name='FIXED', is_circular=False, is_cartesian = True)


# === Conditioner Factory ===

def make_conditioners(
        transformer_type,
        what,
        on,
        shape_info,
        transformer_kwargs={},
        conditioner_type="dense",
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
    net_factory = CONDITIONER_FACTORIES[conditioner_type]
    dim_out_factory = CONDITIONER_OUT_DIMS[transformer_type]

    dim_out = dim_out_factory(what=what, shape_info=shape_info, transformer_kwargs=transformer_kwargs, **kwargs)
    dim_in = shape_info.dim_noncircular(on) + 2 * shape_info.dim_circular(on)
    conditioners = {}
    for name, dim in dim_out.items():
        kwargs["shape_info"] = shape_info
        kwargs["on"] = on
        conditioner = net_factory(dim_in, dim, **kwargs)
        if shape_info.dim_circular(on) > 0:
            conditioner = WrapPeriodic(conditioner, indices=shape_info.circular_indices(on))
        conditioners[name] = conditioner

    return conditioners


def _make_dense_conditioner(dim_in, dim_out, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
    return bg.DenseNet(
        [dim_in, *hidden, dim_out],
        activation=activation
    )





class GNNConditioner(torch.nn.Module):
    '''
    split input into wrapped periodic, nonperiodic and cartesians.
    On cartesians, apply GNN implemented using the nequip library.
    then apply dense net on (periodic || nonperiodic || GNN outputs)
    '''
    def __init__(self, dim_in, dim_out, hidden, activation, GNN_output_dim, **kwargs):
        from bgflow.factory.GNN_factory import CustomTransformerEncoderLayer

        super().__init__()
        self.GNN_output_dim = GNN_output_dim
        self.attention_level = kwargs["attention_level"]
        self.attention_units = kwargs["attention_units"]
        self.shape_info = kwargs["shape_info"]
        self.on = kwargs["on"]
        self.cart_indices = self.shape_info.cartesian_indices(self.on)
        self.cutoff = kwargs["r_max"] if "r_max" in kwargs.keys() else None
        self.use_checkpointing = kwargs["use_checkpointing"]
        self.n_cart = len(self.cart_indices)
        self.n_cart_atoms = self.n_cart//3

        self.labels = []
        for field in self.on:
            if field.is_circular:
                self.labels = (["circular"]*self.shape_info[field][0]*2) + self.labels  #add the sin, cos indices on the left
            elif field.is_cartesian:
                self.labels.extend(["cartesian"] * self.shape_info[field][0])
            else:
                self.labels.extend(["regular"] * self.shape_info[field][0])
        self.cart_indices_after_periodic = np.where(np.array(self.labels) == "cartesian")[0]

        ## this has to reaturn a tensor of shape (batchsize, self.GNN_output_dim)
        if isinstance(kwargs["GNN"], torch.nn.Module):
            self.GNN = kwargs["GNN"]
            if hasattr(self.GNN, "num_uses"):
                self.GNN.num_uses +=1
            else: self.GNN.num_uses = 1

        elif callable(kwargs["GNN"]):
            self.GNN = kwargs["GNN"](**kwargs["GNN_kwargs"])
            self.GNN.num_uses = 1
        self.GNN.num_evaluations = 0

        self.dim_dense_in = int(dim_in - self.n_cart + self.GNN_output_dim)
        self.densenet = bg.DenseNet(
            [self.dim_dense_in, *hidden, dim_out],
            activation=activation
        )

        ### define attention:
        if self.attention_level == "MHA":
            """
            on top of the GNN, have a single Multihead Attention on the atomwise/bondwise feature vectors
            """
            self.attention_dim = self.GNN_output_dim//self.attention_units
            self.MHA = torch.nn.MultiheadAttention(embed_dim = self.attention_dim, num_heads = 8, batch_first = True)
            self.qkv_proj = torch.nn.Linear(self.attention_dim, 3 * self.attention_dim)
        elif self.attention_level == "Transformer":
            """
            on top of the GNN, have a full Transformer Encoder on the atomwise/bondwise feature vectors,
            consisting of MHA and dense layers, with layer norm and skipped connections
            """
            self.attention_dim = self.GNN_output_dim//self.attention_units
            self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.attention_dim,
                                                                  nhead=8,
                                                                  batch_first = True,
                                                                  dropout=0.,
                                                                  dim_feedforward=64)
            self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        elif self.attention_level is not None:
            raise ValueError("unrecognized attention level")

    def forward(self,x):
        if self.use_checkpointing:
            dummy_var = torch.zeros(2, requires_grad = True)  ## checkpointing does not work without having input that requires grad
            return checkpoint(self._forward, x, dummy_var)
        else:
            try:
                return self._forward(x)
            except:
                self.GNN.num_evaluations=0

                print(traceback.format_exc())
                raise KeyboardInterrupt("the forward method was exited, but the GNN buffers have been cleaned up.")
    def _forward(self, x, dummy_var = None):
        x_cart = x[:,self.cart_indices_after_periodic]
        index = torch.ones(x.shape[1], dtype=bool)
        index[self.cart_indices_after_periodic] = False
        x_rest = x[:,index]
        batchsize = x_rest.shape[0]
        if self.GNN.num_evaluations == 0:

            batchsize = x_cart.shape[0]
            x_cart = x_cart.view(batchsize, -1,3)
            GNN_output = self.GNN(x_cart)  #actually call the GNN
            self.GNN._buffer=GNN_output  # save GNN output for eventual recycling by other conditioners.
            self.GNN.num_evaluations += 1

        else:
            GNN_output = self.GNN._buffer  #recycle the GNN output from earlier conditioner
            self.GNN.num_evaluations += 1


        if self.GNN.num_evaluations == self.GNN.num_uses:  #we are at the end of the generators flow.
            self.GNN.num_evaluations = 0  #the next conditioner call will actually call the GNN



        # each conditioner may use some attention on the generated feature vectors
        if self.attention_level is not None and self.attention_units > 20:
            warnings.warn("unsing Attention on the GNN feature vectors might crash as there are many of them.")
        if self.attention_level == "MHA":
            qkv = self.qkv_proj(GNN_output.view(batchsize, -1, self.attention_dim))
            q, k, v = qkv.chunk(3, dim=-1)
            GNN_output = self.MHA(q,k,v, need_weights = False)[0]
        elif self.attention_level == "Transformer":
            GNN_output = self.transformer_encoder(GNN_output.view(batchsize, -1, self.attention_dim))


        formatted_output = GNN_output.reshape(batchsize, -1)
        features = torch.cat([x_rest, formatted_output], dim=-1)

        ''' a dense NN computes the transforms parameters from the (wrapped)
            noncartesian inputs and the feature vector calculated from the cartesian inputs
        '''
        return self.densenet(features)


def _make_GNN_conditioner(dim_in, dim_out, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
    '''
    build an nequip GNN and plug it into the Transformer as conditioner network.
    '''

    GNN_conditioner = GNNConditioner(dim_in, dim_out, hidden, activation, **kwargs)
    return GNN_conditioner



CONDITIONER_FACTORIES = {
    "GNN": _make_GNN_conditioner,
    "dense": _make_dense_conditioner
}


def _spline_out_dims(what, shape_info, transformer_kwargs={}, num_bins=8, **kwargs):
    # input for conditioner
    dim_out = 3 * num_bins * shape_info.dim_all(what) + shape_info.dim_noncircular(what)
    return {"params_net": dim_out}


def _affine_out_dims(what, shape_info, transformer_kwargs={}, use_scaling=True, **kwargs):
    dim_out = shape_info.dim_all(what)
    out_dims = {"shift_transformation": dim_out}
    if use_scaling:
        out_dims["scale_transformation"] = dim_out
    return out_dims


def _mixture_out_dims(what, shape_info, transformer_kwargs={}, num_components=8, **kwargs):
    dim_out1 = num_components * shape_info.dim_all(what)
    return {"weights": dim_out1, "alphas": dim_out1, "params": 3*dim_out1}


CONDITIONER_OUT_DIMS = {
    bg.ConditionalSplineTransformer: _spline_out_dims,
    bg.AffineTransformer: _affine_out_dims,
    #TODO bg.MixtureCDFTransformer: _mixture_out_dims
}


