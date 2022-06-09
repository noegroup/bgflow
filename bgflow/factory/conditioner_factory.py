from ipdb import set_trace as bp
import torch
import bgflow as bg
from ..nn.periodic import WrapPeriodic, WrapDistances, WrapDistancesIC
import numpy as np
from bgflow.factory.tensor_info import FIXED

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
        wrap_fixed = False,
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
        kwargs["cart_indices"] = shape_info.cartesian_indices(on)
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


#####
#transformerencoderlayer
from typing import Optional, Any, Union, Callable

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
class CustomTransformerEncoderLayer(Module):
    r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
    This standard encoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to attention and feedforward
            operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).

    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> src = torch.rand(10, 32, 512)
        >>> out = encoder_layer(src)

    Alternatively, when ``batch_first`` is ``True``:
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> src = torch.rand(32, 10, 512)
        >>> out = encoder_layer(src)
    """
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(CustomTransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        self.qkv_proj = torch.nn.Linear(d_model, 3 * d_model)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(CustomTransformerEncoderLayer, self).__setstate__(state)

    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
        r"""Pass the input through the encoder layer.

        Args:
            src: the sequence to the encoder layer (required).
            src_mask: the mask for the src sequence (optional).
            src_key_padding_mask: the mask for the src keys per batch (optional).

        Shape:
            see the docs in Transformer class.
        """

        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        x = self.self_attn(q, k, v,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)





######




class WrapDistancesConditioner(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden, activation, **kwargs):
        super().__init__()
        ### sequential of wrapdistances, merge distances and other input, and this desne net:
        self.attention_level = kwargs["attention_level"]
        self.cart_indices = kwargs["cart_indices"]
        self.shape_info = kwargs["shape_info"]
        self.on = kwargs["on"]
        self.N = kwargs["N"] #number of RBFs
        self.c = kwargs["c"] # cutoff
        self.p = kwargs["p"] # envelope parameter
        self.cart_indices_after_periodic = self.cart_indices + self.shape_info.dim_circular(self.on)
        self.wrapdistances = WrapDistances(torch.nn.Identity(), indices=self.cart_indices_after_periodic)
        self.wrapdistancespure = WrapDistances(torch.nn.Identity())
        self.n_cart = len(self.cart_indices)

        self.dim_dense_in = int(dim_in - self.n_cart + (((self.n_cart // 3) ** 2 - ((self.n_cart // 3))) / 2)*self.N)
        self.densenet = bg.DenseNet(
            [self.dim_dense_in, *hidden, dim_out],
            activation=activation
        )
        #self.envelope = Envelope(p = self.p, c = self.c)
        wavenumbers = torch.Tensor([np.pi*(i+1)/self.c for i in range(self.N)])
        self.wavenumbers = torch.nn.Parameter(wavenumbers)
        ### define attention:
        if self.attention_level == "MHA":
            self.MHA = torch.nn.MultiheadAttention(embed_dim = self.N, num_heads = 8, batch_first = True)

            self.qkv_proj = torch.nn.Linear(self.N, 3 * self.N)

        if self.attention_level == "Transformer":
            self.encoder_layer = CustomTransformerEncoderLayer(d_model=self.N,
                                                                  nhead=2,
                                                                  batch_first = True,
                                                                  dropout=0.,
                                                                  dim_feedforward=64)
            self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=2)



    def bessel(self, x, wavenumber, c):
        return ((2 / c) ** 0.5 * torch.sin(wavenumber * x) / x)
    def envelope(self, x, c, p):
        x = x/c
        return(1 - ((p+1)*(p+2))/2*x**p + p*(p+2)*x**(p+1) - (p*(p+1))/2*x**(p+2))
    def forward(self, x):
        x_cart = x[:,self.cart_indices_after_periodic]
        index = torch.ones(x.shape[1], dtype=bool)
        index[self.cart_indices_after_periodic] = False
        x_rest = x[:,index]
        distances = self.wrapdistancespure(x_cart)
        binned_distances = self.bessel(distances[...,None], self.wavenumbers, self.c)
        u = self.envelope(distances, c=self.c, p=self.p)

        if self.attention_level == "MHA":
            binned_enveloped_distances = (u.unsqueeze(-1) * binned_distances)#.view(x.shape[0], -1)
            ## feed this into the dense layer
            #bp()
            qkv = self.qkv_proj(binned_enveloped_distances)
            q, k, v = qkv.chunk(3, dim=-1)
            bondwise_output,_ = self.MHA(q,k,v, need_weights = False)
            batchsize = x_rest.shape[0]

            distances_output = bondwise_output.reshape(batchsize, -1)
        #### end attention on atom features
        elif self.attention_level == "Transformer":
            binned_enveloped_distances = (u.unsqueeze(-1) * binned_distances)
            bondwise_output = self.transformer_encoder(binned_enveloped_distances)
            batchsize = x_rest.shape[0]

            distances_output = bondwise_output.reshape(batchsize, -1)
        else:
            binned_enveloped_distances = (u.unsqueeze(-1) * binned_distances).view(x.shape[0], -1)
            distances_output = binned_enveloped_distances




        inputs_and_distances = torch.cat([x_rest, distances_output], dim = -1)
        return self.densenet(inputs_and_distances)



def _make_wrapdistances_conditioner(dim_in, dim_out, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
    ''' take as input the flattened cartesian coordinates and other inputs
    flatten them
    make distance matrix
    subset with mask
    return distances and noncartesian (circular) inputs
    do dense net on these
    (use rbf)
    maybe also wrap the periodics in here and keep book with a shape dictionary
    '''
    wrapdistances_conditioner = WrapDistancesConditioner(dim_in, dim_out, hidden, activation, **kwargs)
    return wrapdistances_conditioner


from nequip.nn import SequentialGraphNetwork
import bgflow as bg
from torch.utils.checkpoint import checkpoint


class GNNConditioner(torch.nn.Module):
    '''
    split input into wrapped periodic, nonperiodic and cartesians.
    On cartesians, apply GNN implemented using the nequip library.
    then apply dense net on (periodic || nonperiodic || GNN outputs)
    '''
    def __init__(self, dim_in, dim_out, hidden, activation, GNN_output_dim, attention_level=None, GNN_scope="bondwise", **kwargs):
        super().__init__()
        #bp()

        ### sequential of wrapdistances, merge distances and other input, and this desne net:
        #self.attention_level = kwargs["attention_level"]
        self.attention_level = attention_level
        self.cart_indices = kwargs["cart_indices"]
        self.shape_info = kwargs["shape_info"]
        self.cutoff = kwargs["r_max"]
        self.on = kwargs["on"]
        self.many_GNNs = kwargs["many_GNNs"]
        self.GNN_scope = GNN_scope
        #self.layers = kwargs["layers"]
        self.use_checkpointing = kwargs["use_checkpointing"]
        self.first = kwargs["first"] if "first" in kwargs.keys() else False
        self.n_cart = len(self.cart_indices)
        self.n_cart_atoms = self.n_cart//3
        self.cart_indices_after_periodic = self.cart_indices + self.shape_info.dim_circular(self.on)
        # = kwargs["many_GNNs"]
        if isinstance(kwargs["GNN_feature_extractor"], list):
            self.GNN = kwargs["GNN_feature_extractor"].pop()
        else:
            self.GNN = kwargs["GNN_feature_extractor"]

        self.GNN._buffer = []
          #  SequentialGraphNetwork.from_parameters(shared_params=None, layers=self.layers)
        #bp()

        if self.GNN_scope == "atomwise":
            #self.atomwise_feature_dim = self.GNN.atomwise_linear._modules["linear"].instructions[0].path_shape[1]
            self.atomwise_feature_dim = GNN_output_dim
            self.GNN_out_dims = self.n_cart_atoms * self.atomwise_feature_dim
        elif self.GNN_scope == "bondwise":
            #self.latent_dim = kwargs["GNN_feature_extractor"].allegro.latents[0].out_features
            self.latent_dim = GNN_output_dim
            self.GNN_out_dims = (self.n_cart_atoms**2 - self.n_cart_atoms)*self.latent_dim
        #self.GNN_out_dims = self.n_cart_atoms * self.GNN.GNN.final_latent.out_features  ##CHANGE THIS
        self.dim_dense_in = int(dim_in - self.n_cart + self.GNN_out_dims)
        self.densenet = bg.DenseNet(
            [self.dim_dense_in, *hidden, dim_out],
            activation=activation
        )
        #self.GNN_buffer = kwargs["GNN_buffer"]
        ### define attention:
        if self.attention_level == "MHA":
            attention_dim = self.atomwise_feature_dim if self.GNN_scope == "atomwise" else self.latent_dim
            self.MHA = torch.nn.MultiheadAttention(embed_dim = attention_dim, num_heads = 8, batch_first = True)

            self.qkv_proj = torch.nn.Linear(self.atomwise_feature_dim, 3 * self.atomwise_feature_dim)
        if self.attention_level == "Transformer":
            attention_dim = self.atomwise_feature_dim if self.GNN_scope == "atomwise" else self.latent_dim
            self.encoder_layer = CustomTransformerEncoderLayer(d_model=attention_dim,
                                                                  nhead=8,
                                                                  batch_first = True,
                                                                  dropout=0.,
                                                                  dim_feedforward=64)
            self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=1)


    def forward(self,x):
        if self.use_checkpointing:
            dummy_var = torch.zeros(2,requires_grad = True)
            #dummy_var.requires_grad = True
            #x.requires_grad = True
            #bp()
            print("checkpoint passed")
            return checkpoint(self._forward, x, dummy_var)
        else:
            return self._forward(x)

    def _forward(self, x, dummy_var = None):
        x_cart = x[:,self.cart_indices_after_periodic]
        index = torch.ones(x.shape[1], dtype=bool)
        index[self.cart_indices_after_periodic] = False
        x_rest = x[:,index]
        GNN_calculated = False
        #if True:
        if self.first and len(self.GNN._buffer) == 0:
            #bp()
            GNN_calculated = True
            batchsize = x_cart.shape[0]

            x_cart = x_cart.view(x_cart.shape[0],-1,3)
            distances = torch.cdist(x_cart, x_cart)
            in_bonds = distances <= self.cutoff
            ###
            #remove edges from node to itself:
            indices = in_bonds.nonzero()
            indices = indices[indices[:,1] != indices[:,2]]
            edge_index = torch.vstack([indices[:,1],indices[:,2]])
            batch_index = indices[:,0]
            offset_tensor = (batch_index * self.n_cart_atoms).repeat((2, 1))

            edge_index_batch = edge_index + offset_tensor #make sure that all edges are in their respective graphs ( multiple graphs in a batch)

            atom_types = torch.arange(self.n_cart_atoms).repeat(batchsize)

            batch = torch.arange(x_cart.shape[0]).repeat_interleave(self.n_cart_atoms)
            r_max = torch.full((batchsize,), self.cutoff)

            data = dict(pos=x_cart.view(-1,3),
                        edge_index=edge_index_batch.to(x_cart.device),
                        batch=batch.to(x_cart.device),
                        atom_types=atom_types.to(x_cart.device),
                        r_max=r_max.to(x_cart.device))


            #GNN_output = checkpoint(self.GNN,data)


            GNN_output = self.GNN(data)
            #()
            if self.GNN_scope == "atomwise":
                GNN_output = GNN_output["outputs"].view(batchsize, -1, self.atomwise_feature_dim)
                #self.GNN._buffer.append(GNN_output)
            elif self.GNN_scope == "bondwise":
                GNN_output = GNN_output["edge_features"].view(batchsize, -1, GNN_output["edge_features"].shape[-1])
                #bp()
                #GNN_output =#reduce number of bond features by two?

                #bp()#.view(batchsize, -1, self.atomwise_feature_dim)
            #bp()
            if not self.many_GNNs:
                self.GNN._buffer.append(GNN_output)
            #bp()
            #self.GNN_buffer.append(GNN_output)


            #assert len(GNN_buffer) == 1, "GNN buffer is "



        else:
            #bp()
            GNN_output = self.GNN._buffer[0]


        if self.first and len(self.GNN._buffer) != 0 and GNN_calculated == False:
           #bp()
            self.last_buffer = self.GNN._buffer.pop()
            del self.last_buffer


        #### insert attention on the atom features here
        #multihead attention: generate QKV from every atom vector,
        #get attention matrix by Q*K1,K2...
        #sum values with their attention weights.
        #concatenate, Linear layer
        if self.attention_level == "MHA":
            ## feed this into the dense layer
            ## currently not working with bondwise GNN_scope
            #bp()
            qkv = self.qkv_proj(GNN_output)
            q, k, v = qkv.chunk(3, dim=-1)
            GNN_output,_ = self.MHA(q,k,v, need_weights = False)
        #### end attention on atom features
        elif self.attention_level == "Transformer":
            ## currently not working with bondwise GNN_scope

            GNN_output = self.transformer_encoder(GNN_output)
            #batchsize = x_rest.shape[0]

            #distances_output = bondwise_output.reshape(batchsize, -1)
        batchsize = x_rest.shape[0]

        formatted_output = GNN_output.reshape(batchsize, -1)

        inputs_and_distances = torch.cat([x_rest, formatted_output], dim = -1)


        return self.densenet(inputs_and_distances)


def _make_GNN_conditioner(dim_in, dim_out, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
    '''
    build an nequip GNN and plug it into the Transformer as conditioner network.
    '''

    GNN_conditioner = GNNConditioner(dim_in, dim_out, hidden, activation, **kwargs)
    return GNN_conditioner



CONDITIONER_FACTORIES = {
    "GNN": _make_GNN_conditioner,
    "dense": _make_dense_conditioner,
    "wrapdistances": _make_wrapdistances_conditioner,
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
