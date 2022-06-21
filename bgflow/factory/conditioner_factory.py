import warnings

from ipdb import set_trace as bp
import torch
import bgflow as bg
from ..nn.periodic import WrapPeriodic, WrapDistances, WrapDistancesIC
import numpy as np
from bgflow.factory.GNN_factory import CustomTransformerEncoderLayer
import traceback
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
    #bp()
    for name, dim in dim_out.items():
        #kwargs["cart_indices"] = shape_info.cartesian_indices(on)
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


class WrapDistancesConditioner(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden, activation, **kwargs):
        super().__init__()
        self.attention_level = kwargs["attention_level"]
        self.cart_indices = kwargs["cart_indices"]
        self.shape_info = kwargs["shape_info"]
        self.on = kwargs["on"]
        self.N = kwargs["N"]  # number of RBFs
        self.c = kwargs["c"]  # cutoff
        self.p = kwargs["p"]  # envelope parameter
        #bp()




        self.cart_indices_after_periodic = self.cart_indices + self.shape_info.dim_circular(self.on)
        self.wrapdistances = WrapDistances(torch.nn.Identity(), indices=self.cart_indices_after_periodic)
        self.wrapdistancespure = WrapDistances(torch.nn.Identity())
        self.n_cart = len(self.cart_indices)

        self.dim_dense_in = int(dim_in - self.n_cart + (((self.n_cart // 3) ** 2 - ((self.n_cart // 3))) / 2)*self.N)
        self.densenet = bg.DenseNet(
            [self.dim_dense_in, *hidden, dim_out],
            activation=activation
        )
        wavenumbers = torch.Tensor([np.pi*(i+1)/self.c for i in range(self.N)])
        self.wavenumbers = torch.nn.Parameter(wavenumbers)
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
        #taken from the paper: https://arxiv.org/pdf/2003.03123.pdf
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
        elif self.attention_level is None:
            binned_enveloped_distances = (u.unsqueeze(-1) * binned_distances).view(x.shape[0], -1)
            distances_output = binned_enveloped_distances
        else:
            raise ValueError("unrecognized attention level")



        inputs_and_distances = torch.cat([x_rest, distances_output], dim = -1)
        return self.densenet(inputs_and_distances)



def _make_wrapdistances_conditioner(dim_in, dim_out, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
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
    def __init__(self, dim_in, dim_out, hidden, activation, GNN_output_dim, **kwargs):
        super().__init__()

        ### sequential of wrapdistances, merge distances and other input, and this desne net:
        #self.attention_level = kwargs["attention_level"]
        self.attention_level = kwargs["attention_level"]
        #self.cart_indices = kwargs["cart_indices"]
        self.shape_info = kwargs["shape_info"]
        self.on = kwargs["on"]
        self.cart_indices = self.shape_info.cartesian_indices(self.on)

        self.cutoff = kwargs["r_max"]

        self.GNN_scope = kwargs["GNN_scope"]
        self.use_checkpointing = kwargs["use_checkpointing"]
        self.n_cart = len(self.cart_indices)
        self.n_cart_atoms = self.n_cart//3
        #bp()

        self.labels = []
        for field in self.on:
            if field.is_circular:
                self.labels = (["circular"]*self.shape_info[field][0]*2) + self.labels  #add the sin, cos indices on the left
            elif field.is_cartesian:
                self.labels.extend(["cartesian"] * self.shape_info[field][0])
            else:
                self.labels.extend(["regular"] * self.shape_info[field][0])
        self.cart_indices_after_periodic = np.where(np.array(self.labels) == "cartesian")[0]

        #bp()


        #self.cart_indices_after_periodic = self.cart_indices + self.shape_info.dim_circular(self.on)
        #bp()

        ## this has to reaturn a tensor of shape (batchsize, num_atoms or num_bonds, self.GNN_output_dim)
        if isinstance(kwargs["GNN"], torch.nn.Module):
            self.GNN = kwargs["GNN"]
            if hasattr(self.GNN, "num_uses"):
                self.GNN.num_uses +=1
            else: self.GNN.num_uses = 1

        elif callable(kwargs["GNN"]):
            self.GNN = kwargs["GNN"](**kwargs["GNN_kwargs"])
            self.GNN.num_uses = 1
        self.GNN.num_evaluations = 0


        if self.GNN_scope == "atomwise":
            """
            the GNN will create a fixed size atom_features vector for every node in the graph.
            you can specify how this vector is constructed from the node/edge features using 
            whatever GNN architecture, including Tensor products, local cutoff radii etc...
            """
            self.GNN_output_dim = GNN_output_dim
            self.GNN_out_dims = self.n_cart_atoms * self.GNN_output_dim
        elif self.GNN_scope == "bondwise":  ##
            """
            you can also feed the edge embeddings to the dense net, but this means you will not be able to have a local cutoff,
            as this could mean variable # of edge features...
            usually leads to strong overfitting (seen in the validation loss), but this does not harm sample quality.
            """

            self.GNN_output_dim = GNN_output_dim
            self.GNN_out_dims = (self.n_cart_atoms**2 - self.n_cart_atoms)*self.GNN_output_dim
        self.dim_dense_in = int(dim_in - self.n_cart + self.GNN_out_dims)
        self.densenet = bg.DenseNet(
            [self.dim_dense_in, *hidden, dim_out],
            activation=activation
        )

        ### define attention:
        if self.attention_level == "MHA":
            """
            on top of the GNN, have a single Multihead Attention on the atomwise/bondwise feature vectors
            """
            attention_dim = self.GNN_output_dim if self.GNN_scope == "atomwise" else self.latent_dim
            self.MHA = torch.nn.MultiheadAttention(embed_dim = attention_dim, num_heads = 8, batch_first = True)
            self.qkv_proj = torch.nn.Linear(self.GNN_output_dim, 3 * self.GNN_output_dim)
        elif self.attention_level == "Transformer":
            """
            on top of the GNN, have a full Transformer Encoder on the atomwise/bondwise feature vectors,
            consisting of MHA and dense layers, with layer norm and skipped connections
            """
            attention_dim = self.GNN_output_dim if self.GNN_scope == "atomwise" else self.latent_dim
            self.encoder_layer = CustomTransformerEncoderLayer(d_model=attention_dim,
                                                                  nhead=8,
                                                                  batch_first = True,
                                                                  dropout=0.,
                                                                  dim_feedforward=64)
            self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=1)
        elif self.attention_level is not None:
            raise ValueError("unrecognized attention level")

    def forward(self,x):
        if self.use_checkpointing:
            dummy_var = torch.zeros(2,requires_grad = True)  ## checkpointing does not work without having input that requires grad
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
        if self.GNN.num_evaluations == 0:

            batchsize = x_cart.shape[0]
            x_cart = x_cart.view(x_cart.shape[0],-1,3)
            distances = torch.cdist(x_cart, x_cart)
            in_bonds = distances <= self.cutoff
            indices = in_bonds.nonzero()
            indices = indices[indices[:,1] != indices[:,2]]  #remove edges from node to itself:
            edge_index = torch.vstack([indices[:,1],indices[:,2]])
            batch_index = indices[:,0]
            offset_tensor = (batch_index * self.n_cart_atoms).repeat((2, 1))

            edge_index_batch = edge_index + offset_tensor #make sure that all edges are in their respective graphs (multiple graphs in a batch)
            atom_types = torch.arange(self.n_cart_atoms).repeat(batchsize)

            batch = torch.arange(x_cart.shape[0]).repeat_interleave(self.n_cart_atoms)
            r_max = torch.full((batchsize,), self.cutoff)

            ## this is an atomicdatadict, it is used by the nequip GNNs to keep track of multiple graphs in a batch.
            ## The graphs may have a different number of atoms in their neighborhoods
            data = dict(pos=x_cart.view(-1,3),
                        edge_index=edge_index_batch.to(x_cart.device),
                        batch=batch.to(x_cart.device),
                        atom_types=atom_types.to(x_cart.device),
                        r_max=r_max.to(x_cart.device))


            GNN_output = self.GNN(data)  #actually call the GNN
            GNN_output = GNN_output.view(batchsize, -1, self.GNN_output_dim)


            self.GNN._buffer=GNN_output  # save GNN output for eventual recycling by other conditioners.
            self.GNN.num_evaluations +=1

        else:
            GNN_output = self.GNN._buffer  #recycle the GNN output from earlier conditioner
            self.GNN.num_evaluations += 1


        if self.GNN.num_evaluations == self.GNN.num_uses:  #we are at the end of the generators flow.
            self.GNN.num_evaluations = 0  #the next conditioner call will actually call the GNN



        # each conditioner may use some attention on the generated atomwise features
        if self.attention_level is not None and self.GNN_scope == "bondwise":
            warnings.warn("unsing Attention on the GNN Edge Features will must likely crash as there are too many of them.")
        if self.attention_level == "MHA":
            ## currently not working with bondwise GNN_scope
            qkv = self.qkv_proj(GNN_output)
            q, k, v = qkv.chunk(3, dim=-1)
            GNN_output,_ = self.MHA(q,k,v, need_weights = False)
        elif self.attention_level == "Transformer":
            ## currently not working with bondwise GNN_scope
            GNN_output = self.transformer_encoder(GNN_output)

        batchsize = x_rest.shape[0]
        formatted_output = GNN_output.reshape(batchsize, -1)
        features = torch.cat([x_rest, formatted_output], dim=-1)

        # a dense NN computes the transforms parameters from the (wrapped)
        # noncartesian inputs and the feature vectors calculated from the cartesian inputs
        return self.densenet(features)


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
