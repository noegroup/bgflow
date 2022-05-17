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

def _make_schnett_conditioner(dim_in, dim_out, cartesian_indices, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
    pass


class WrapDistancesConditioner(torch.nn.Module):
    def __init__(self, dim_in, dim_out, hidden, activation, **kwargs):
        super().__init__()
        ### sequential of wrapdistances, merge distances and other input, and this desne net:
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
        binned_enveloped_distances = (u.unsqueeze(-1)*binned_distances).view(x.shape[0],-1)
        inputs_and_distances = torch.cat([x_rest, binned_enveloped_distances], dim = -1)
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



class AllegroConditioner(torch.nn.Module):
    '''
    split input into wrapped periodic, nonperiodic and cartesians.
    On cartesians, apply allegro GNN.
    then apply dense net on (periodic || nonperiodic || GNN outputs)
    '''
    def __init__(self, dim_in, dim_out, hidden, activation, **kwargs):
        super().__init__()
        ### sequential of wrapdistances, merge distances and other input, and this desne net:
        self.cart_indices = kwargs["cart_indices"]
        self.shape_info = kwargs["shape_info"]
        self.cutoff = kwargs["c"]
        self.on = kwargs["on"]
        self.layers = kwargs["layers"]
        self.n_cart = len(self.cart_indices)
        self.n_cart_atoms = self.n_cart//3
        self.cart_indices_after_periodic = self.cart_indices + self.shape_info.dim_circular(self.on)
        self.allegro_gnn = SequentialGraphNetwork.from_parameters(shared_params=None, layers=self.layers)
        self.allegro_out_dims = self.n_cart_atoms * kwargs["layers"]["allegro"][1]["latent_kwargs"]["mlp_latent_dimensions"][-1] #TODO# # consider each edge only once
        self.dim_dense_in = int(dim_in - self.n_cart + self.allegro_out_dims)
        self.densenet = bg.DenseNet(
            [self.dim_dense_in, *hidden, dim_out],
            activation=activation
        )



    def forward(self, x):
        x_cart = x[:,self.cart_indices_after_periodic]
        index = torch.ones(x.shape[1], dtype=bool)
        index[self.cart_indices_after_periodic] = False
        x_rest = x[:,index]
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


        allegro_output = self.allegro_gnn(data)
        formatted_output = allegro_output["outputs"].view(batchsize, -1)
        inputs_and_distances = torch.cat([x_rest, formatted_output], dim = -1)
        return self.densenet(inputs_and_distances)


def _make_allegro_conditioner(dim_in, dim_out, hidden=(128, 128), activation=torch.nn.SiLU(), **kwargs):
    '''
    build an allegro GNN and plug it into the Transformer as conditioner network.
    '''

    allegro_conditioner = AllegroConditioner(dim_in, dim_out, hidden, activation, **kwargs)
    return allegro_conditioner



CONDITIONER_FACTORIES = {
    "allegro": _make_allegro_conditioner,
    "dense": _make_dense_conditioner,
    #"schnett": _make_schnett_conditioner,
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
