from nequip.nn import SequentialGraphNetwork
from nequip.nn.radial_basis import BesselBasis

from nequip.nn.embedding import (
    OneHotAtomEncoding
)

import nequip
import allegro
import torch
from allegro.nn import (
    NormalizedBasis,
    EdgewiseEnergySum,
    Allegro_Module,
    ScalarMLP,
)

# define the Normalized Basis that is also shifted a bit to increase stability:

class NormalizedBasis(torch.nn.Module):
    """Normalized version of a given radial basis.
    It is passed distance values from the training data so it can ensure that
     the RBFs output is normalized when passed data like this.

    Args:
        data: distance values that are exemplaric for the kind of data the RBFs will be passed, shape might be for example:
        torch.Size([40323])
        basis (constructor): callable to build the underlying basis
        basis_kwargs (dict): parameters for the underlying basis
        offset: in nm, the offset to apply to all distances to avoid large RBF values at small distance values

    """

    num_basis: int

    def __init__(
        self,
        data = None,
        original_basis=BesselBasis,
        original_basis_kwargs: dict = {},
        norm_basis_mean_shift: bool = True,
        offset = 1.
    ):
        super().__init__()
        self.offset = offset
        #### shift all entries to the right a bit.
        data += self.offset
        #### change r_max accordingly
        original_basis_kwargs["r_max"] += self.offset
        self.basis = original_basis(**original_basis_kwargs)
        self.num_basis = self.basis.num_basis

        with torch.no_grad():
            if data is None:
                raise ValueError("gotta pass data to inform the Basis Function")
            bs = self.basis(data)
            assert bs.ndim == 2
            if norm_basis_mean_shift:
                basis_std, basis_mean = torch.std_mean(bs, dim=0)
            else:
                basis_std = bs.square().mean().sqrt()
                basis_mean = torch.as_tensor(
                    0.0, device=basis_std.device, dtype=basis_std.dtype
                )
        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_o = x + self.offset
        return (self.basis(x_o) - self._mean) * self._inv_std




#gradient checkpointing to lower memory consumption at the cost of speed
use_checkpointing = False
#hyperparams
hparams = {
"r_max" : 1.2,
"num_types" : 10,
"num_basis" : 32, #8
"p" : 6,#48
"avg_num_neighbors" : 9, #
"num_layers" : 3,
"env_embed_multiplicity" : 32 ,#16
"latent_dim" : 32, #16 #512? 32 geht
"two_body_latent_indermediate_dims" : [64, 128, 256], #[64,64,64]#[64, 128, 256] #war mal 64 128 256 512 #64s
"nonscalars_include_parity" : False, #True
"irreps_edge_sh" :  '1x0e+1x1o+1x2e',# calculate only vectors and scalars :'1x0e+1x1o'
"RBF_distance_offset" : 1.
}


def make_config_dict(**kwargs):
    #bp()
    r_max = kwargs["r_max"]
    num_types = kwargs["num_types"]
    num_basis = kwargs["num_basis"]
    p = kwargs["p"]
    avg_num_neighbors = kwargs["avg_num_neighbors"]
    num_layers = kwargs["num_layers"]
    env_embed_multiplicity = kwargs["env_embed_multiplicity"]
    latent_dim = kwargs["latent_dim"]
    two_body_latent_indermediate_dims = kwargs["two_body_latent_indermediate_dims"]
    nonscalars_include_parity = kwargs["nonscalars_include_parity"]
    irreps_edge_sh = kwargs["irreps_edge_sh"]
    RBF_distance_offset = kwargs["RBF_distance_offset"]

             
    return  {'one_hot': (nequip.nn.embedding._one_hot.OneHotAtomEncoding,
                                    {'irreps_in': None, 'set_features': True, 'num_types': num_types}),


                   'radial_basis': (nequip.nn.embedding._edge.RadialBasisEdgeEncoding,
                                    {'basis': NormalizedBasis,
                                     'cutoff': nequip.nn.cutoffs.PolynomialCutoff,
                                     'basis_kwargs': {'data': None,
                                                      'original_basis': nequip.nn.radial_basis.BesselBasis,
                                                      'original_basis_kwargs': {'num_basis': num_basis,
                                                                                'trainable': True,
                                                                                'r_max': r_max},
                                                      'norm_basis_mean_shift': True,
                                                      'offset': RBF_distance_offset},
                                     'cutoff_kwargs': {'p': p, 'r_max': r_max},
                                     'out_field': 'edge_embedding'}),


                   'spharm': (nequip.nn.embedding._edge.SphericalHarmonicEdgeAttrs,
                                    {'edge_sh_normalization': 'component',
                                     'edge_sh_normalize': True,
                                     'out_field': 'edge_attrs',
                                     'irreps_edge_sh': irreps_edge_sh}),





                   'allegro': (allegro.nn._allegro.Allegro_Module, 
                                   {'avg_num_neighbors': avg_num_neighbors,
                                     'r_start_cos_ratio': 0.8, # unused
                                     'PolynomialCutoff_p': p,
                                     'per_layer_cutoffs': None,
                                     'cutoff_type': 'polynomial',
                                     'field': 'edge_attrs',
                                     'edge_invariant_field': 'edge_embedding',
                                     'node_invariant_field': 'node_attrs',
                                     'env_embed_multiplicity': env_embed_multiplicity,
                                     'embed_initial_edge': True,
                                     'linear_after_env_embed': False,
                                     'nonscalars_include_parity': nonscalars_include_parity,
                                     'two_body_latent': allegro.nn._fc.ScalarMLPFunction,
                                     'two_body_latent_kwargs': {'mlp_nonlinearity': 'silu',
                                                                'mlp_initialization': 'uniform',
                                                                'mlp_dropout_p': 0.0,
                                                                'mlp_batchnorm': False,#False
                                                                'mlp_latent_dimensions': [*two_body_latent_indermediate_dims, latent_dim]},
                                     'env_embed': allegro.nn._fc.ScalarMLPFunction,
                                     'env_embed_kwargs': {'mlp_nonlinearity': None,
                                                          'mlp_initialization': 'uniform',
                                                          'mlp_dropout_p': 0.0,
                                                          'mlp_batchnorm': False, #False
                                                          'mlp_latent_dimensions': []},
                                     'latent': allegro.nn._fc.ScalarMLPFunction,
                                     'latent_kwargs': {'mlp_nonlinearity': 'silu',
                                                       'mlp_initialization': 'uniform',
                                                       'mlp_dropout_p': 0.0,
                                                       'mlp_batchnorm': False, #False
                                                       'mlp_latent_dimensions': [latent_dim]},
                                     'latent_resnet': True,
                                     'latent_resnet_update_ratios': None,
                                     'latent_resnet_update_ratios_learnable': False,
                                     'latent_out_field': 'edge_features',
                                     'pad_to_alignment': 1,
                                     'sparse_mode': None,
                                     'r_max': r_max,
                                     'num_layers': num_layers,
                                     'num_types': num_types}),
                  'atomwise_gather': (allegro.nn._edgewise.EdgewiseReduce,
                                      {'field': 'edge_features',
                                       'out_field': 'outputs',
                                       'avg_num_neighbors': avg_num_neighbors}
                                     )
                  }



config_dict = make_config_dict(**hparams)
