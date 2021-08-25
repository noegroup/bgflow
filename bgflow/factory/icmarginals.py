
import numpy as np
import torch

from .tensor_info import BONDS, ANGLES, TORSIONS, FIXED, AUGMENTED
from ..distribution.normal import TruncatedNormalDistribution
from ..distribution.distribution import SloppyUniform
#from ..utils.ff import lookup_bonds, lookup_angles

__all__ = ["InternalCoordinateMarginals"]


class InternalCoordinateMarginals(dict):
    def __init__(
            self,
            current_dims,
            ctx,
            bond_mu=1.0,
            bond_sigma=1.0,
            bond_lower=1e-5,
            bond_upper=np.infty,
            angle_mu=0.5,
            angle_sigma=1.0,
            angle_lower=1e-5,
            angle_upper=1.0,
            torsion_lower=0.0,
            torsion_upper=1.0,
            fixed_scale=20.0,
            bonds=BONDS,
            angles=ANGLES,
            torsions=TORSIONS,
            fixed=FIXED,
            augmented=AUGMENTED,
    ):
        self.ctx = ctx
        self.current_dims = current_dims

        super().__init__()
        if bonds in current_dims:
            self[bonds] = TruncatedNormalDistribution(
                mu=bond_mu*torch.ones(current_dims[bonds], **ctx),
                sigma=bond_sigma*torch.ones(current_dims[bonds], **ctx),
                lower_bound=torch.tensor(bond_lower, **ctx),
                upper_bound=torch.tensor(bond_upper, **ctx),
            )

        # angles
        if angles in current_dims:
            self[angles] = TruncatedNormalDistribution(
                mu=angle_mu*torch.ones(current_dims[angles], **ctx),
                sigma=angle_sigma*torch.ones(current_dims[angles], **ctx),
                lower_bound=torch.tensor(angle_lower, **ctx),
                upper_bound=torch.tensor(angle_upper, **ctx),
            )

        # angles
        if torsions in current_dims:
            self[torsions] = SloppyUniform(
                low=torsion_lower*torch.ones(current_dims[torsions], **ctx),
                high=torsion_upper*torch.ones(current_dims[torsions], **ctx)
            )

        # fixed
        if fixed in current_dims:
            self[fixed] = torch.distributions.Normal(
                loc=torch.zeros(current_dims[fixed], **ctx),
                scale=fixed_scale*torch.ones(current_dims[fixed], **ctx)
            )

        # augmented
        if augmented in current_dims:
            self[augmented] = torch.distributions.Normal(
                loc=torch.zeros(current_dims[augmented], **ctx),
                scale=torch.ones(current_dims[augmented], **ctx)
            )

    def __eq__(self, other):
        return super().__eq__(other) and (other.ctx == self.ctx)

    def inform_with_force_field(
            self,
            system,
            coordinate_transform,
            temperature,
            bonds=BONDS,
            angles=ANGLES,
            torsions=TORSIONS
    ):
        import bgmol
        if bonds in self.current_dims:
            self[bonds] = bgmol.bond_marginal_estimate(
                system, coordinate_transform, temperature, **self.ctx
            )
        if angles in self.current_dims:
            self[angles] = bgmol.angle_marginal_estimate(
                system, coordinate_transform, temperature, **self.ctx
            )

    def inform_with_data(
            self,
            data,
            coordinate_transform,
            bond_lower=1e-5,
            bond_upper=np.infty,
            angle_lower=1e-5,
            angle_upper=1.0,
            torsion_lower=0.0,
            torsion_upper=1.0,
            torsion_bins=50,
            constrained_bond_indices=None,
            bonds=BONDS,
            angles=ANGLES,
            torsions=TORSIONS,
    ):
        bond_values, angle_values, torsion_values, *_ = coordinate_transform.forward(data)

        if bonds in self.current_dims:
            bond_mu = bond_values.mean(axis=0)
            bond_sigma = bond_values.std(axis=0)
            if constrained_bond_indices is not None:
                bond_mu = bond_mu[constrained_bond_indices]
                bond_sigma = bond_sigma[constrained_bond_indices]
            self[bonds] = TruncatedNormalDistribution(
                mu=torch.tensor(bond_mu, **self.ctx),
                sigma=torch.tensor(bond_sigma, **self.ctx),
                lower_bound=torch.tensor(bond_lower, **self.ctx),
                upper_bound=torch.tensor(bond_upper, **self.ctx),
            )

        if angles in self.current_dims:
            angle_mu = angle_values.mean(axis=0)
            angle_sigma = angle_values.std(axis=0)
            self[angles] = TruncatedNormalDistribution(
                mu=torch.tensor(angle_mu, **self.ctx),
                sigma=torch.tensor(angle_sigma, **self.ctx),
                lower_bound=torch.tensor(angle_lower, **self.ctx),
                upper_bound=torch.tensor(angle_upper, **self.ctx),
            )

        torsion_values = torsion_values.detach().cpu().numpy()
        density, edges = np.histogram(torsion_values, range=(torsion_lower, torsion_upper), density=True)
        density
        edges


        


