"""High-level Builder API for Boltzmann generators."""

import warnings
import copy

import numpy as np
import torch
import logging
from ..nn.flow.sequential import SequentialFlow
from ..nn.flow.coupling import SetConstantFlow
from ..nn.flow.transformer.spline import ConditionalSplineTransformer
from ..nn.flow.coupling import CouplingFlow, SplitFlow, WrapFlow, MergeFlow
from ..nn.flow.crd_transform.ic import GlobalInternalCoordinateTransformation
from ..nn.flow.inverted import InverseFlow
from ..nn.flow.cdf import CDFTransform
from ..distribution.distribution import UniformDistribution
from ..distribution.normal import NormalDistribution
from ..distribution.product import ProductDistribution, ProductEnergy
from ..bg import BoltzmannGenerator
from .tensor_info import (
    TensorInfo, BONDS, ANGLES, TORSIONS, FIXED, ORIGIN, ROTATION, AUGMENTED, TARGET
)
from .conditioner_factory import make_conditioners
from .transformer_factory import make_transformer
from .distribution_factory import make_distribution
from .icmarginals import InternalCoordinateMarginals
#from ..utils.ff import lookup_bonds

__all__ = ["BoltzmannGeneratorBuilder"]


logger = logging.getLogger('bgflow')


def _tuple(thing):
    """Turn something into a tuple; everything but tuple and list is turned into a one-element tuple
    containing that thing.
    """
    if isinstance(thing, tuple) and not hasattr(thing, "_fields"):  # exclude namedtuples
        return thing
    elif isinstance(thing, list):
        return tuple(thing)
    else:
        return thing,


class BoltzmannGeneratorBuilder:
    """Builder class for Boltzmann Generators.

    Parameters
    ----------
    prior_dims : ShapeDictionary
        The tensor dimensions sampled by the prior.
    target : bgflow.Energy, optional
        The target energy that we want to learn to sample from.
    device : torch.device
        The device that the generator should run on.
    dtype : torch.dtype
        The data type that the generator should use.

    Attributes
    ----------
    default_transformer_type : bgflow.nn.flow.transformer.base.Transformer
        The transformer type that is used by default (default: bgflow.ConditionalSplineTransformer).
    default_transformer_kwargs : dict
        The default keyword arguments for the transformer construction (default: {}).
    default_conditioner_kwargs: dict
        The default keyword arguments for the conditioner construction (default: {}).
    default_prior_type : bgflow.distribution.distribution.Distribution
        The transformer type that is used by default (default: bgflow.UniformDistribution).
    default_prior_kwargs: dict
        The default keyword arguments for the prior construction (default: {}).
    ctx : dict
        A dictionary that contains the dtype and device.
    prior_dims : ShapeDictionary
        The shapes of tensors sampled by the prior;
        we often use product distributions as priors that can sample multiple tensors at once.
    current_dims : ShapeDictionary
        The "current" shapes; they are initialized with the prior shapes and changed
        according to the flow transformations that are added.
    layers : list of torch.nn.Module
        A list of flow transformations.
    constants : list of TensorInfo
        A list of tensors that are not sampled from the prior but set constant.
    # TODO


    Examples
    --------
    >>>  shape_info = ShapeDictionary()
    >>>  shape_info[BONDS] = (10, )
    >>>  shape_info[ANGLES] = (20, )
    >>>  builder = BoltzmannGeneratorBuilder(shape_info, device=torch.device("cpu"), dtype=torch.float32)
    >>>  split1 = TensorInfo("SPLIT_1")
    >>>  split2 = TensorInfo("SPLIT_2")
    >>>  # split angle priors into two channels
    >>>  builder.add_split(ANGLES, (split1, split2), (8, 12))
    >>>  # condition first on second angle channel, then condition bonds on first angle channel
    >>>  builder.add_condition(split1, on=split2)
    >>>  builder.add_condition(BONDS, on=split1)
    >>>  generator = builder.build_generator(zero_parameters=True)
    >>>  samples = generator.sample(11)

    """
    def __init__(self, prior_dims, target=None, device=None, dtype=None):
        self.default_transformer_type = ConditionalSplineTransformer
        self.default_transformer_kwargs = dict()
        self.default_conditioner_kwargs = dict()
        self.default_prior_type = UniformDistribution
        self.default_prior_kwargs = dict()

        self.ctx = {"device": device, "dtype": dtype}
        self.prior_dims = prior_dims
        self.current_dims = self.prior_dims.copy()
        self.layers = []
        self.constants = []
        # transformer and prior factories  (use defaults everwhere)
        self.transformer_type = dict()
        self.transformer_kwargs = dict()
        self.prior_type = dict()
        self.prior_kwargs = dict()
        self._constrained_bonds = dict()
        # default targets
        self.targets = dict()
        if target is not None:
            self.targets[TARGET] = target
        if AUGMENTED in self.prior_dims:
            dim = self.prior_dims[AUGMENTED]
            self.targets[AUGMENTED] = NormalDistribution(dim, torch.zeros(dim, **self.ctx))

        dimstring = "; ".join(f"{field.name}: {self.prior_dims[field]}"  for field in prior_dims)
        logger.info(f"BG Builder  :::  ({dimstring})")

    def build_generator(self, zero_parameters=False, check_target=True):
        """Build the Boltzmann Generator. The layers are cleared after building.

        Parameters
        ----------
        zero_parameters : bool, optional
            Whether the flow should be initialized with all trainable parameters set to zero.
        check_target : bool, optional
            Whether a warning is printed if not all output tensors have target energies.

        Returns
        -------
        generator : bgflow.bg.BoltzmannGenerator
            The Boltzmann generator.
        """
        # TODO (Jonas): if build_target returns None -> return Generator
        generator = BoltzmannGenerator(
            prior=self.build_prior(),
            flow=self.build_flow(zero_parameters=zero_parameters),
            target=self.build_target(check_target=check_target)
        )
        self.clear()
        return generator

    def build_flow(self, zero_parameters=False):
        """Build the normalizing flow.

        Parameters
        ----------
        zero_parameters : bool, optional
            Whether the flow should be initialized with all trainable parameters set to zero.

        Returns
        -------
        flow : bgflow.nn.flow.sequential.SequentialFlow
            The diffeomorphic transformation.
        """
        flow = SequentialFlow(self.layers)
        if zero_parameters:
            for p in flow.parameters():
                p.data.zero_()
        return flow

    def build_prior(self):
        """Build the prior.

        Returns
        -------
        prior : bgflow.energy.product.ProductDistribution
            The prior.
        """
        priors = []
        for field in self.prior_dims:
            if field in self.constants:
                continue
            prior_type = self.prior_type.get(field, self.default_prior_type)
            prior_kwargs = self.prior_kwargs.get(field, self.default_prior_kwargs)
            prior = make_distribution(
                distribution_type=prior_type,
                shape=self.prior_dims[field],
                **self.ctx,
                **prior_kwargs
            )
            priors.append(prior)
        if len(priors) > 1:
            return ProductDistribution(priors)
        else:
            return priors[0]

    def build_target(self, check_target=False):
        """Build the target energy.

        Parameters
        ----------
        check_target : bool, optional
            Whether a warning is printed if not all output tensors have target energies.

        Returns
        -------
        target : bgflow.energy.product.ProductEnergy
            The target energy.
        """
        targets = []
        for field in self.current_dims:
            if field in self.targets:
                targets.append(self.targets[field])
            elif check_target:
                warnings.warn(f"No target energy for {field}.", UserWarning)

        if len(targets) > 1:
            return ProductEnergy(targets)
        elif len(targets) == 1:
            return targets[0]
        else:
            return None

    def clear(self):
        """Remove all transform layers."""
        self.layers = []
        logger.info(f"--------------- cleared builder ----------------")
        self.current_dims = self.prior_dims.copy()

    def add_condition(self, what, on=tuple(), **kwargs):
        """Add a coupling layer, i.e. a transformation of the tensor `what`
        that is conditioned on the tensors `on`.

        Parameters
        ----------
        what : TensorInfo
            The tensor to be transformed.
        on : Sequence[TensorInfo]
            The tensor on which the transformation is conditioned.
        **kwargs : Keyword arguments
            Additional keyword arguments for the conditioner factory.

        Notes
        -----
        Always take transformer of what[0].

        """
        on = _tuple(on)
        if len(on) == 0:
            raise ValueError("Need to condition on something.")
        what = _tuple(what)
        if len(what) == 0:
            raise ValueError("Need to transform something.")

        transformer_types = [self.transformer_type.get(el, self.default_transformer_type) for el in what]
        transformer_type = transformer_types[0]
        if not all(ttype == transformer_type for ttype in transformer_types):
            raise ValueError("Fields with different transformer_type cannot be transformed together.")
        transformer_kwargss = [self.transformer_kwargs.get(el, self.default_transformer_kwargs) for el in what]
        transformer_kwargs = transformer_kwargss[0]
        if not all(tkwargs == transformer_kwargs for tkwargs in transformer_kwargss):
            raise ValueError("Fields with different transformer_kwargs cannot be transformed together.")

        conditioner_kwargs = copy.copy(self.default_conditioner_kwargs)
        conditioner_kwargs.update(kwargs)
        conditioners = make_conditioners(
            transformer_type=transformer_type,
            transformer_kwargs=transformer_kwargs,
            what=what,
            on=on,
            shape_info=self.current_dims,
            **conditioner_kwargs
        )
        transformer = make_transformer(
            transformer_type=transformer_type,
            what=what,
            shape_info=self.current_dims,
            conditioners=conditioners,
            **transformer_kwargs
        )
        coupling = CouplingFlow(
            transformer=transformer,
            transformed_indices=[self.current_dims.index(f) for f in what],
            cond_indices=[self.current_dims.index(f) for f in on]
        ).to(**self.ctx)
        logger.info(
            f"  + Coupling Layer: "
            f"({', '.join([field.name for field in on])}) "
            f"-> ({', '.join([field.name for field in what])})"
        )
        self.layers.append(coupling)

    def add_set_constant(self, what, tensor):
        if what in self.current_dims:
            if what in self.prior_dims:
                self.constants.append(what)
            else:
                if self.current_dims[what] != tuple(tensor.shape):
                    raise ValueError(f"Constant tensor {tensor} must have shape {self.current_dims[what]}")
        else:
            if what in self.prior_dims:
                raise ValueError(f"Cannot set {what} constant; field was already deleted or replaced.")
            else:
                self.current_dims[what] = tuple(tensor.shape)
        fix_flow = SetConstantFlow(
            indices=[self.current_dims.index(what)],
            values=[tensor]
        )
        logger.info(f"  + Set Constant: {what}")
        self.layers.append(fix_flow)

    def add_transform(self, layer, what=None):
        if what is None:
            self.layers.append(layer)
        else:
            return NotImplemented

    def add_split(self, what, into, sizes_or_indices, dim=-1):
        into = list(into)
        for i, el in enumerate(into):
            if isinstance(el, str):
                into[i] = TensorInfo(name=el, is_circular=what.is_circular)
        input_index = self.current_dims.index(what)
        split_flow = SplitFlow(*sizes_or_indices, dim=dim)
        if split_flow._sizes is None:
            sizes = (len(size) for size in sizes_or_indices)
        else:
            sizes = sizes_or_indices
        self.current_dims.split(what, into, sizes, dim=dim)
        output_indices = [self.current_dims.index(el) for el in into]
        wrap_flow = WrapFlow(split_flow, indices=(input_index,), out_indices=output_indices)
        logger.info(f"  + Split: {what.name} -> ({', '.join([field.name for field in into])})")
        self.layers.append(wrap_flow)
        return tuple(into)

    def add_merge(self, what, to, dim=-1, output_index=None, sizes_or_indices=None):
        if isinstance(to, str):
            to = TensorInfo(name=to, is_circular=what[0].is_circular)
        if not all(w.is_circular == to.is_circular for w in what):
            raise ValueError(
                "Merging non-circular with circular tensors is dangerous. "
                "Found discrepancies in f{what} and f{to}."
            )
        input_indices = [self.current_dims.index(el) for el in what]
        if sizes_or_indices is None:
            sizes_or_indices = [self.current_dims[el][dim] for el in what]
        merge_flow = MergeFlow(*sizes_or_indices, dim=dim)
        self.current_dims.merge(what, to=to, index=output_index)
        output_index = self.current_dims.index(to)
        wrap_flow = WrapFlow(merge_flow, indices=input_indices, out_indices=(output_index,))
        logger.info(f"  + Merge: ({', '.join([field.name for field in what])}) -> {[to.name]}")
        self.layers.append(wrap_flow)
        return to

    def add_map_to_cartesian(
            self,
            coordinate_transform,
            fixed_origin_and_rotation=True,
            bonds=BONDS,
            angles=ANGLES,
            torsions=TORSIONS,
            fixed=FIXED,
            origin=ORIGIN,
            rotation=ROTATION,
            out=TARGET
    ):
        ic_fields = [bonds, angles, torsions]

        # fix origin and rotation
        if isinstance(coordinate_transform, GlobalInternalCoordinateTransformation):
            ic_fields.extend([origin, rotation])
            if fixed_origin_and_rotation:
                self.add_set_constant(origin, torch.zeros(1, 3, **self.ctx))
                self.add_set_constant(rotation, torch.eye(3, **self.ctx).unsqueeze(0))
        else:
            ic_fields.append(fixed)

        # merge constrained bonds
        if bonds in self._constrained_bonds:
            constrained_bond_indices, unconstrained_bond_indices, constrained_lengths = self._constrained_bonds[bonds]
            constrained_bonds = TensorInfo(f"{bonds.name}_constrained", bonds.is_circular)
            self.add_set_constant(constrained_bonds, constrained_lengths)
            self.add_merge(
                (bonds, constrained_bonds),
                to=bonds,
                sizes_or_indices=(unconstrained_bond_indices, constrained_bond_indices)
            )

        indices = [self.current_dims.index(ic) for ic in ic_fields]
        wrap_around_ics = WrapFlow(
            InverseFlow(coordinate_transform),
            indices=indices,
            out_indices=(self.current_dims.index(bonds),)
        )
        self.current_dims.merge(ic_fields, out)
        self.layers.append(wrap_around_ics)

    def add_map_to_ic_domains(self, cdfs=dict()):
        if len(cdfs) == 0:
            cdfs = InternalCoordinateMarginals(self.current_dims, self.ctx)
        for field in cdfs:
            if field in self.current_dims:
                icdf_flow = InverseFlow(CDFTransform(cdfs[field]))
                self.layers.append(WrapFlow(icdf_flow, (self.current_dims.index(field),)))
            else:
                warnings.warn(f"Field {field} not in current dims. CDF is ignored.")

    def set_constrained_bonds(self, system, coordinate_transform, bonds=BONDS):
        # parse constraints
        lengths, force_constants = lookup_bonds(system, coordinate_transform.bond_indices, temperature=1.0)
        constrained_bond_indices = []
        constrained_bond_lengths = []
        for i, (length, force_constant) in enumerate(zip(lengths, force_constants)):
            if np.isinf(force_constant):
                constrained_bond_indices.append(i)
                constrained_bond_lengths.append(length)
        unconstrained_bond_indices = np.setdiff1d(np.arange(self.prior_dims[bonds][-1]), constrained_bond_indices)

        # impose
        if len(constrained_bond_indices) > 0:
            self._constrained_bonds[bonds] = (
                constrained_bond_indices,
                unconstrained_bond_indices.tolist(),
                torch.tensor(constrained_bond_lengths, **self.ctx)
            )
            if len(self.layers) > 0:
                warnings.warn(
                    "Changing prior dimensions on a builder that has layers. "
                    "This may break the layers that have already been added.",
                    UserWarning
                )
            self.prior_dims[bonds] = (self.prior_dims[bonds][-1] - len(constrained_bond_indices), )
            self.current_dims[bonds] = self.prior_dims[bonds]

