"""High-level Builder API for Boltzmann generators."""

import warnings
from typing import Mapping, Sequence

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
from ..nn.flow.base import Flow
from ..nn.flow.modulo import IncreaseMultiplicityFlow
from ..nn.flow.modulo import CircularShiftFlow
from ..nn.flow.torchtransform import TorchTransform
from ..distribution.distributions import UniformDistribution
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
    default_prior_type : bgflow.distribution.distributions.Distribution
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
    param_groups : Mapping[str, list[torch.nn.Parameter]]
        Parameter groups indexed by group names.
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
        self.default_conditioner_type = "dense"
        self.default_transformer_kwargs = dict()
        self.default_conditioner_kwargs = dict()
        self.default_prior_type = UniformDistribution
        self.default_prior_kwargs = dict()

        self.ctx = {"device": device, "dtype": dtype}
        self.prior_dims = prior_dims
        self.current_dims = self.prior_dims.copy()
        self.layers = []
        # transformer and prior factories  (use defaults everwhere)
        self.transformer_type = dict()
        self.transformer_kwargs = dict()
        self.conditioner_type = dict()
        self.conditioner_kwargs = dict()
        self.prior_type = dict()
        self.prior_kwargs = dict()
        # default targets
        self.targets = dict()
        if target is not None:
            self.targets[TARGET] = target
        if AUGMENTED in self.prior_dims:
            dim = self.prior_dims[AUGMENTED]
            self.targets[AUGMENTED] = NormalDistribution(dim, torch.zeros(dim, **self.ctx))
        self.param_groups = dict()
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
            warnings.warn("Initializing the flow with zeros makes it much less flexible", UserWarning)
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

    def add_condition(
            self,
            what,
            on=tuple(),
            param_groups=tuple(),
            conditioner_type=None,
            transformer_type=None,
            transformer_kwargs=dict(),
            **conditioner_kwargs
    ):
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

        if transformer_type is None:
            transformer_types = [self.transformer_type.get(el, self.default_transformer_type) for el in what]
            if not all(ttype == transformer_types[0] for ttype in transformer_types):
                raise ValueError("Fields with different transformer_type cannot be transformed together.")
            transformer_type = transformer_types[0]

        transformer_kwargss = [self.transformer_kwargs.get(el, self.default_transformer_kwargs) for el in what]
        transformer_kwargss = [{**defaults, **transformer_kwargs} for defaults in transformer_kwargss]
        if not all(tkwargs == transformer_kwargss[0] for tkwargs in transformer_kwargss):
            raise ValueError("Fields with different transformer_kwargs cannot be transformed together.")
        transformer_kwargs = transformer_kwargss[0]

        if conditioner_type is None:
            conditioner_types = [self.conditioner_type.get(el, self.default_conditioner_type) for el in what]
            if not all(ttype == conditioner_types[0] for ttype in conditioner_types):
                raise ValueError("Fields with different conditioner_type cannot be transformed together.")
            conditioner_type = conditioner_types[0]

        conditioner_kwargss = [self.conditioner_kwargs.get(el, self.default_conditioner_kwargs) for el in what]
        conditioner_kwargss = [{**defaults, **conditioner_kwargs} for defaults in conditioner_kwargss]
        if not all(ckwargs == conditioner_kwargss[0] for ckwargs in conditioner_kwargss):
            raise ValueError("Fields with different conditioner_kwargs cannot be transformed together.")
        conditioner_kwargs = conditioner_kwargss[0]
        conditioners = make_conditioners(
            transformer_type=transformer_type,
            conditioner_type=conditioner_type,
            transformer_kwargs=transformer_kwargs,
            what=what,
            on=on,
            shape_info=self.current_dims.copy(),
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
        self.add_layer(coupling, param_groups=param_groups)

    def add_set_constant(self, what, tensor):
        if what in self.current_dims:
            if self.current_dims[what] != tuple(tensor.shape):
                raise ValueError(f"Constant tensor {tensor} must have shape {self.current_dims[what]}")
        else:
            if what in self.prior_dims:
                raise ValueError(f"Cannot set {what} constant; field was already deleted or replaced.")
            else:
                self.current_dims[what] = tuple(tensor.shape)
        index = self.current_dims.index(what)
        fix_flow = SetConstantFlow(
            indices=[index],
            values=[tensor.to(**self.ctx)]
        )
        logger.info(f"  + Set Constant: {what} at index {index}")
        self.layers.append(fix_flow)
    

    def add_layer(self, flow, what=None, inverse=False, param_groups=tuple()):
        """Add a flow layer.
        The layer must not change the dimensions of the
        tensors that it is being applied to.

        Parameters
        ----------
        flow : bgflow.Flow
            The flow transform that is applied to all inputs or a selection of inputs.
        what : Sequence[bflow.TensorInfo], optional
            The fields that the flow is applied to.
            If None, apply the flow to all fields in current_dims.
        inverse : bool, optional
            If True, add the inverse flow.
        param_groups : Sequence[str]
            A list of group names.
        """
        if what is not None:
            what = _tuple(what)
        if inverse:
            flow = InverseFlow(flow)
        if what is not None:
            # wrap flow
            input_indices = [self.current_dims.index(el) for el in what]
            output_indices = input_indices
            flow = WrapFlow(flow, input_indices, output_indices)
        self._add_to_param_groups(flow.parameters(), param_groups)
        self.layers.append(flow)


    def add_split(self, what, into, sizes_or_indices, dim=-1):
        into = list(into)
        for i, el in enumerate(into):
            if isinstance(el, str):
                into[i] = TensorInfo(name=el, is_circular=what.is_circular)
        input_index = self.current_dims.index(what)
        split_flow = SplitFlow(*sizes_or_indices, dim=dim)
        if split_flow._sizes is None:
            sizes = [len(size) for size in sizes_or_indices]
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
                f"Merging non-circular with circular tensors is dangerous and therefore disabled. "
                f"Found discrepancies in f{what} and f{to}."
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
                self.add_set_constant(rotation, torch.tensor([0.5, 0.5, 0.5], **self.ctx))
        else:
            ic_fields.append(fixed)

        indices = [self.current_dims.index(ic) for ic in ic_fields]
        wrap_around_ics = WrapFlow(
            InverseFlow(coordinate_transform),
            indices=indices,
            out_indices=(min(indices),) # first index of the input
        )
        self.current_dims.merge(ic_fields, out)
        self.layers.append(wrap_around_ics)




    def add_map_to_ic_domains(self, cdfs=dict(), return_layers=False):
        if len(cdfs) == 0:
            cdfs = InternalCoordinateMarginals(self.current_dims, self.ctx)
        new_layers = []
        for field in cdfs:
            if field in self.current_dims:
                if isinstance(cdfs[field], Flow):
                    icdf_flow = cdfs[field]
                else:
                    icdf_flow = InverseFlow(CDFTransform(cdfs[field]))
                flow = WrapFlow(icdf_flow, (self.current_dims.index(field),))
                self.layers.append(flow)
                new_layers.append(icdf_flow)
            else:
                warnings.warn(f"Field {field} not in current dims. CDF is ignored.")
        if return_layers:
            return new_layers

    def add_merge_constraints(
            self,
            constrained_indices,
            constrained_values,
            field=BONDS
    ):
        """Augment a tensor by constants elements.

        Parameters
        ----------
        constrained_indices : np.ndarray
            Indices of the constrained elements in the resulting tensor.
        constrained_values : np.ndarray or torch.Tensor
            Constant values to be inserted at the indices.
        field : bgflow.TensorInfo, optional
            The field into which the constants are merged.
        """
        assert field in self.current_dims
        assert len(constrained_indices) == len(constrained_values)
        if len(constrained_indices) == 0:
            warnings.warn(
                "add_merge_constraints was skipped, "
                "because no bond indices were specified.",
                UserWarning
            )
            return
        n_bonds = len(constrained_indices) + self.current_dims[field][-1]
        constrained_indices = np.array(constrained_indices)
        unconstrained_indices = np.setdiff1d(np.arange(n_bonds), constrained_indices)
        if not isinstance(constrained_values, torch.Tensor):
            constrained_values = torch.tensor(constrained_values, **self.ctx)
        field_constrained = TensorInfo(f"{field.name}_constrained", field.is_circular)
        self.add_set_constant(field_constrained, constrained_values)
        self.add_merge(
            (field, field_constrained),
            to=field,
            sizes_or_indices=(unconstrained_indices, constrained_indices)
        )

    def add_constrain_chirality(self, halpha_torsion_indices, right_handed=False, torsions=TORSIONS):
        """Constrain the chirality of aminoacids
         by constraining their normalized halpha torsions to [0.5,1] instead of [0,1].

        Parameters
        ----------
        halpha_torsion_indices : Sequence[int] or Sequence[bool]
            An index array for the torsions.

        torsions : torch.TensorInfo
        """
        loc = torch.zeros(*self.current_dims[TORSIONS], **self.ctx)
        scale = torch.ones(*self.current_dims[TORSIONS], **self.ctx)
        loc[halpha_torsion_indices] = 0.5 * (1 - right_handed)
        scale[halpha_torsion_indices] = 0.5
        affine = TorchTransform(torch.distributions.AffineTransform(loc=loc, scale=scale), 1)
        return self.add_layer(affine, what=(torsions, ))

    def add_torsion_multiplicities(self, multiplicities, torsions=TORSIONS):
        """TODO:docs"""
        fmod_layer = IncreaseMultiplicityFlow(multiplicities).to(**self.ctx)
        return self.add_layer(fmod_layer, what=(torsions, ))

    def add_torsion_shifts(self, shifts, torsions=TORSIONS):
        """TODO:docs"""
        fmod_layer = CircularShiftFlow(shifts).to(**self.ctx)
        return self.add_layer(fmod_layer, what=(torsions, ))

    def _add_to_param_groups(self, parameters, param_groups):
        parameters = list(parameters)
        for group in param_groups:
            if group not in self.param_groups:
                self.param_groups[group] = []
            self.param_groups[group].extend(parameters)
            # remove duplicate parameters if parameters are shared between layers:
            # self.param_groups[group] = list(set(self.param_groups[group]))


