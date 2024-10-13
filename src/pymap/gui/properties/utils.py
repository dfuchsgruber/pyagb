"""Setting and getting properties of the model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Type

import numpy as np
from agb.model.type import ModelParents, ModelValue

from pymap.configuration import AttributePathType

if TYPE_CHECKING:
    from pymap.gui.properties.parameters.base import ModelParameterMixin
    from pymap.project import Project


def type_to_parameter(
    project: Project, datatype_name: str
) -> Type[ModelParameterMixin]:
    """Translates a datatype into a parameter class.

    Parameters:
    -----------
    project : pymap.project.Project
        The pymap project.
    datatype_name : str
        The name of the datatype.

    Returns:
    --------
    parameter_class : parameterTypes.Parameter
        The corresponding parameter class.
    """
    from agb.model.array import (
        FixedSizeArrayType,
        VariableSizeArrayType,
    )
    from agb.model.bitfield import BitfieldType
    from agb.model.pointer import DynamicLabelPointer, PointerType
    from agb.model.scalar_type import ScalarType
    from agb.model.structure import Structure
    from agb.model.unbounded_array import UnboundedArrayType
    from agb.model.union import UnionType

    from pymap.gui.properties.parameters.array import (
        FixedSizeArrayTypeParameter,
        UnboundedArrayTypeParameter,
        VariableSizeArrayTypeParameter,
    )
    from pymap.gui.properties.parameters.bitfield import BitfieldTypeParameter
    from pymap.gui.properties.parameters.pointer import PointerParameter
    from pymap.gui.properties.parameters.scalar import ScalarTypeParameter
    from pymap.gui.properties.parameters.structure import StructureTypeParameter
    from pymap.gui.properties.parameters.union import UnionTypeParameter

    datatype = project.model[datatype_name]
    if isinstance(datatype, DynamicLabelPointer):
        raise NotImplementedError('Dynamic label pointers not yet supported.')
    elif isinstance(datatype, PointerType):
        return PointerParameter
    elif isinstance(datatype, BitfieldType):
        return BitfieldTypeParameter
    elif isinstance(datatype, ScalarType):
        return ScalarTypeParameter
    elif isinstance(datatype, Structure):
        return StructureTypeParameter
    elif isinstance(datatype, FixedSizeArrayType):
        return FixedSizeArrayTypeParameter
    elif isinstance(datatype, VariableSizeArrayType):
        return VariableSizeArrayTypeParameter
    elif isinstance(datatype, UnboundedArrayType):
        return UnboundedArrayTypeParameter
    elif isinstance(datatype, UnionType):
        return UnionTypeParameter
    else:
        raise RuntimeError(f'Unsupported datatype class {type(datatype)} of {datatype}')


def get_member_by_path(value: ModelValue, path: AttributePathType) -> ModelValue:
    """Returns an attribute of a structure by its path."""
    for edge in path:
        if isinstance(value, np.ndarray):
            value = value[edge]  # type: ignore
        if isinstance(value, list) and isinstance(edge, int):
            value = value[edge]
        elif isinstance(value, dict) and isinstance(edge, str):
            value = value[edge]
        else:
            raise RuntimeError(f'Unsupported edge type {type(edge)}')
    return value


def set_member_by_path(
    _target: ModelValue,
    value: ModelValue,  # type: ignore
    path: AttributePathType,
):
    """Sets the value of a structure by its path.

    Parameters:
    -----------
    target : dict
        The structure that holds the requested value
    value : str
        The value to apply
    path : list
        A path to access the attribute
    """
    target = _target
    for edge in path[:-1]:
        match _target:
            case list() if isinstance(edge, int):
                target: ModelValue = _target[edge]  # type: ignore
            case dict() if isinstance(edge, (str)):
                target: ModelValue = _target[edge]  # type: ignore
            case _:  # type: ignore
                raise RuntimeError(f'Unsupported edge type {type(edge)}')
    assert isinstance(_target, (dict, list))
    target[path[-1]] = value  # type: ignore


def get_parents_by_path(value: ModelValue, path: AttributePathType) -> ModelParents:
    """Builds the parents of an instance based on its path.

    Note that the requested data instance is not needed to be present
    for this method to work. Just all its parent have to be.

    Parameters:
    -----------
    value : dict
        The origin structure that contains a data instance.
    path : list
        A path to access the data instance.

    Returns:
    --------
    parents : list
        The model parents of this data instance.
    """
    parents = [value]
    for member in path[:-1]:
        if isinstance(value, list) and isinstance(member, int):
            value = value[member]
        elif isinstance(value, dict) and isinstance(member, str):
            value = value[member]
        else:
            raise RuntimeError(f'Unsupported edge type {type(member)}')
        parents.append(value)
    return parents
