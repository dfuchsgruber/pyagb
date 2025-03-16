"""Parameter for a bitfield type."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyqtgraph.parametertree.parameterTypes as parameterTypes  # type: ignore

from agb.model.bitfield import BitfieldType
from agb.model.type import ModelContext, ModelValue
from pymap.gui.properties.parameters.scalar import ScalarTypeParameter

if TYPE_CHECKING:
    from pymap.project import Project

from .base import ModelParameterMixin
from .constants import ConstantsTypeParameter


class BitfieldTypeParameter(ModelParameterMixin, parameterTypes.GroupParameter):
    """Parameter for a bitfield type."""

    def __init__(
        self,
        name: str,
        project: Project,
        datatype_name: str,
        value: ModelValue,
        context: ModelContext,
        parent_parameter: ModelParameterMixin | None,
        **kwargs: dict[Any, Any],
    ):
        """Initializes the Bitifield Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : dict
            The values of the bitfield.
        context : list
            The context.
        parent_parameter : parameterTypes.Parameter
            The parents of the parameter according to the data model.
        """
        super().__init__(
            name, project, datatype_name, value, context, parent_parameter, **kwargs
        )
        parameterTypes.GroupParameter.__init__(self, name=name, **kwargs)  # type: ignore
        # Add all children
        assert isinstance(self.datatype, BitfieldType)
        assert isinstance(value, dict)
        for name, constant, _ in self.datatype.structure:
            if constant is not None:
                child = ConstantsTypeParameter(
                    name,
                    list(self.project.constants[constant]),
                    self.project,
                    datatype_name,
                    value[name],
                    list(self.context) + [name],
                    self,
                    **kwargs,
                )
                child.setValue(value[name])  # type: ignore
            else:
                child = ScalarTypeParameter(
                    name,
                    self.project,
                    datatype_name,
                    value[name],
                    list(self.context) + [name],
                    self,
                    **kwargs,
                )
            if name not in self.datatype.hidden_members:
                self.addChild(child)  # type: ignore

    @property
    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        assert isinstance(self.datatype, BitfieldType)
        return {
            name: self.child(name).model_value  # type: ignore
            for name, _, _ in self.datatype.structure
        }

    def update(self, value: ModelValue):
        """Recursively updates the values of all children."""
        assert isinstance(self.datatype, BitfieldType)
        assert isinstance(value, dict)
        for name, _, _ in self.datatype.structure:
            self.child(name).update(value[name])  # type: ignore
