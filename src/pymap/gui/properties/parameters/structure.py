"""Parameter for a structure field."""

from __future__ import annotations

from typing import Any

import pyqtgraph.parametertree.parameterTypes as parameterTypes  # type: ignore
from agb.model.structure import Structure  # type: ignore
from agb.model.type import ModelContext, ModelValue
from pyqtgraph.parametertree.Parameter import Parameter  # type: ignore

from pymap.gui.properties.parameters.base import ModelParameterMixin
from pymap.project import Project  # type: ignore

from ..utils import type_to_parameter


class StructureTypeParameter(ModelParameterMixin, parameterTypes.GroupParameter):
    """Parameter for a structure field."""

    def __init__(self, name: str, project: Project, datatype_name: str,
                 value: ModelValue, context: ModelContext,
                 model_parent: 'ModelParameterMixin | None',
                 **kwargs: dict[Any, Any]):
        """Initializes the Structure Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : dict
            The values of the structure.
        context : list
            The context.
        model_parents : parameterTypes.Parameter
            The parents of the parameter according to the data model.
        """
        super().__init__(name, project, datatype_name, value, context, model_parent,
                         **kwargs)
        assert isinstance(value, dict)
        assert isinstance(self.datatype, Structure)
        parameterTypes.GroupParameter.__init__(name=name, **kwargs) # type: ignore
        # Add all children
        for name, type_name, _ in sorted(self.datatype.structure, key=lambda x: x[2]):
            if name not in self.datatype.hidden_members:
                child: Parameter = type_to_parameter(project, type_name)(name, project, # type: ignore
                                                                         type_name,
                                                              value[name],
                                                              context + [name],
                                                              self)
                assert child is not None
                self.addChild(child) # type: ignore

    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        assert isinstance(self.datatype, Structure)
        return { name : self.child(name).model_value() # type: ignore
                for name, _, _ in self.datatype.structure
                if name not in self.datatype.hidden_members}

    def update(self, value: ModelValue):
        """Recursively updates the values of all children."""
        assert isinstance(value, dict)
        assert isinstance(self.datatype, Structure)
        for name, _, _ in sorted(self.datatype.structure, key=lambda x: x[2]):
            if name not in self.datatype.hidden_members:
                self.child(name).update(value[name]) # type: ignore

