"""Parameter for a union field."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyqtgraph.parametertree.parameterTypes as parameterTypes  # type: ignore

from agb.model.type import ModelContext, ModelValue
from agb.model.union import UnionType
from pymap.gui.properties.parameters.base import ModelParameterMixin

if TYPE_CHECKING:
    from pymap.project import Project

from ..utils import type_to_parameter


class UnionTypeParameter(ModelParameterMixin, parameterTypes.GroupParameter):
    """Parameter for a union field."""

    def __init__(
        self,
        name: str,
        project: Project,
        datatype_name: str,
        value: ModelValue,
        context: ModelContext,
        parent_parameter: ModelParameterMixin | None,
        **kwargs: Any,
    ):
        """Initializes the Union Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : dict
            The values of all union subtypes.
        context : list
            The context.
        parent_parameter : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        super().__init__(
            name, project, datatype_name, value, context, parent_parameter, **kwargs
        )
        parameterTypes.GroupParameter.__init__(self, name=name, **kwargs)  # type: ignore
        assert isinstance(self.datatype, UnionType)
        assert isinstance(value, dict)
        # Create children for all possible subtypes
        self.values: dict[str, ModelParameterMixin] = {}
        for name in self.datatype.subtypes:
            subtype = self.datatype.subtypes[name]
            self.values[name] = type_to_parameter(self.project, subtype)(
                f'{subtype}',
                self.project,
                subtype,
                value[name],
                list(context) + [name],
                self,
                # title=f'View as <{subtype}>',
            )
            self.addChild(self.values[name])  # type: ignore
        # self.update_value()

    # def update_value(self):
    #     """Displays the correct union subtype."""
    #     # Get the active name
    #     assert isinstance(self.datatype, UnionType)
    #     active_name = self.datatype.name_get(
    #         self.project, self.context, self.model_parents
    #     )
    #     print(f'Parent changed to {active_name}')
    #     for name in self.values:
    #         child = self.values[name]
    #         if name == active_name and child.parent() is not self:
    #             self.addChild(child)  # type: ignore
    #         elif name != active_name and child.parent() is self:
    #             child.remove()

    @property
    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The values of the parameter.
        """
        return {name: self.values[name].model_value for name in self.values}

    def update(self, value: ModelValue):
        """Updates all children of this parameter."""
        assert isinstance(value, dict)
        assert isinstance(self.datatype, UnionType)
        for name in self.values:
            self.child(self.datatype.subtypes[name]).update(value[name])  # type: ignore

    def children_names_disabled(self) -> list[str]:
        """Returns the names of the children that are disabled."""
        assert isinstance(self.datatype, UnionType)
        active_name = self.datatype.name_get(
            self.project, self.context, self.model_parents
        )
        return [
            self.datatype.subtypes[name]
            for name in self.values
            if name != active_name and self.values[name].parent() is self
        ]
