"""Parameter for a text field that is a scalar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agb.model.type import ModelContext, ModelValue, ScalarModelValue

from pymap.gui.properties.parameters.base import ModelParameterMixin

if TYPE_CHECKING:
    from pymap.project import Project

from .constants import ConstantsTypeParameter


class ScalarTypeParameter(ConstantsTypeParameter):
    """Parameter for a text field that is a scalar.

    Args:
        ConstantsTypeParameter (_type_): The type of the parameter.
    """

    def __init__(
        self,
        name: str,
        project: Project,
        datatype_name: str,
        value: ModelValue,
        context: ModelContext,
        parent_parameter: ModelParameterMixin | None = None,
        **kwargs: dict[Any, Any],
    ):
        """Initializes the ScalarType Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : int or str
            The value of the scalar type.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        constant: str | None = getattr(project.model[datatype_name], 'constant', None)
        if constant is not None:
            constants = [value for value in project.constants[constant]]
        else:
            constants = []
        super().__init__(
            name,
            constants,
            project,
            datatype_name,
            value,
            context,
            parent_parameter,
            **kwargs,
        )
        assert isinstance(value, ScalarModelValue)
        self.setValue(value)  # type: ignore

    def update(self, value: ModelValue):
        """Updates the value of the parameter.

        Args:
            value (ModelValue): The new value.
        """
        assert isinstance(
            value, ScalarModelValue
        ), f'Expected ScalarModelValue, got {value}'
        self.setValue(value)  # type: ignore
