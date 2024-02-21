"""Parameter for a text field that is a scalar."""

from __future__ import annotations

from typing import Any

from agb.model.type import ModelContext, ModelValue

from pymap.gui.properties.parameters.base import ModelParameterMixin
from pymap.project import Project

from .constants import ConstantsTypeParameter


class ScalarTypeParameter(ModelParameterMixin, ConstantsTypeParameter):
    """Parameter for a text field that is a scalar.

    Args:
        ConstantsTypeParameter (_type_): The type of the parameter.
    """

    # Parameter for the tree that builds upon a scalar type
    def __init__(self, name: str, project: Project, datatype_name: str,
                 value: ModelValue, context: ModelContext,
                 model_parent: 'ModelParameterMixin | None',
                 **kwargs: dict[Any, Any]):
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
        super().__init__(name, project, datatype_name, value, context,
                         model_parent, **kwargs)
        # Make constants appear in the combo box
        constant: str | None = getattr(self.datatype, 'constant', None)
        if constant is not None:
            constants = [value for value in self.project.constants[constant]]
        else:
            constants = []
        ConstantsTypeParameter.__init__(self, name, constants, **kwargs)
        self.setValue(value) # type: ignore
