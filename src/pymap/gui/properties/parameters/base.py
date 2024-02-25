"""Model Parameter Mixin."""

from __future__ import annotations

from typing import Any

from agb.model.type import ModelContext, ModelParents, ModelValue
from pyqtgraph.parametertree.Parameter import Parameter  # type: ignore

from pymap.project import Project


class ModelParameterMixin(Parameter):
    """Model Parameter Mixin."""

    def __init__(
        self,
        name: str,
        project: Project,
        datatype_name: str,
        value: ModelValue,
        context: ModelContext,
        model_parent: ModelParameterMixin | None,
        **kwargs: Any,
    ):
        """Initializes the Bitifield Parameter class.

        Parameters:
        -----------
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        context : list
            The context.
        model_parents : parameterTypes.Parameter
            The parents of the parameter according to the data model.
        """
        self.datatype = project.model[datatype_name]
        self.project = project
        self.context = context
        self.parent_parameter = model_parent

    @property
    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        raise NotImplementedError

    @property
    def model_parents(self) -> ModelParents:
        """Gets the parents of this parameter according to the data model.

        Returns:
            ModelParents: The parents of the parameter.
        """
        model_parents: ModelParents = []
        root = self.parent_parameter
        while root is not None:
            model_parents.append(root.model_value)
            root = root.parent_parameter
        return model_parents
