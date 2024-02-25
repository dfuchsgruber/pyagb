"""Parameter for a pointer."""

from __future__ import annotations

from typing import Any

from agb.model.pointer import PointerType
from agb.model.type import ModelContext, ModelValue
from pyqtgraph.parametertree import parameterTypes  # type: ignore

from pymap.gui.properties.parameters.base import ModelParameterMixin
from pymap.project import Project

from ..utils import type_to_parameter


class PointerParameter(ModelParameterMixin, parameterTypes.GroupParameter):
    """Parameter for a pointer."""

    add_reference = 'Add reference'  # Name and text of the 'Add reference'
    remove_reference = 'Remove reference'  # Name and text of the 'Remove reference'
    referred = 'referred'  # Name of the subtree that holds the referred values

    def __init__(
        self,
        name: str,
        project: Project,
        datatype_name: str,
        value: ModelValue,
        context: ModelContext,
        parameter_parent: ModelParameterMixin | None = None,
        **kwargs: dict[Any, Any],
    ):
        """Initializes the Pointer Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        value : object or None
            The value of the refered data or None if the pointer points to null.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        super().__init__(
            name, project, datatype_name, value, context, parameter_parent, **kwargs
        )
        parameterTypes.GroupParameter.__init__(self, name=name, **kwargs)  # type: ignore
        if value is not None:
            self.add_new(referred=value)
        else:
            self._add_add()

    def add_new(self, referred: ModelValue):
        """Adds a new instance of a refered value if currently null is referred.

        That is, if i.e. no child is held by this group.

        Parameters:
        -----------
        value : object or None
            The value to add. If None the datatype is default initialized.
        """
        assert isinstance(self.datatype, PointerType)
        referred_datatype_name = self.datatype.datatype
        referred_parameter_class = type_to_parameter(
            self.project, referred_datatype_name
        )
        if referred is None:
            referred = self.project.model[referred_datatype_name](
                self.project, self.context, self.model_parents
            )
        child = referred_parameter_class(
            PointerParameter.referred,
            self.project,
            referred_datatype_name,
            referred,
            self.context,
            self.parent_parameter,
            removable=True,
            title=f'Reference to <{referred_datatype_name}>',
        )

        self.addChild(child)  # type: ignore
        # Remove the add button if present
        try:
            self.child(PointerParameter.add_reference).remove()  # type: ignore
        except Exception:
            pass

    def _add_add(self):
        """Adds the add-button to this parameter."""
        child_add = parameterTypes.ActionParameter(name=PointerParameter.add_reference)
        child_add.sigActivated.connect(lambda: self.add_new(referred=None))  # type: ignore
        self.addChild(child_add)  # type: ignore

    def treeStateChanged(self, param: Any, changes: Any):
        """Called when the state of the tree changes.

        Args:
            param (Any): The parameter that changed.
            changes (Any): The changes that occured.
        """
        super().treeStateChanged(param, changes)  # type: ignore
        if not self.hasChild():
            # We do not have a child anymore, add the add-button
            self._add_add()

    def remove(self):
        """Removes the instance of the referred value."""
        if self.hasChild():
            self.child(PointerParameter.referred).remove()  # type: ignore
        # Remove the delete button if present
        try:
            self.child(PointerParameter.remove_reference).remove()  # type: ignore
        except Exception:
            pass

    def hasChild(self):
        """Checks if this parameter currently refers to a child."""
        try:
            self.child(PointerParameter.referred)  # type: ignore
            return True
        except Exception:
            return False

    def update(self, value: ModelValue):
        """Updates the pointer reference."""
        if value is None:
            self.remove()
        else:
            if self.hasChild():
                self.child(PointerParameter.referred).update(value)  # type: ignore
            else:
                self.add_new(referred=value)

    @property
    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : dict
            The value of the parameter.
        """
        try:
            return self.child(PointerParameter.referred).model_value  # type: ignore
        except Exception:
            return None
