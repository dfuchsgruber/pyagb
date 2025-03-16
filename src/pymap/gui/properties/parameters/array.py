"""Parameter for a array types."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pyqtgraph.parametertree.parameterTypes as parameterTypes  # type: ignore
from pyqtgraph.parametertree.Parameter import Parameter  # type: ignore

from agb.model.array import ArrayType, FixedSizeArrayType, VariableSizeArrayType
from agb.model.type import ModelContext, ModelValue
from agb.model.unbounded_array import UnboundedArrayType

if TYPE_CHECKING:
    from pymap.project import Project

from ..utils import type_to_parameter
from .base import ModelParameterMixin


class ArrayTypeParameter(ModelParameterMixin, parameterTypes.GroupParameter):
    """A parameter for an array type."""

    def __init__(
        self,
        name: str,
        project: Project,
        datatype_name: str,
        values: ModelValue,
        context: ModelContext,
        parent_parameter: ModelParameterMixin | None,
        **kwargs: dict[Any, Any],
    ):
        """Initializes the Structure Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        parameterTypes.GroupParameter.__init__(self, name=name, **kwargs)  # type: ignore
        super().__init__(
            name, project, datatype_name, values, context, parent_parameter, **kwargs
        )
        # Add the list that enables navigating through the array
        size: int = self.size_get()
        self.addChild(  # type: ignore
            parameterTypes.ListParameter(
                name='idx',
                title='Index',  # type: ignore
                values=list(range(size)),
                value=None,
                default=None,
            )
        )
        # Create one parameter for each value
        self.values: list[ModelParameterMixin] = []
        assert isinstance(values, list)
        for idx, value in enumerate(values):
            self._insert(idx, value=value)
        self.update_value()

    def size_get(self) -> int:
        """Gets the size of the array.

        Returns:
            int: The size of the array.
        """
        assert isinstance(self.datatype, ArrayType)
        return self.datatype.size_get(self.project, self.context, self.model_parents)

    def _insert(self, idx: int | None | str, value: ModelValue = None):
        """Inserts a new element into the array.

        Parameters:
        -----------
        idx : int or str or None
            The index to insert the element at.
        value : object or None
            The value to insert. If None a new value will be created.
        """
        assert isinstance(self.datatype, ArrayType)
        if idx is not None and idx != '' and int(idx) in range(self.size_get() + 1):
            value_datatype_name = self.datatype.datatype
            value_parameter_class = type_to_parameter(self.project, value_datatype_name)
            parents = self.model_parents
            assert isinstance(parents, list), f'Expected list, got {type(parents)}'
            if value is None:
                value = self.project.model[value_datatype_name](
                    self.project,
                    list(self.context) + [int(idx)],
                    parents + [self.model_value],
                )
            self.values.insert(
                int(idx),  # type: ignore
                value_parameter_class(
                    str(idx),
                    self.project,
                    value_datatype_name,
                    value,
                    list(self.context) + [idx],
                    self,
                    title=f'<{value_datatype_name}>',
                ),
            )  # type: ignore

    def treeStateChanged(self, param: Parameter, changes: Any):
        """Called when the state of the tree changes.

        Args:
            param (Parameter): The parameter that changed.
            changes (Any): The changes that occured.
        """
        super().treeStateChanged(param, changes)  # type: ignore
        if param is self.child('idx'):  # type: ignore
            # Load the element at this index
            self.update_value()

    def update_value(self):
        """Updates the value of the array."""
        # Remove all subtree children
        for subtree in self.values:
            if subtree.parent() is not None:
                assert subtree.parent() is self
                subtree.remove()
        # Only show the child that matches the current index
        idx: int | str | None = self.child('idx').value()  # type: ignore
        if idx is not None and idx != '' and int(idx) in range(self.size_get()):  # type: ignore
            self.addChild(self.values[int(idx)])  # type: ignore

    @property
    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : list
            The value of the parameter.
        """
        return list(map(lambda child: child.model_value, self.values))  # type: ignore

    def update(self, value: ModelValue):
        """Updates the values in the array.

        Parameters:
        -----------
        values : list
            The new values.
        """
        assert isinstance(value, list)
        for child, v in zip(self.values, value):
            child.update(v)  # type: ignore


class FixedSizeArrayTypeParameter(ArrayTypeParameter):
    """Parameter for a fixed size array type."""

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
        """Initializes the Array Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        assert isinstance(self.datatype, FixedSizeArrayType)
        super().__init__(
            name, project, datatype_name, value, context, parent_parameter, **kwargs
        )

    def size_get(self) -> int:
        """Gets the size of the array.

        Returns:
            int: The size of the array.
        """
        assert isinstance(self.datatype, FixedSizeArrayType)
        return self.datatype.size_get(self.project, self.context, self.model_parents)


class VariableSizeArrayTypeParameter(ArrayTypeParameter):
    """Parameter for a variable size array type."""

    allowed_types = (VariableSizeArrayType, UnboundedArrayType)

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
        """Initializes the Array Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        super().__init__(
            name, project, datatype_name, value, context, parent_parameter, **kwargs
        )
        assert isinstance(self.datatype, self.allowed_types)
        # Add widgets to add and remove elements
        remove_parameter = parameterTypes.ActionParameter(name='Remove')
        self.addChild(remove_parameter)  # type: ignore
        remove_parameter.sigActivated.connect(  # type: ignore
            lambda: self._remove(idx=self.child('idx').value())  # type: ignore
        )  # type: ignore
        append_parameter = parameterTypes.ActionParameter(name='Append')
        self.addChild(append_parameter)  # type: ignore
        append_parameter.sigActivated.connect(self._append)  # type: ignore
        # Make length in parent read-only
        parameter = self._size_location()
        if parameter is not None:
            parameter.setReadonly(True)
        self.update_value()

    def _adaptLimits(self):
        # Assert size still matches
        if self.size_get() != len(self.values):
            raise RuntimeError(
                f'Size mismatch. Parent uses {self.size_get()} '
                f'but array only holds {len(self.values)}'
            )
        self.child('idx').setLimits(list(range(len(self.values))))  # type: ignore

    def _append(self):
        """Appends a default element to the array."""
        size = self.size_get()
        self._insert(size)
        # Increment size
        self.size_set(size + 1)
        # Change limits
        self._adaptLimits()

    def _remove(self, idx: int | None | str):
        """Removes an element from the array.

        Parameters:
        -----------
        idx : int or str or None
            The index to remove.
        """
        if idx is not None and idx != '' and int(idx) in range(self.size_get()):
            idx = int(idx)
            self.values[idx].remove()
            del self.values[idx]
            self.size_set(len(self.values))
            self._adaptLimits()
            self.child('idx').setValue(len(self.values) - 1)  # type: ignore
            # self.update_value()

    def _size_location(self) -> ModelParameterMixin | None:
        """Gets the parameter that controls the size of the array.

        Returns:
            Parameter: The parameter that controls the size of the array.
        """
        assert isinstance(self.datatype, VariableSizeArrayType)
        n_parents, location = self.datatype.size_path
        if n_parents <= 0:
            raise RuntimeError(
                f'Upwards parent traversals must be positive, not {n_parents}'
            )
        root = self
        for _ in range(n_parents):
            root = root.model_parent  # type: ignore
        for member in location:
            root = root.child(member)  # type: ignore
        return root  # type: ignore

    def size_get(self) -> int:
        """Gets the size of the array.

        Returns:
            int: The size of the array.
        """
        assert isinstance(self.datatype, VariableSizeArrayType)
        parameter = self._size_location()
        if parameter is not None:
            return self.datatype.size_cast(parameter.value(), self.project)  # type: ignore
        return 0

    def size_set(self, size: int | str):
        """Sets the size of the array.

        Args:
            size (int): The new size of the array.
        """
        self._size_location().setValue(size)  # type: ignore


class UnboundedArrayTypeParameter(VariableSizeArrayTypeParameter):
    """Parameter for an unbounded array type."""

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
        """Initializes the Array Parameter class.

        Parameters:
        -----------
        name : str
            The name of the parameter
        project : pymap.project.Project
            The underlying pymap project.
        datatype_name : str
            The name of the datatype associated with the parameter.
        values : list
            The values of the array.
        context : list
            The context.
        model_parent : parameterTypes.Parameter
            The parent of the parameter according to the data model.
        """
        assert isinstance(value, list)
        # First, initialize with a stub to make the size available
        self.values: list[Parameter | None] = [None] * len(value)  # type: ignore
        super().__init__(
            name, project, datatype_name, value, context, parent_parameter, **kwargs
        )

    def size_get(self) -> int:
        """Gets the size of the array.

        Returns:
            int: The size of the array.
        """
        return len(self.values)

    def size_set(self, size: int | str | None):
        """Sets the size of the array. No effect.

        Args:
            size (int | str | None): The new size of the array.
        """
        pass

    def _size_location(self) -> ModelParameterMixin | None:
        pass
