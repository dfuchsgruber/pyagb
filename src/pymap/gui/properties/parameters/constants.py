"""Parameter for a text field that is associated with constants."""

from __future__ import annotations

from typing import Any

import pyqtgraph.parametertree.parameterTypes as parameterTypes  # type: ignore
from agb.model.type import ModelContext, ModelValue
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QComboBox, QWidget

from pymap.gui.properties.parameters.base import ModelParameterMixin
from pymap.project import Project


class ConstantComboBox(QComboBox):
    """Subclass this thing in order to manually filter out undo events."""

    def __init__(self, parent: QWidget | None = None):
        """Initializes the combo box.

        Args:
            parent (QWidget, optional): The parent widget. Defaults to None.
        """
        super().__init__(parent)
        self.setEditable(True)
        self.installEventFilter(self)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter out undo and redo events."""
        if event.type() == QEvent.Type.ShortcutOverride:
            assert isinstance(event, QKeyEvent)
            if event.matches(QKeySequence.StandardKey.Undo) or event.matches(
                QKeySequence.StandardKey.Redo
            ):
                return True

        return super().eventFilter(watched, event)


class ConstantParameterItem(parameterTypes.WidgetParameterItem):
    """Inherit from the standard list parameter item class but with widget editable."""

    def __init__(self, param: ConstantsTypeParameter, depth: int):
        """Initializes the parameter item.

        Args:
            param (ConstantsTypeParameter): The parameter.
            depth (int): The depth.
        """
        self.constants = param.constants
        super().__init__(param, depth)

    def makeWidget(self) -> ConstantComboBox:
        """Creates the widget.

        Returns:
            ConstantComboBox: The widget.
        """
        w = ConstantComboBox()
        self.widget = w
        w.setMaximumHeight(20)  ## set to match height of spin box and line edit

        w.setEditText(w.currentText())

        w.sigChanged = w.editTextChanged  # type: ignore
        w.setValue = self.setValue  # type: ignore
        w.value = w.currentText  # type: ignore
        w.setEditable(True)
        w.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        w.addItems(self.constants)
        return w

    def setValue(self, val: str) -> None:
        """Sets the value of the widget.

        Args:
            val (Any): The value.
        """
        self.widget.setEditText(str(val))


class ConstantsTypeParameter(ModelParameterMixin, parameterTypes.ListParameter):
    """Parameter for a field that is associated with constants."""

    itemClass = ConstantParameterItem

    def __init__(
        self,
        name: str,
        constants: list[str],
        project: Project,
        datatype_name: str,
        value: ModelValue,
        context: ModelContext,
        model_parent: ModelParameterMixin | None = None,
        **kwargs: dict[Any, Any],
    ):
        """Initializes the parameter.

        Args:
            name (str): The name of the parameter.
            project (Project): The project.
            constants (list[str]): The constants.
            project (Project): The project.
            datatype_name (str): The name of the datatype.
            value (ModelValue): The value.
            context (ModelContext): The context.
            model_parent (ModelParameterMixin | None, optional): The parent.
                Defaults to None.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            name, project, datatype_name, value, context, model_parent, **kwargs
        )
        self.constants = constants
        parameterTypes.ListParameter.__init__(  # type: ignore
            self,
            name=name,
            limits=constants,
            **kwargs,
        )

    @property
    def model_value(self) -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : str
            The value of the parameter.
        """
        return self.value()

    def update(self, value: ModelValue):
        """Updates this parameter."""
        assert isinstance(value, str), f'Expected str, got {type(value)}'
        self.setValue(value)  # type: ignore
