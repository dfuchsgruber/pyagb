"""Parameter for a text field that is associated with constants."""

from __future__ import annotations

from typing import Any

import pyqtgraph.parametertree.parameterTypes as parameterTypes  # type: ignore
from agb.model.type import ModelValue
from PySide6.QtCore import QEvent, QObject, Qt
from PySide6.QtGui import QKeyEvent, QKeySequence
from PySide6.QtWidgets import QComboBox, QWidget


class ConstantComboBox(QComboBox):
    """Subclass this thing in order to manually filter out undo events."""

    def __init__(self, parameter: ConstantParameterItem | None = None,
                 parent: QWidget | None = None) -> None:
        """Initializes the combo box.

        Args:
            parameter (ConstantParameterItem | None, optional): The parameter it refers
                to. Defaults to None.
            parent (QWidget | None, optional): _description_. Defaults to None.
        """
        super().__init__(parent)
        self.parameter = parameter

    def setEditText(self, text: str) -> None:
        """Set the text of the combo box.

        Args:
            text (str): The text to set.
        """
        if self.parameter is not None:
            self.parameter.setValue(text)
        else:
            return super().setEditText(text)

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        """Filter out undo and redo events.

        Args:
            watched (QObject): The object being watched.
            event (QEvent): The event.

        Returns:
            bool: Whether the event was handled.
        """
        if event.type() == QEvent.Type.KeyPress and isinstance(event, QKeyEvent):
            if event.matches(QKeySequence.StandardKey.Undo) or \
                event.matches(QKeySequence.StandardKey.Redo):
                return False
        return super().eventFilter(watched, event)



# Parameter item for parameters associated with constants
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
        w = ConstantComboBox(self)
        w.setMaximumHeight(20)  ## set to match height of spin box and line edit
        # TODO
        w.setEditText(w.currentText())


        # w.sigChanged = w.editTextChanged
        # w.setValue = self.setValue
        # w.value = w.currentText
        w.setEditable(True)
        w.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.widget = w
        w.addItems(self.constants)
        return w

    def setValue(self, val: str) -> None:
        """Sets the value of the widget.

        Args:
            val (Any): The value.
        """
        self.widget.setEditText(str(val))



class ConstantsTypeParameter(parameterTypes.ListParameter):
    """Parameter for a field that is associated with constants."""

    itemClass = ConstantParameterItem

    def __init__(self, name: str, constants: list[str],
                 **kwargs: dict[Any, Any]):
        """Initializes the parameter.

        Args:
            name (str): The name of the parameter.
            project (Project): The project.
            constants (list[str]): The constants.
            kwargs (dict): Additional arguments.
        """
        super().__init__(name, **kwargs) # type: ignore
        self.constants = constants
        parameterTypes.ListParameter.__init__(self, # type: ignore
                                              name=name, limits=constants, **kwargs) # type: ignore


    def model_value(self)  -> ModelValue:
        """Gets the value of this parameter according to the data model.

        Returns:
        --------
        value : str
            The value of the parameter.
        """
        return self.value()

    def update(self, value: str):
        """Updates this parameter."""
        self.setValue(value) # type: ignore
