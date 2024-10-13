"""Tree widget for editing the map footer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agb.model.type import ModelContext
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QWidget
from typing_extensions import ParamSpec

from pymap.gui.history.statement import (
    UndoRedoStatements,
)
from pymap.gui.properties.tree import ModelValueNotAvailableError, PropertiesTree

from .history import ChangeFooterProperty

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui


class FooterWidget(PropertiesTree):
    """Class for properties of the map footer."""

    def __init__(self, main_gui: PymapGui, parent: QWidget | None = None):
        """Initializes the footer widget.

        Args:
            main_gui (PymapGui): The main GUI.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(
            'footer_widget',
            main_gui,
            parent=parent,
        )
        self.undo_stack = QUndoStack()

    @property
    def datatype(self) -> str:
        """Returns the datatype.

        Returns:
            str: The datatype.
        """
        assert self.main_gui.project is not None, 'Project is None'
        return self.main_gui.project.config['pymap']['footer']['datatype']

    @property
    def model_value(self) -> Any:
        """Returns the model value.

        Returns:
            Any: The model value.
        """
        if self.main_gui.footer is None:
            raise ModelValueNotAvailableError
        return self.main_gui.footer

    @property
    def model_context(self) -> ModelContext:
        """Returns the model context.

        Returns:
            ModelContext: The model context.
        """
        return []

    def value_changed(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ) -> None:
        """Handles changes in the value.

        Args:
            statements_redo (UndoRedoStatements): The redo statements.
            statements_undo (UndoRedoStatements): The undo statements.
        """
        self.undo_stack.push(
            ChangeFooterProperty(
                self,
                statements_redo,
                statements_undo,
            )
        )
