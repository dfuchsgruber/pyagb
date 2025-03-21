"""History actions for map headers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from agb.model.type import ModelValue
from pymap.gui.history.statement import ChangeProperty, UndoRedoStatements

if TYPE_CHECKING:
    from pymap.gui.header import HeaderWidget
    from pymap.gui.main.gui import PymapGui


class AssignFooter(QUndoCommand):
    """Class for assigning a footer."""

    def __init__(self, main_gui: PymapGui, label_new: str, label_old: str):
        """Initializes the footer assignment.

        Args:
            main_gui (PymapGui): reference to the main gui
            label_new (str): Label of the new footer
            label_old (str): Label of the old footer
        """
        super().__init__(
            'Assign Footer',
        )
        self.main_gui = main_gui
        self.label_new = label_new
        self.label_old = label_old

    def _assign(self, label: str):
        """Helper for assigning a label."""
        if self.main_gui.project is None or self.main_gui.header is None:
            return
        self.main_gui.open_footer(label)

    def redo(self):
        """Performs the tileset assignment."""
        self._assign(self.label_new)

    def undo(self):
        """Undoes the tileset assignment."""
        self._assign(self.label_old)


class ChangeHeaderProperty(ChangeProperty):
    """Change a property of the header."""

    def __init__(
        self,
        header_widget: HeaderWidget,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the header property change.

        Args:
            header_widget (HeaderWidget): reference to the header widget
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__(
            statements_redo, statements_undo, text='Change Header Property'
        )
        self.header_widget = header_widget

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        return self.header_widget.main_gui.header

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        self.header_widget.update()

    def undo(self):
        """Executes the redo statements."""
        super().undo()
        self.header_widget.update()
