"""History actions for map footers."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand

from agb.model.type import ModelValue
from pymap.gui.history.statement import UndoRedoStatements, ChangeProperty

if TYPE_CHECKING:
    from pymap.gui.footer import FooterWidget
    from pymap.gui.main.gui import PymapGui


class AssignTileset(QUndoCommand):
    """Class for assigning a new tileset."""

    def __init__(
        self, main_gui: PymapGui, primary: bool, label_new: str, label_old: str
    ):
        """Initializes the tileset assignment.

        Args:
            main_gui (PymapGui): reference to the main gui
            primary (bool): Whether the primary or secondary tileset is being assigned
            label_new (str): Label of the new tileset
            label_old (str): Label of the old tileset
        """
        super().__init__()
        self.main_gui = main_gui
        self.label_new = label_new
        self.label_old = label_old
        self.primary = primary

    def _assign(self, label: str):
        """Helper for assigning a label."""
        if self.main_gui.project is None:
            return
        if self.primary:
            self.main_gui.open_tilesets(label_primary=label)
        else:
            self.main_gui.open_tilesets(label_secondary=label)

    def redo(self):
        """Performs the tileset assignment."""
        self._assign(self.label_new)

    def undo(self):
        """Undoes the tileset assignment."""
        self._assign(self.label_old)


class ChangeFooterProperty(ChangeProperty):
    """Change a property of the footer."""

    def __init__(
        self,
        footer_widget: FooterWidget,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the property change.

        Args:
            footer_widget (FooterWidget): The footer widget
            statements_redo (list[str]): Statements to be executed for redo
            statements_undo (list[str]): Statements to be executed for undo
        """
        super().__init__(statements_redo, statements_undo)
        self.footer_widget = footer_widget

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        return self.footer_widget.main_gui.footer

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        self.footer_widget.update()

    def undo(self):
        """Executes the redo statements."""
        super().undo()
        self.footer_widget.update()
