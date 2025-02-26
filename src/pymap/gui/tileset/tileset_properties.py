"""Properties tree for the tileset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelContext, ModelValue
from PySide6.QtWidgets import QWidget

from pymap.gui.history import ChangeTilesetProperty
from pymap.gui.history.statement import (
    UndoRedoStatements,
)
from pymap.gui.properties.tree import ModelValueNotAvailableError, PropertiesTree

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class TilesetProperties(PropertiesTree):
    """Tree to display additional properties of the tilesets."""

    def __init__(
        self,
        tileset_widget: TilesetWidget,
        is_primary: bool,
        parent: QWidget | None = None,
    ):
        """Initializes the tileset properties.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            is_primary (bool): Whether this is the primary tileset.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(
            f'tileset_widget_{"primary" if is_primary else "secondary"}',
            tileset_widget.main_gui,
            parent=parent,
        )
        self.tileset_widget = tileset_widget
        self.is_primary = is_primary

    @property
    def datatype(self) -> str:
        """Returns the datatype.

        Returns:
            str: The datatype.
        """
        assert self.tileset_widget.main_gui.project is not None, 'Project is None'
        return self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.is_primary else 'tileset_secondary'
        ]['datatype']

    @property
    def model_value(self) -> ModelValue:
        """Returns the model value.

        Returns:
            Any: The model value.
        """
        if not self.tileset_widget.tileset_loaded:
            raise ModelValueNotAvailableError()
        if self.is_primary:
            return self.tileset_widget.main_gui.tileset_primary
        return self.tileset_widget.main_gui.tileset_secondary

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
            statements_redo (UndoRedoStatements): Redo statements.
            statements_undo (UndoRedoStatements): Undo statements.
        """
        self.tileset_widget.undo_stack.push(
            ChangeTilesetProperty(
                self.tileset_widget,
                self.is_primary,
                statements_redo,
                statements_undo,
            )
        )
