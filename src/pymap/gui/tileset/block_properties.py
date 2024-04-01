"""Properties tree for blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelContext, ModelValue
from PySide6.QtWidgets import (
    QWidget,
)

from pymap.gui.history import ChangeBlockProperty
from pymap.gui.history.statement import (
    UndoRedoStatements,
)
from pymap.gui.properties.tree import ModelValueNotAvailableError, PropertiesTree

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class BlockProperties(PropertiesTree):
    """Tree to display block properties."""

    def __init__(self, tileset_widget: TilesetWidget, parent: QWidget | None = None):
        """Initializes the block properties.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(
            'block_properties',
            tileset_widget.main_gui,
            parent=parent,
        )
        self.tileset_widget = tileset_widget

    @property
    def datatype(self) -> str:
        """Returns the datatype.

        Returns:
            str: The datatype.
        """
        assert self.tileset_widget.main_gui.project is not None, 'Project is None'
        return self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary'
            if self.tileset_widget.selected_block_idx < 0x280
            else 'tileset_secondary'
        ]['behaviour_datatype']

    @property
    def model_value(self) -> ModelValue:
        """Returns the model value.

        Returns:
            Any: The model value.
        """
        if not self.tileset_widget.tileset_loaded:
            raise ModelValueNotAvailableError()
        return self.tileset_widget.main_gui.get_tileset_behaviour(
            self.tileset_widget.selected_block_idx
        )

    @property
    def model_context(self) -> ModelContext:
        """Returns the model context.

        Returns:
            ModelContext: The model context.
        """
        assert self.tileset_widget.main_gui.project is not None, 'Project is None'
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary'
            if self.tileset_widget.selected_block_idx < 0x280
            else 'tileset_secondary'
        ]
        return config['behaviours_path'] + [
            self.tileset_widget.selected_block_idx % 0x280
        ]

    def value_changed(
        self, statements_redo: UndoRedoStatements, statements_undo: UndoRedoStatements
    ) -> None:
        """Handles changes in the value.

        Args:
            statements_redo (UndoRedoStatements): The redo statements.
            statements_undo (UndoRedoStatements): The undo statements.
        """
        self.tileset_widget.undo_stack.push(
            ChangeBlockProperty(
                self.tileset_widget,
                self.tileset_widget.selected_block_idx,
                statements_redo,
                statements_undo,
            )
        )
