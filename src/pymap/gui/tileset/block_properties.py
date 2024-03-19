"""Properties tree for blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelValue
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import (
    QHeaderView,
    QWidget,
)
from typing_extensions import ParamSpec

from pymap.gui import properties
from pymap.gui.history import ChangeBlockProperty
from pymap.gui.history.statement import model_value_difference_to_undo_redo_statements

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class BlockProperties(ParameterTree):
    """Tree to display block properties."""

    def __init__(self, tileset_widget: TilesetWidget, parent: QWidget | None = None):
        """Initializes the block properties.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        self.tileset_widget = tileset_widget
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.root = None

    def load_block(self):
        """Loads the currently displayed blocks properties."""
        self.clear()
        if not self.tileset_widget.tileset_loaded:
            self.root = None
        else:
            assert (
                self.tileset_widget.main_gui.project is not None
            ), 'Tileset widget has no project'
            config = self.tileset_widget.main_gui.project.config['pymap'][
                'tileset_primary'
                if self.tileset_widget.selected_block_idx < 0x280
                else 'tileset_secondary'
            ]
            datatype = config['behaviour_datatype']
            behaviour = self.tileset_widget.main_gui.get_tileset_behaviour(
                self.tileset_widget.selected_block_idx
            )

            self.root = properties.type_to_parameter(
                self.tileset_widget.main_gui.project, datatype
            )(
                '.',
                self.tileset_widget.main_gui.project,
                datatype,
                behaviour,
                config['behaviours_path']
                + [self.tileset_widget.selected_block_idx % 0x280],
                None,
            )
            self.addParameters(self.root, showTop=False)  # type: ignore
            self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    def update(self):
        """Updates all values in the tree according to the current properties."""
        if not self.tileset_widget.tileset_loaded:
            return
        assert self.tileset_widget.main_gui.project is not None
        behaviour = self.tileset_widget.main_gui.get_tileset_behaviour(
            self.tileset_widget.selected_block_idx
        )
        self.root.blockSignals(True)  # type: ignore
        self.root.update(behaviour)  # type: ignore
        self.root.blockSignals(False)  # type: ignore

    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Signal handler for when the tree changes.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        if not self.tileset_widget.tileset_loaded:
            return
        root = self.tileset_widget.main_gui.get_tileset_behaviour(
            self.tileset_widget.selected_block_idx
        )
        assert self.root is not None
        self.tileset_widget.undo_stack.push(
            ChangeBlockProperty(
                self.tileset_widget,
                self.tileset_widget.selected_block_idx,
                *model_value_difference_to_undo_redo_statements(self.root, root),
            )
        )

    def get_value(self) -> ModelValue:
        """Gets the model value of the current block or None if no block is selected."""
        if self.root is None:
            return None
        return self.root.model_value

    def set_value(self, behaviour: ModelValue):
        """Replaces the entrie properties of the current block if one is selected."""
        if self.model is None:  # type: ignore
            return
        self.root.blockSignals(True)  # type: ignore
        self.root.update(behaviour)  # type: ignore
        self.root.blockSignals(False)  # type: ignore
        self.tree_changed(None)
