"""Properties tree for the tileset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelValue
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import QHeaderView, QWidget
from typing_extensions import ParamSpec

from pymap.gui.history import ChangeTilesetProperty
from pymap.gui.history.statement import (
    model_value_difference_to_undo_redo_statements,
)
from pymap.gui.properties import type_to_parameter

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class TilesetProperties(ParameterTree):
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
        ParameterTree.__init__(self, parent=parent)  # type: ignore
        self.tileset_widget = tileset_widget
        self.is_primary = is_primary
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.root = None

    def _load_tileset(self):
        """Implementation of loading the tileset into the tree."""
        if not self.tileset_widget.tileset_loaded:
            return
        assert self.tileset_widget.main_gui.project is not None
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.is_primary else 'tileset_secondary'
        ]
        datatype = config['datatype']
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        self.root = type_to_parameter(self.tileset_widget.main_gui.project, datatype)(
            '.',
            self.tileset_widget.main_gui.project,
            datatype,
            tileset,
            [],
            None,
        )
        self.addParameters(self.root, showTop=False)  # type: ignore
        self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    def load_tileset(self):
        """Loads the current tileset into the tree."""
        self.clear()
        self.root = None
        self._load_tileset()

    def update(self):
        """Updates all values in the tree according to the current properties."""
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        assert self.root is not None
        self.root.blockSignals(True)  # type: ignore
        self.root.update(tileset)  # type: ignore
        self.root.blockSignals(False)  # type: ignore

    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Signal handler for when the tree changes.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        if not self.tileset_widget.tileset_loaded:
            return
        root = (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        assert self.root is not None
        self.tileset_widget.undo_stack.push(
            ChangeTilesetProperty(
                self.tileset_widget,
                self.is_primary,
                *model_value_difference_to_undo_redo_statements(root, self.root),
            )
        )

    def get_value(self) -> ModelValue:
        """Gets the model value of the current block or None if no block is selected."""
        if self.root is None:
            return None
        return self.root.model_value

    def set_value(self, behaviour: ModelValue):
        """Replaces the entry properties of the current block if one is selected."""
        if not self.tileset_widget.tileset_loaded:
            return
        assert self.root is not None
        self.root.blockSignals(True)  # type: ignore
        self.root.update(behaviour)
        self.root.blockSignals(False)  # type: ignore
        self.tree_changed(None)
