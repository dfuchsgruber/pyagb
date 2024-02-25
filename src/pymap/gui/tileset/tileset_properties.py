"""Properties tree for the tileset."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelValue
from deepdiff import DeepDiff  # type: ignore
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import QHeaderView, QWidget
from typing_extensions import ParamSpec

from pymap.gui.history import ChangeTilesetProperty
from pymap.gui.history.statement import UndoRedoStatements
from pymap.gui.properties import type_to_parameter

from .child import TilesetChildWidgetMixin, if_tileset_loaded

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class TilesetProperties(ParameterTree, TilesetChildWidgetMixin):
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
        TilesetChildWidgetMixin.__init__(self, tileset_widget)
        self.tileset_widget = tileset_widget
        self.is_primary = is_primary
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.root = None

    @if_tileset_loaded
    def _load_tileset(self):
        """Implementation of loading the tileset into the tree."""
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
            [],
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

    @if_tileset_loaded
    def tree_changed(self, changes: list[tuple[object, object, object]] | None):
        """Signal handler for when the tree changes.

        Args:
            changes (list[tuple[object, object, object]] | None): The changes.
        """
        root = (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        assert self.root is not None
        diffs = DeepDiff(root, self.root.model_value)
        statements_redo: UndoRedoStatements = []
        statements_undo: UndoRedoStatements = []
        for change in ('type_changes', 'values_changed'):
            if change in diffs:
                for path in diffs[change]:  # type: ignore
                    value_new = diffs[change][path]['new_value']  # type: ignore
                    value_old = diffs[change][path]['old_value']  # type: ignore
                    statements_redo.append(f"{path} = '{value_new}'")
                    statements_undo.append(f"{path} = '{value_old}'")
                    self.tileset_widget.undo_stack.push(
                        ChangeTilesetProperty(
                            self.tileset_widget,
                            self.is_primary,
                            statements_redo,
                            statements_undo,
                        )
                    )

    def get_value(self) -> ModelValue:
        """Gets the model value of the current block or None if no block is selected."""
        if self.root is None:
            return None
        return self.root.model_value

    @if_tileset_loaded
    def set_value(self, behaviour: ModelValue):
        """Replaces the entry properties of the current block if one is selected."""
        self.root.blockSignals(True)  # type: ignore
        self.root.update(behaviour)  # type: ignore
        self.root.blockSignals(False)  # type: ignore
        self.tree_changed(None)
