"""Properties tree for blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agb.model.type import ModelValue
from pyqtgraph.parametertree.ParameterTree import ParameterTree  # type: ignore
from PySide6.QtWidgets import (
    QWidget,
)
from PySide6.QtWidgets import QHeaderView
from typing_extensions import ParamSpec
from pymap.gui import properties

from pymap.gui.properties import type_to_parameter

from .child import TilesetChildWidgetMixin, if_tileset_loaded

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class BlockProperties(ParameterTree, TilesetChildWidgetMixin):
    """Tree to display block properties."""

    def __init__(self, tileset_widget: TilesetWidget, parent: QWidget | None = None):
        """Initializes the block properties.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)  # type: ignore
        TilesetChildWidgetMixin.__init__(self, tileset_widget)
        self.tileset_widget = tileset_widget
        self.setHeaderLabels(['Property', 'Value'])  # type: ignore
        self.header().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)  # type: ignore
        self.header().setStretchLastSection(True)  # type: ignore
        self.root = None

    @if_tileset_loaded
    def load_block(self):
        """Loads the currently displayed blocks properties."""
        self.clear()
        if (
            self.tileset_widget.main_gui.project is None
            or self.tileset_widget.main_gui.header is None
            or self.tileset_widget.main_gui.footer is None
            or self.tileset_widget.main_gui.tileset_primary is None
            or self.tileset_widget.main_gui.tileset_secondary is None
        ):
            self.root = None
        else:
            config = self.tileset_widget.main_gui.project.config['pymap'][
                'tileset_primary'
                if self.tileset_widget.selected_block_idx < 0x280
                else 'tileset_secondary'
            ]
            tileset = (
                self.tileset_widget.main_gui.tileset_primary
                if self.tileset_widget.selected_block_idx < 0x280
                else self.tileset_widget.main_gui.tileset_secondary
            )
            datatype = config['behaviour_datatype']
            behaviours = properties.get_member_by_path(
                tileset, config['behaviours_path']
            )
            assert isinstance(behaviours, list)
            behaviour = behaviours[self.tileset_widget.selected_block_idx % 0x280]
            parents = properties.get_parents_by_path(
                tileset,
                config['behaviours_path']
                + [self.tileset_widget.selected_block_idx % 0x280],
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
                parents,
            )
            self.addParameters(self.root, showTop=False)  # type: ignore
            self.root.sigTreeStateChanged.connect(self.tree_changed)  # type: ignore

    def update(self):
        """Updates all values in the tree according to the current properties."""
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary'
            if self.tileset_widget.selected_block_idx < 0x280
            else 'tileset_secondary'
        ]
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.tileset_widget.selected_block_idx < 0x280
            else self.tileset_widget.main_gui.tileset_secondary
        )
        behaviour = properties.get_member_by_path(tileset, config['behaviours_path'])[
            self.tileset_widget.selected_block_idx % 0x280
        ]
        self.root.blockSignals(True)
        self.root.update(behaviour)
        self.root.blockSignals(False)

    def tree_changed(self, changes):
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary'
            if self.tileset_widget.selected_block_idx < 0x280
            else 'tileset_secondary'
        ]
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.tileset_widget.selected_block_idx < 0x280
            else self.tileset_widget.main_gui.tileset_secondary
        )
        root = properties.get_member_by_path(tileset, config['behaviours_path'])[
            self.tileset_widget.selected_block_idx % 0x280
        ]
        diffs = DeepDiff(root, self.root.model_value)
        statements_redo = []
        statements_undo = []
        for change in ('type_changes', 'values_changed'):
            if change in diffs:
                for path in diffs[change]:
                    value_new = diffs[change][path]['new_value']
                    value_old = diffs[change][path]['old_value']
                    statements_redo.append(f"{path} = '{value_new}'")
                    statements_undo.append(f"{path} = '{value_old}'")
                    self.tileset_widget.undo_stack.push(
                        history.ChangeBlockProperty(
                            self.tileset_widget,
                            self.tileset_widget.selected_block_idx,
                            statements_redo,
                            statements_undo,
                        )
                    )

    def get_value(self) -> ModelValue:
        """Gets the model value of the current block or None if no block is selected."""
        if self.root is None:
            return None
        return self.root.model_value

    def set_value(self, behaviour: ModelValue):
        """Replaces the entrie properties of the current block if one is selected."""
        if self.model is None:
            return
        self.root.blockSignals(True)
        self.root.update(behaviour)
        self.root.blockSignals(False)
        self.tree_changed(None)
