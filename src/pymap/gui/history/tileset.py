"""History actions for tilesets."""

from __future__ import annotations
from copy import deepcopy

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand
from agb.model.type import ModelValue

from pymap.gui import properties, render
import numpy as np
import numpy.typing as npt

from pymap.gui.history.statement import UndoRedoStatements

if TYPE_CHECKING:
    from pymap.gui.gui import PymapGui
    from pymap.gui.tileset import TilesetWidget


class AssignGfx(QUndoCommand):
    """Class for assigning a gfx to a tileset."""

    def __init__(
        self, main_gui: PymapGui, primary: bool, label_new: str, label_old: str
    ):
        """Initializes the gfx assignment.

        Args:
            main_gui (PymapGui): reference to the main gui
            primary (bool): Whether the primary or secondary gfx is being assigned
            label_new (str): Label of the new gfx
            label_old (str): Label of the old gfx
        """
        super().__init__()
        self.main_gui = main_gui
        self.primary = primary
        self.label_new = label_new
        self.label_old = label_old

    def _assign(self, label: str):
        """Helper for assigning a label."""
        if (
            self.main_gui.project is None
            or self.main_gui.header is None
            or self.main_gui.footer is None
            or self.main_gui.tileset_primary is None
            or self.main_gui.tileset_secondary is None
        ):
            return
        if self.primary:
            self.main_gui.open_gfxs(label_primary=label)
        else:
            self.main_gui.open_gfxs(label_secondary=label)

    def redo(self):
        """Performs the tileset assignment."""
        self._assign(self.label_new)

    def undo(self):
        """Undoes the tileset assignment."""
        self._assign(self.label_old)


class ChangeBlockProperty(QUndoCommand):
    """Change a property of any block."""

    def __init__(
        self,
        tileset_widget: TilesetWidget,
        block_idx: int,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the block property change.

        Args:
            tileset_widget (TilesetWidget): reference to the tileset widget
            block_idx (int): index of the block
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__()
        self.tileset_widget = tileset_widget
        self.block_idx = block_idx
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """Executes the redo statements."""
        assert self.tileset_widget.main_gui is not None
        assert self.tileset_widget.main_gui.project is not None
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.block_idx < 0x280 else 'tileset_secondary'
        ]
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.block_idx < 0x280
            else self.tileset_widget.main_gui.tileset_secondary
        )
        blocks = properties.get_member_by_path(tileset, config['behaviours_path'])
        assert isinstance(blocks, list)
        root = blocks[self.block_idx % 0x280]  # noqa: F841
        for statement in self.statements_redo:
            exec(statement)
        self.tileset_widget.block_properties.update()

    def undo(self):
        """Executes the redo statements."""
        assert self.tileset_widget.main_gui is not None
        assert self.tileset_widget.main_gui.project is not None
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.block_idx < 0x280 else 'tileset_secondary'
        ]
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.block_idx < 0x280
            else self.tileset_widget.main_gui.tileset_secondary
        )
        blocks = properties.get_member_by_path(tileset, config['behaviours_path'])
        assert isinstance(blocks, list)
        root = [self.block_idx % 0x280]
        for statement in self.statements_undo:
            exec(statement)
        self.tileset_widget.block_properties.update()


class ChangeTilesetProperty(QUndoCommand):
    """Change a property of a tileset."""

    def __init__(
        self,
        tileset_widget: TilesetWidget,
        is_primary: bool,
        statements_redo: UndoRedoStatements,
        statements_undo: UndoRedoStatements,
    ):
        """Initializes the tileset property change.

        Args:
            tileset_widget (TilesetWidget): reference to the tileset widget
            is_primary (bool): Whether the primary or secondary tileset is being assigned
            statements_redo (list[str]): statements to be executed for redo
            statements_undo (list[str]): statements to be executed for undo
        """
        super().__init__()
        self.tileset_widget = tileset_widget
        self.is_primary = is_primary
        self.statements_redo = statements_redo
        self.statements_undo = statements_undo

    def redo(self):
        """Executes the redo statements."""
        root = (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        for statement in self.statements_redo:
            exec(statement)
        tree_widget = (
            self.tileset_widget.properties_tree_tsp
            if self.is_primary
            else self.tileset_widget.properties_tree_tss
        )
        tree_widget.update()

    def undo(self):
        """Executes the redo statements."""
        root = (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        for statement in self.statements_undo:
            exec(statement)
        tree_widget = (
            self.tileset_widget.properties_tree_tsp
            if self.is_primary
            else self.tileset_widget.properties_tree_tss
        )
        tree_widget.update()


class SetTiles(QUndoCommand):
    """Changes the tiles of any block."""

    def __init__(
        self,
        tileset_widget: TilesetWidget,
        block_idx: int,
        layer: int,
        x: int,
        y: int,
        tiles_new: npt.NDArray[np.int_],
        tiles_old: npt.NDArray[np.int_],
    ):
        """Initializes the tile change.

        Args:
            tileset_widget (TilesetWidget): reference to the tileset widget
            block_idx (int): index of the block
            layer (int): which layer to change
            x (int): x coordinate
            y (int): y coordinate
            tiles_new (npt.NDArray[np.int_]): new tiles
            tiles_old (npt.NDArray[np.int_]): old tiles
        """
        super().__init__()
        self.tileset_widget = tileset_widget
        self.block_idx = block_idx
        self.layer = layer
        self.x = x
        self.y = y
        self.tiles_new = deepcopy(tiles_new)
        self.tiles_old = deepcopy(tiles_old)

    def _set_tiles(self, tiles: npt.NDArray[np.int_]):
        """Helper method to set tiles."""
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.block_idx < 0x280
            else self.tileset_widget.main_gui.tileset_secondary
        )
        assert self.tileset_widget.main_gui.project is not None
        path = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.block_idx < 0x280 else 'tileset_secondary'
        ]['blocks_path']
        blocks = properties.get_member_by_path(tileset, path)
        assert isinstance(blocks, list)
        block = np.array(blocks[self.block_idx % 0x280]).reshape(3, 2, 2)
        block[
            self.layer,
            self.y : self.y + tiles.shape[0],
            self.x : self.x + tiles.shape[1],
        ] = tiles
        blocks[self.block_idx % 0x280] = block.flatten().tolist()
        # Update the block
        assert self.tileset_widget.main_gui.tiles is not None
        assert self.tileset_widget.main_gui.blocks is not None
        self.tileset_widget.main_gui.blocks[self.block_idx] = render.get_block(
            blocks[self.block_idx % 0x280],
            self.tileset_widget.main_gui.tiles,
        )
        self.tileset_widget.load_blocks()
        if self.layer == 0:
            self.tileset_widget.block_lower_scene.update_block()
        elif self.layer == 1:
            self.tileset_widget.block_mid_scene.update_block()
        elif self.layer == 2:
            self.tileset_widget.block_upper_scene.update_block()

    def redo(self):
        """Performs the change on the block."""
        self._set_tiles(self.tiles_new)

    def undo(self):
        """Undoes the change on the block."""
        self._set_tiles(self.tiles_old)


class SetPalette(QUndoCommand):
    """Changes a palette of a tileset."""

    def __init__(
        self,
        tileset_widget: TilesetWidget,
        pal_idx: int,
        palette_new: ModelValue,
        palette_old: ModelValue,
    ):
        """Initializes the palette change.

        Args:
            tileset_widget (TilesetWidget): reference to the tileset widget
            pal_idx (int): index of the palette
            palette_new (ModelValue): the new palette
            palette_old (ModelValue): the old palette
        """
        super().__init__()
        self.tileset_widget = tileset_widget
        self.pal_idx = pal_idx
        self.palette_new = palette_new
        self.palette_old = palette_old

    def _set_palette(self, palette: ModelValue):
        """Helper method to set a palette."""
        assert self.tileset_widget.main_gui.project is not None
        if self.pal_idx < 7:
            palettes = properties.get_member_by_path(
                self.tileset_widget.main_gui.tileset_primary,
                self.tileset_widget.main_gui.project.config['pymap']['tileset_primary'][
                    'palettes_path'
                ],
            )
        else:
            palettes = properties.get_member_by_path(
                self.tileset_widget.main_gui.tileset_secondary,
                self.tileset_widget.main_gui.project.config['pymap'][
                    'tileset_secondary'
                ]['palettes_path'],
            )
        assert isinstance(palettes, list)
        palettes[self.pal_idx % 7] = palette
        # Update tiles and blocks
        self.tileset_widget.main_gui.load_blocks()
        self.tileset_widget.reload()

    def redo(self):
        """Sets the new palette."""
        self._set_palette(self.palette_new)

    def undo(self):
        """Undoes the setting of the new palette."""
        self._set_palette(self.palette_old)


class SetTilesetAnimation(QUndoCommand):
    """Changes the animation of a tileset."""

    def __init__(
        self, tileset_widget: TilesetWidget, primary: bool, value_new: ModelValue
    ):
        """Initializes the tileset animation change.

        Args:
            tileset_widget (TilesetWidget): reference to the tileset widget
            primary (bool): Whether the primary or secondary tileset is being assigned
            value_new (ModelValue): the new value
        """
        super().__init__()
        self.tileset_widget = tileset_widget
        self.primary = primary
        self.value_new = value_new
        widget = (
            self.tileset_widget.animation_primary_line_edit
            if self.primary
            else self.tileset_widget.animation_secondary_line_edit
        )
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        assert self.tileset_widget.main_gui.project is not None
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.primary else 'tileset_secondary'
        ]
        self.value_old = properties.get_member_by_path(
            tileset, config['animation_path']
        )

    def _set_value(self, value: ModelValue):
        """Helper method to set a value to the custom LineEdit."""
        widget = (
            self.tileset_widget.animation_primary_line_edit
            if self.primary
            else self.tileset_widget.animation_secondary_line_edit
        )
        tileset = (
            self.tileset_widget.main_gui.tileset_primary
            if self.primary
            else self.tileset_widget.main_gui.tileset_secondary
        )
        assert self.tileset_widget.main_gui.project is not None
        config = self.tileset_widget.main_gui.project.config['pymap'][
            'tileset_primary' if self.primary else 'tileset_secondary'
        ]
        widget.blockSignals(True)
        widget.setText(str(value))
        widget.blockSignals(False)
        properties.set_member_by_path(tileset, str(value), config['animation_path'])

    def redo(self):
        """Sets the new value."""
        self._set_value(self.value_new)

    def undo(self):
        """Undoes the setting of the new value."""
        self._set_value(self.value_old)