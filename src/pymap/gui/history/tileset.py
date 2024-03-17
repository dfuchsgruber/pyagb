"""History actions for tilesets."""

from __future__ import annotations
from copy import deepcopy

from typing import TYPE_CHECKING

from PySide6.QtGui import QUndoCommand
from agb.model.type import ModelValue

from pymap.gui import properties, render
import numpy as np
import numpy.typing as npt

from pymap.gui.history.statement import UndoRedoStatements, ChangeProperty

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui
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


class ChangeBlockProperty(ChangeProperty):
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
        super().__init__(statements_redo, statements_undo)
        self.tileset_widget = tileset_widget
        self.block_idx = block_idx

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        return self.tileset_widget.main_gui.get_block(self.block_idx).tolist()

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        self.tileset_widget.block_properties.update()

    def undo(self):
        """Executes the redo statements."""
        super().undo()
        self.tileset_widget.block_properties.update()


class ChangeTilesetProperty(ChangeProperty):
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
        super().__init__(statements_redo, statements_undo)
        self.tileset_widget = tileset_widget
        self.is_primary = is_primary

    def get_root(self) -> ModelValue:
        """Returns the root object of the property to change with this command."""
        return (
            self.tileset_widget.main_gui.tileset_primary
            if self.is_primary
            else self.tileset_widget.main_gui.tileset_secondary
        )

    def redo(self):
        """Executes the redo statements."""
        super().redo()
        tree_widget = (
            self.tileset_widget.properties_tree_tsp
            if self.is_primary
            else self.tileset_widget.properties_tree_tss
        )
        tree_widget.update()

    def undo(self):
        """Executes the redo statements."""
        super().undo()
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
        tiles_new: npt.NDArray[np.object_],
        tiles_old: npt.NDArray[np.object_],
    ):
        """Initializes the tile change.

        Args:
            tileset_widget (TilesetWidget): reference to the tileset widget
            block_idx (int): index of the block
            layer (int): which layer to change
            x (int): x coordinate
            y (int): y coordinate
            tiles_new (npt.NDArray[np.object_]): new tiles
            tiles_old (npt.NDArray[np.object_]): old tiles
        """
        super().__init__()
        self.tileset_widget = tileset_widget
        self.block_idx = block_idx
        self.layer = layer
        self.x = x
        self.y = y
        self.tiles_new = deepcopy(tiles_new)
        self.tiles_old = deepcopy(tiles_old)

    def _set_tiles(self, tiles: npt.NDArray[np.object_]):
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
