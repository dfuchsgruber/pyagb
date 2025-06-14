"""History actions for blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from PySide6.QtGui import QUndoCommand

from pymap.gui.properties.utils import set_member_by_path
from pymap.gui.types import MapLayers, Tilemap

if TYPE_CHECKING:
    from pymap.gui.main.gui import PymapGui


class ResizeBuffer:
    """Buffer for resizing blocks.

    It stores the old blocks and the new dimensions.
    """

    def __init__(
        self, width_new: int, height_new: int, values_old: Tilemap, fill_value: int = 0
    ):
        """Initializes the buffer.

        Args:
            width_new (int): The new width of the buffer
            height_new (int): The new height of the buffer
            values_old (Tilemap): The old values, of shape [height, width, ...]
            fill_value (int, optional): The value to fill the buffer with.
                Defaults to 0.
        """
        self.width_old, self.height_old = values_old.shape[1], values_old.shape[0]
        self.width_new = width_new
        self.height_new = height_new
        width = max(self.width_new, self.width_old)
        height = max(self.height_new, self.height_old)
        self.buffer = np.full(
            (height, width, *values_old.shape[2:]), fill_value, dtype=int
        )
        self.buffer[: self.height_old, : self.width_old, ...] = values_old.copy()

    @property
    def old(self) -> Tilemap:
        """Returns the old blocks.

        Returns:
            Tilemap: The old blocks
        """
        return self.buffer[: self.height_old, : self.width_old, ...].copy()

    @property
    def new(self) -> Tilemap:
        """Returns the new blocks.

        Returns:
            Tilemap: The new blocks
        """
        return self.buffer[: self.height_new, : self.width_new, ...].copy()


class ResizeMap(QUndoCommand):
    """Action for resizing the map blocks."""

    def __init__(
        self,
        main_gui: PymapGui,
        height_new: int,
        width_new: int,
        values_old: Tilemap,
    ):
        """Initializes the resize map action.

        Args:
            main_gui (PymapGui): reference to the main gui
            height_new (int): new height
            width_new (int): new width
            values_old (Tilemap): old values
        """
        super().__init__(
            f'Resize map to {width_new}x{height_new}',
        )
        self.main_gui = main_gui
        self.buffer = ResizeBuffer(width_new, height_new, values_old)
        self.smart_shape_buffers = {
            name: ResizeBuffer(width_new, height_new, smart_shape.buffer)
            for name, smart_shape in main_gui.smart_shapes.items()
        }

    def _change_size(self, blocks: Tilemap, smart_shape_buffers: dict[str, Tilemap]):
        assert self.main_gui.project is not None
        self.main_gui.set_map_dimensions(blocks.shape[1], blocks.shape[0])
        set_member_by_path(
            self.main_gui.footer,
            blocks,
            self.main_gui.project.config['pymap']['footer']['map_blocks_path'],
        )
        for name, buffer in smart_shape_buffers.items():
            self.main_gui.smart_shapes[name].buffer = buffer

        self.main_gui.map_widget.load_header()

    def redo(self):
        """Resizes the map blocks."""
        self._change_size(
            self.buffer.new,
            {name: buffer.new for name, buffer in self.smart_shape_buffers.items()},
        )

    def undo(self):
        """Undoes resizing of the map blocks."""
        self._change_size(
            self.buffer.old,
            {name: buffer.old for name, buffer in self.smart_shape_buffers.items()},
        )


class ResizeBorder(QUndoCommand):
    """Action for resizing the map border."""

    def __init__(
        self,
        main_gui: PymapGui,
        height_new: int,
        width_new: int,
        values_old: Tilemap,
    ):
        """Initializes the resize map action.

        Args:
            main_gui (PymapGui): reference to the main gui
            height_new (int): new height
            width_new (int): new width
            values_old (Tilemap): old values
        """
        super().__init__(
            f'Resize border to {width_new}x{height_new}',
        )
        self.main_gui = main_gui
        self.buffer = ResizeBuffer(width_new, height_new, values_old)

    def _change_size(self, blocks: Tilemap):
        assert self.main_gui.project is not None
        self.main_gui.set_border_dimensions(blocks.shape[1], blocks.shape[0])
        set_member_by_path(
            self.main_gui.footer,
            blocks,
            self.main_gui.project.config['pymap']['footer']['border_path'],
        )
        self.main_gui.map_widget.load_header()

    def redo(self):
        """Resizes the map blocks."""
        self._change_size(self.buffer.new)

    def undo(self):
        """Undoes resizing of the map blocks."""
        self._change_size(self.buffer.old)


class SetBorder(QUndoCommand):
    """Action for setting border blocks."""

    def __init__(
        self,
        main_gui: PymapGui,
        x: int,
        y: int,
        blocks_new: Tilemap,
        blocks_old: Tilemap,
    ):
        """Initializes the set border action.

        Args:
            main_gui (PymapGui): reference to the main gui
            x (int): Which coordinates
            y (int): Which coordinates
            blocks_new (Tilemap): Which blocks to set
            blocks_old (Tilemap): The old blocks
        """
        super().__init__(
            f'Set border at {x}, {y}',
        )
        self.main_gui = main_gui
        self.x = x
        self.y = y
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks: Tilemap):
        # Helper method for setting a set of blocks
        assert self.main_gui.project is not None
        border_blocks = self.main_gui.get_borders()
        border_blocks[
            self.y : self.y + blocks.shape[0], self.x : self.x + blocks.shape[1], 0
        ] = blocks[:, :, 0]
        self.main_gui.map_widget.update_blocks()
        for y, x in np.ndindex(blocks.shape[:2]):
            self.main_gui.map_widget.blocks_tab.update_border_block_at(
                self.x + x, self.y + y
            )

    def redo(self):
        """Performs the setting of border blocks."""
        self._set_blocks(self.blocks_new)

    def undo(self):
        """Undos setting of border blocks."""
        self._set_blocks(self.blocks_old)


class SetBlocks(QUndoCommand):
    """Action for setting blocks."""

    def __init__(
        self,
        main_gui: PymapGui,
        x: int,
        y: int,
        layers: MapLayers,
        blocks_new: Tilemap,
        blocks_old: Tilemap,
    ):
        """Initializes the set blocks action.

        Args:
            main_gui (PymapGui): reference to the main gui
            x (int): at which x coordinate
            y (int): at which y coordinate
            layers (Sequence[int]): which layers to set
            blocks_new (Tilemap): Which blocks to set
            blocks_old (Tilemap): The old blocks
        """
        super().__init__(
            f'Set blocks at {x}, {y}',
        )
        self.main_gui = main_gui
        self.x = x
        self.y = y
        self.layers = layers
        self.blocks_new = blocks_new
        self.blocks_old = blocks_old

    def _set_blocks(self, blocks: Tilemap):
        # Helper method for setting a set of blocks
        footer_blocks = self.main_gui.get_map_blocks()
        footer_blocks[
            self.y : self.y + blocks.shape[0],
            self.x : self.x + blocks.shape[1],
            self.layers,
        ] = blocks[:, :, self.layers]
        self.main_gui.map_widget.update_map_at(self.x, self.y, self.layers, blocks)

    def redo(self):
        """Performs the setting of blocks."""
        self._set_blocks(self.blocks_new)

    def undo(self):
        """Undos setting of blocks."""
        self._set_blocks(self.blocks_old)


class ReplaceBlocks(QUndoCommand):
    """Action for replacing a set of blocks with another."""

    def __init__(
        self,
        main_gui: PymapGui,
        idx: Any,
        layer: int,
        value_new: Tilemap,
        value_old: Tilemap,
    ):
        """Initializes the replace blocks action.

        Args:
            main_gui (PymapGui): reference to the main gui
            idx (tuple[int, int]): which coordinates
            layer (int): which layer to replace
            value_new (Tilemap): new value
            value_old (Tilemap): old value
        """
        super().__init__('Replace blocks')
        self.main_gui = main_gui
        self.idx = idx
        self.layer = layer
        self.value_new = value_new
        self.value_old = value_old

    def redo(self):
        """Performs the flood fill."""
        map_blocks = self.main_gui.get_map_blocks()
        assert isinstance(map_blocks, np.ndarray)
        y, x = self.idx
        map_blocks[y, x, self.layer] = self.value_new
        self.main_gui.map_widget.update_map_at_indices(
            self.idx, self.value_new, layer=self.layer
        )

    def undo(self):
        """Performs the flood fill."""
        map_blocks = self.main_gui.get_map_blocks()
        assert isinstance(map_blocks, np.ndarray)
        y, x = self.idx
        map_blocks[y, x, self.layer] = self.value_old
        self.main_gui.map_widget.update_map_at_indices(
            self.idx, self.value_old, layer=self.layer
        )
