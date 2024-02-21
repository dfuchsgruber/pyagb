"""Scene for the individual blocks."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPen
from PySide6.QtWidgets import (
    QGraphicsScene,
    QGraphicsSceneContextMenuEvent,
    QGraphicsSceneMouseEvent,
    QMenu,
    QWidget,
)
from typing_extensions import ParamSpec

from .. import history
from .child import TilesetChildWidgetMixin, if_tileset_loaded

_P = ParamSpec("_P")

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class BlocksScene(QGraphicsScene, TilesetChildWidgetMixin):
    """Scene for the individual blocks."""

    def __init__(self, tileset_widget: TilesetWidget, parent: QWidget | None=None):
        """Initializes the scene.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        TilesetChildWidgetMixin.__init__(self, tileset_widget)
        self.add_selection_rect()
        self.clipboard = None

    def add_selection_rect(self):
        """Adds the selection rectangle."""
        color = QColor.fromRgbF(1.0, 0.0, 0.0, 1.0)
        pen = QPen(color, 1.0 * self.tileset_widget.zoom_slider.value() / 10)
        self.selection_rect = self.addRect(0, 0, 0, 0, pen = pen, brush = QBrush(0))

    @if_tileset_loaded
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 16 / self.tileset_widget.zoom_slider.value()), \
            int(pos.y() * 10 / 16 / self.tileset_widget.zoom_slider.value())
        if 8 > x >= 0 and 128 > y >= 0:
            self.tileset_widget.info_label.setText(f'Block {hex(8 * y + x)}')
        else:
            self.tileset_widget.info_label.setText('')

    @if_tileset_loaded
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for moving the mouse."""
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 16 / self.tileset_widget.zoom_slider.value()), \
            int(pos.y() * 10 / 16 / self.tileset_widget.zoom_slider.value())
        if 8 > x >= 0 and 128 > y >= 0 and \
            (event.button() == Qt.MouseButton.RightButton \
            or event.button() == Qt.MouseButton.LeftButton):
            self.tileset_widget.set_current_block(8 * y + x)

    @if_tileset_loaded
    def update_selection_rect(self):
        """Updates the selection rectangle."""
        x, y = self.tileset_widget.selected_block % 8, \
            self.tileset_widget.selected_block // 8
        size = 16 * self.tileset_widget.zoom_slider.value() / 10
        x, y = int(x * size), int(y * size)
        self.selection_rect.setRect(x, y, int(size), int(size))
        self.setSceneRect(0, 0, int(8 * size), int(128 * size))


    @if_tileset_loaded
    def _paste(self, block_idx: int, paste_tiles: bool=True,
               paste_behaviour: bool=True):
        """Pastes the block data.

        Args:
            block_idx (int): The block index.
            paste_tiles (bool): Whether to paste the tiles.
            paste_behaviour (bool): Whether to paste the behaviour.
        """
        if self.clipboard is None:
            return
        block = self.tileset_widget.main_gui.get_block(block_idx)
        block_clipboard, behaviour_clipboard = self.clipboard
        self.tileset_widget.undo_stack.beginMacro('Paste Block')
        if paste_behaviour:
            self.tileset_widget.block_properties.set_value(behaviour_clipboard)
        if paste_tiles:
            for layer in range(3):
                self.tileset_widget.undo_stack.push(history.SetTiles(
                    self.tileset_widget, block_idx, layer, 0, 0,
                    block_clipboard[layer].copy(), block[layer].copy()
                ))
        self.tileset_widget.undo_stack.endMacro()


    @if_tileset_loaded
    def _clear(self, block_idx: int, clear_tiles: bool=True,
               clear_behaviour: bool=True):
        """Clears the block data.

        Args:
            block_idx (int): The block index.
            clear_tiles (bool): Whether to clear the tiles.
            clear_behaviour (bool): Whether to clear the behaviour.
        """
        block = self.tileset_widget.main_gui.get_block(block_idx)
        self.tileset_widget.undo_stack.beginMacro('Clear Block')
        if clear_behaviour:
            self.tileset_widget.clear_behaviour()
        if clear_tiles:
            for layer in range(3):
                self.tileset_widget.undo_stack.push(history.SetTiles(
                    self.tileset_widget, block_idx, layer, 0, 0,
                    np.array([self.tileset_widget.get_empty_block_tile()
                              for _ in range(4)]).reshape((2, 2)), block[layer].copy()
                ))
        self.tileset_widget.undo_stack.endMacro()


    @if_tileset_loaded
    def contextMenuEvent(self, event: QGraphicsSceneContextMenuEvent) -> None:
        """Event handler for the context menu."""
        assert self.tileset_widget.main_gui.project is not None
        pos = event.scenePos()
        x, y = int(pos.x() * 10 / 16 / self.tileset_widget.zoom_slider.value()), \
            int(pos.y() * 10 / 16 / self.tileset_widget.zoom_slider.value())
        block_idx = 8 * y + x

        # Create a context menu to capture inputs
        menu = QMenu()
        copy_action = menu.addAction("Action") # type: ignore
        menu.addSeparator()
        paste_action = menu.addAction('Paste') # type: ignore
        paste_tiles_action = menu.addAction('Paste Tiles') # type: ignore
        menu.addSeparator()
        clear_all_action = menu.addAction('Clear') # type: ignore
        clear_tiles_action = menu.addAction('Clear Tiles') # type: ignore
        if self.clipboard is None:
            paste_action.setEnabled(False)
            paste_tiles_action.setEnabled(False)
        action = menu.exec(event.screenPos())
        if action == copy_action:
                self.clipboard = self.tileset_widget.main_gui.get_block(block_idx), \
                    self.tileset_widget.block_properties.get_value()
        elif action == paste_action:
            self._paste(block_idx)
        elif action == paste_tiles_action:
            self._paste(block_idx, paste_behaviour=False)
        elif action == clear_all_action:
            self._clear(block_idx)
        elif action == clear_tiles_action:
            self._clear(block_idx, clear_behaviour=False)

