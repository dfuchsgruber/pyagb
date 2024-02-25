"""Scene for an individual block."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
from PIL.ImageQt import ImageQt
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsSceneMouseEvent,
    QWidget,
)
from typing_extensions import ParamSpec

from .. import history
from ..render import select_blocks
from .child import TilesetChildWidgetMixin, if_tileset_loaded

_P = ParamSpec('_P')

if TYPE_CHECKING:
    from .tileset import TilesetWidget


class BlockScene(QGraphicsScene, TilesetChildWidgetMixin):
    """Scene for the current block."""

    def __init__(
        self, tileset_widget: TilesetWidget, layer: int, parent: QWidget | None = None
    ):
        """Initializes the scene.

        Args:
            tileset_widget (TilesetWidget): The tileset widget.
            layer (int): The layer of the block.
            parent (QWidget | None, optional): The parent. Defaults to None.
        """
        super().__init__(parent=parent)
        TilesetChildWidgetMixin.__init__(self, tileset_widget)
        self.layer = layer
        self.last_draw = None
        self.selection_box = None

    @if_tileset_loaded
    def update_block(self):
        """Updates the display of this block."""
        assert self.tileset_widget.main_gui.tiles is not None
        self.clear()
        block = self.tileset_widget.selected_block[self.layer]
        image = Image.new('RGBA', (16, 16))

        for (y, x), tile in np.ndenumerate(block.reshape(2, 2)):
            assert isinstance(tile, dict)
            assert isinstance(tile['palette_idx'], int)
            assert isinstance(tile['tile_idx'], int)
            tile_img: Image.Image = self.tileset_widget.main_gui.tiles[
                tile['palette_idx']
            ][tile['tile_idx']]
            if tile['horizontal_flip']:
                tile_img = tile_img.transpose(Image.FLIP_LEFT_RIGHT)
            if tile['vertical_flip']:
                tile_img = tile_img.transpose(Image.FLIP_TOP_BOTTOM)
            image.paste(tile_img, box=(8 * x, 8 * y))
        size = int(self.tileset_widget.zoom_slider.value() * 16 / 10)
        item = QGraphicsPixmapItem(
            QPixmap.fromImage(
                ImageQt(image.convert('RGB').convert('RGBA')).scaled(size, size)
            )
        )
        self.addItem(item)
        item.setAcceptHoverEvents(True)
        self.setSceneRect(0, 0, size, size)

    @if_tileset_loaded
    def update_selection_box(self):
        """Pastes the selection box to the current selection."""
        self.tileset_widget.set_selection(
            select_blocks(
                self.tileset_widget.selected_block[int(self.layer)], *self.selection_box
            )
        )

    @if_tileset_loaded
    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for hover events on the map image."""
        pos = event.scenePos()
        x, y = (
            int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
            int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
        )
        if 2 > x >= 0 and 2 > y >= 0:
            if self.last_draw is not None and self.last_draw != (x, y):
                self.last_draw = x, y
                selection_height, selection_width = self.tileset_widget.selection.shape
                # Trim the selection to fit into the 2 x 2 window
                selection = self.tileset_widget.selection[
                    : min(2 - y, selection_height), : min(2 - x, selection_width)
                ].copy()
                block = self.tileset_widget.selected_block
                # Extract the old tiles
                tiles_old = block[
                    int(self.layer),
                    y : y + selection.shape[0],
                    x : x + selection.shape[1],
                ].copy()
                self.tileset_widget.undo_stack.push(
                    history.SetTiles(
                        self.tileset_widget,
                        self.tileset_widget.selected_block,
                        int(self.layer),
                        x,
                        y,
                        selection.copy(),
                        tiles_old.copy(),
                    )
                )
            if self.selection_box is not None:
                x0, x1, y0, y1 = self.selection_box
                if x1 != x + 1 or y1 != y + 1:
                    self.selection_box = x0, x + 1, y0, y + 1
                    self.update_selection_box()
                    # Clear the selection box of the tiles widget
                    self.tileset_widget.tiles_scene.selection_box = None
                    self.tileset_widget.tiles_scene.select_tiles()

    @if_tileset_loaded
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for pressing the mouse."""
        pos = event.scenePos()
        x, y = (
            int(pos.x() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
            int(pos.y() * 10 / 8 / self.tileset_widget.zoom_slider.value()),
        )
        if 2 > x >= 0 and 2 > y >= 0:
            if event.button() == Qt.MouseButton.LeftButton:
                self.last_draw = -1, -1  # This triggers the drawing routine
                self.tileset_widget.undo_stack.beginMacro('Drawing Tiles')
                self.mouseMoveEvent(event)
            elif event.button() == Qt.MouseButton.LeftButton:
                self.selection_box = x, x + 1, y, y + 1
                self.update_selection_box()
                # Select the palette of this tile
                pal_idx = self.tileset_widget.selection[0, 0]['palette_idx']
                self.tileset_widget.tiles_palette_combobox.setCurrentIndex(pal_idx)
                # Select the tile in the tiles widget
                tile_idx = self.tileset_widget.selection[0, 0]['tile_idx']
                x, y = tile_idx % 16, tile_idx // 16
                hflip, vflip = (
                    self.tileset_widget.tiles_mirror_horizontal_checkbox.isChecked(),
                    self.tileset_widget.tiles_mirror_vertical_checkbox.isChecked(),
                )
                if hflip:
                    x = 15 - x
                if vflip:
                    y = 63 - y
                self.tileset_widget.tiles_scene.selection_box = x, x + 1, y, y + 1
                self.tileset_widget.tiles_scene.select_tiles()
                self.tileset_widget.tiles_scene.selection_box = None

                # Ensure the rect is visible
                assert self.tileset_widget
                self.tileset_widget.tiles_scene.ensure_selection_rect_visible()

    @if_tileset_loaded
    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent):
        """Event handler for releasing the mouse."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_draw = None
            self.tileset_widget.undo_stack.endMacro()
        elif event.button() == Qt.MouseButton.RightButton:
            self.selection_box = None
